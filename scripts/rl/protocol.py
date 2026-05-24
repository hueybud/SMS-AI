"""IPC wire format for Dolphin <-> Python.

Mirrors Source/Core/Core/AIController.cpp's IpcBackend exactly. Single
source of truth — change this file and the C++ side together or things
will break in subtle binary ways.

All multi-byte fields are native little-endian (Win/x86 + Linux/x86 are
both LE; we don't run on PPC builds of Dolphin from Python).

Packet framing
--------------
Every packet on the wire is::

    u32 payload_len    (excludes itself; covers tag + body)
    u8  tag
    ... body ...

Packet tags
-----------
- ``TAG_STATE``    (0x01, Dolphin -> Python, every frame)
- ``TAG_ACTION``   (0x02, Python -> Dolphin, every frame)
- ``TAG_RESET``    (0x10, Python -> Dolphin, episode boundary)
- ``TAG_SHUTDOWN`` (0x11, Python -> Dolphin, end of session)

State packet body (after tag)::

    u32  frame_id
    u8   reset_context        (1 = first frame after a phase / episode reset)
    u8   mirror_x             (1 = AI's team attacks left)
    u8   game_phase           (raw eGameState byte: 0=pre, 1=kickoff, 2=goal,
                                3=transition, 4/5=active play.  Reward
                                shaping should gate on phase ∈ {4,5}.)
    u16  score_left
    u16  score_right
    f32  core_features[183]

Action packet body (after tag)::

    u32  frame_id             (echoed from the STATE this responds to)
    f32  btn_probs[7]         (A, B, X, Y, lob_pass, chip_shot, R; thresholded > 0.5)
    f32  stick_vals[4]        (stick_x, stick_y, cstick_x, cstick_y; range [-1,1])

Reset packet body::

    u32  savestate_id

Shutdown packet body: empty.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Must match AIModelDims::CORE_FEATURE_DIM in AIController.h.
CORE_FEATURE_DIM = 183
BUTTON_DIM = 7
STICK_DIM = 4

TAG_STATE = 0x01
TAG_ACTION = 0x02
TAG_RESET = 0x10
TAG_SHUTDOWN = 0x11
# Pause / resume the live emulator while the trainer runs a PPO update.
# C++ side routes these through Core::QueueHostJob so they're safe to
# trigger from the receiver thread.  See AIController.cpp::ReceiverLoop.
TAG_PAUSE = 0x12
TAG_RESUME = 0x13

# State header format (after the 1-byte tag).  IBBBHH = u32 frame_id,
# u8 reset_context, u8 mirror_x, u8 game_phase, u16 score_left, u16 score_right.
_STATE_HEADER_FMT = "<IBBBHH"
_STATE_HEADER_SIZE = struct.calcsize(_STATE_HEADER_FMT)
_STATE_FEAT_BYTES = CORE_FEATURE_DIM * 4
_STATE_BODY_SIZE = _STATE_HEADER_SIZE + _STATE_FEAT_BYTES

# Action body format (after the 1-byte tag): u32 frame_id, 7 floats, 4 floats.
_ACTION_BODY_FMT = "<I" + "f" * BUTTON_DIM + "f" * STICK_DIM
_ACTION_BODY_SIZE = struct.calcsize(_ACTION_BODY_FMT)


@dataclass
class StateFrame:
    """A single STATE packet decoded into Python-friendly types."""

    frame_id: int
    reset_context: bool
    mirror_x: bool
    game_phase: int  # raw eGameState; reward shaping should gate on {4,5}
    score_left: int
    score_right: int
    core_features: np.ndarray  # shape (183,), dtype=float32


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes from ``sock`` or raise ConnectionError."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError(
                f"socket closed by peer with {remaining}/{n} bytes still to read"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_packet(sock: socket.socket) -> Tuple[int, bytes]:
    """Read one length-prefixed packet from ``sock``.

    Returns ``(tag, body)`` where ``body`` excludes the tag byte.
    Raises ``ConnectionError`` on socket close, ``ValueError`` on a
    malformed length prefix.
    """
    raw_len = _recv_exact(sock, 4)
    (payload_len,) = struct.unpack("<I", raw_len)
    if payload_len == 0 or payload_len > (1 << 20):
        raise ValueError(f"bogus payload_len={payload_len}")
    payload = _recv_exact(sock, payload_len)
    tag = payload[0]
    body = payload[1:]
    return tag, body


def send_packet(sock: socket.socket, tag: int, body: bytes) -> None:
    """Send one length-prefixed packet over ``sock``."""
    payload_len = 1 + len(body)
    header = struct.pack("<IB", payload_len, tag & 0xFF)
    sock.sendall(header + body)


def unpack_state(body: bytes) -> StateFrame:
    """Decode a STATE packet body (without tag) into a StateFrame."""
    if len(body) != _STATE_BODY_SIZE:
        raise ValueError(
            f"STATE body wrong size: got {len(body)}, expected {_STATE_BODY_SIZE} "
            f"(header {_STATE_HEADER_SIZE} + features {_STATE_FEAT_BYTES})"
        )
    frame_id, reset_b, mirror_b, game_phase, score_l, score_r = struct.unpack_from(
        _STATE_HEADER_FMT, body, 0
    )
    feats = np.frombuffer(
        body, dtype=np.float32, count=CORE_FEATURE_DIM, offset=_STATE_HEADER_SIZE
    ).copy()  # copy so the caller can hold it past this socket read's lifetime
    return StateFrame(
        frame_id=frame_id,
        reset_context=bool(reset_b),
        mirror_x=bool(mirror_b),
        game_phase=int(game_phase),
        score_left=score_l,
        score_right=score_r,
        core_features=feats,
    )


def pack_action(
    frame_id: int,
    btn_probs: np.ndarray,
    stick_vals: np.ndarray,
) -> bytes:
    """Pack an ACTION packet body (without tag).

    ``btn_probs`` shape (7,) and ``stick_vals`` shape (4,) — both float32-able.
    """
    if btn_probs.shape != (BUTTON_DIM,):
        raise ValueError(f"btn_probs shape {btn_probs.shape} != ({BUTTON_DIM},)")
    if stick_vals.shape != (STICK_DIM,):
        raise ValueError(f"stick_vals shape {stick_vals.shape} != ({STICK_DIM},)")
    btn = np.asarray(btn_probs, dtype=np.float32)
    stk = np.asarray(stick_vals, dtype=np.float32)
    return struct.pack(_ACTION_BODY_FMT, int(frame_id), *btn.tolist(), *stk.tolist())


def pack_reset(savestate_id: int = 0) -> bytes:
    """Pack a RESET packet body (without tag)."""
    return struct.pack("<I", int(savestate_id))


def pack_shutdown() -> bytes:
    """Pack a SHUTDOWN packet body (empty)."""
    return b""


def send_pause(sock: socket.socket) -> None:
    """Ask the emulator to pause until a matching ``send_resume()``.

    The pause takes effect a frame or two later (host-thread-dispatched);
    don't expect immediate freeze.  Safe to call repeatedly — extra
    PAUSEs while already paused are no-ops on the C++ side.
    """
    send_packet(sock, TAG_PAUSE, b"")


def send_resume(sock: socket.socket) -> None:
    """Resume the emulator after a ``send_pause()``."""
    send_packet(sock, TAG_RESUME, b"")
