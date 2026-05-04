"""Single-Dolphin Environment.

Thin wrapper around a connected IPC socket. This is the surface the
rollout worker (Phase C) will eventually call N-at-a-time via a
batched env. For Phase A / B we use it directly with N=1.

Protocol cadence (matches AIController.cpp::OnFrameEnd):

  Dolphin --STATE(t)--> Python
  Dolphin <--ACTION(t)-- Python
  Dolphin --STATE(t+1)--> Python
  ...

Frame ids are monotonic per-controller, starting at 1. Python echoes the
state's frame_id in its action so the C++ side can drop stale actions
(see ``IpcBackend::HandleAction``).
"""

from __future__ import annotations

import socket
from typing import Optional

import numpy as np

from . import protocol
from .protocol import StateFrame


class Env:
    """One Dolphin = one Env. Speaks the IPC protocol."""

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._closed = False

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def recv_state(self) -> StateFrame:
        """Block on the next STATE packet from Dolphin."""
        if self._closed:
            raise RuntimeError("Env is closed")
        tag, body = protocol.recv_packet(self._sock)
        if tag != protocol.TAG_STATE:
            raise RuntimeError(f"expected STATE (0x{protocol.TAG_STATE:02x}), got 0x{tag:02x}")
        return protocol.unpack_state(body)

    def send_action(
        self,
        frame_id: int,
        btn_probs: np.ndarray,
        stick_vals: np.ndarray,
    ) -> None:
        """Send an ACTION packet without waiting for a response."""
        if self._closed:
            raise RuntimeError("Env is closed")
        body = protocol.pack_action(frame_id, btn_probs, stick_vals)
        protocol.send_packet(self._sock, protocol.TAG_ACTION, body)

    def step(
        self,
        frame_id: int,
        btn_probs: np.ndarray,
        stick_vals: np.ndarray,
    ) -> StateFrame:
        """Send action + block for next state. Convenience for synchronous play."""
        self.send_action(frame_id, btn_probs, stick_vals)
        return self.recv_state()

    def reset(self, savestate_id: int = 0) -> StateFrame:
        """Ask Dolphin to reload a savestate.

        NOTE (Phase A): the C++ side currently logs the request and does not
        actually load anything (TODO(rl-mvp) in Movie.cpp::InitAIControllerIpc).
        Once the savestate plumbing lands, the next STATE packet will have
        ``reset_context=True``.
        """
        if self._closed:
            raise RuntimeError("Env is closed")
        body = protocol.pack_reset(savestate_id)
        protocol.send_packet(self._sock, protocol.TAG_RESET, body)
        return self.recv_state()

    def close(self) -> None:
        """Stop responding. Doesn't tear down the Dolphin process — that's
        the ``Dolphin`` wrapper's job."""
        self._closed = True
