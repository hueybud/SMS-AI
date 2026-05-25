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

import select
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
        # Running total of STATE packets drained as stale.  train.py reads
        # this between rollouts to log per-rollout drain counts (= number
        # of game frames not represented in the trajectory).
        self.drained_states: int = 0

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
        drain: bool = True,
    ) -> tuple[StateFrame, list[StateFrame]]:
        """Send action + block for next state. Convenience for synchronous play.

        With ``drain=True`` (default), after reading the next STATE we
        peek the socket and consume any additional STATE packets that
        piled up while we were thinking, keeping only the freshest.

        Why drain: the C++ side produces STATE packets at 60 fps
        regardless of Python's read rate.  RL inference is ~17ms vs
        16.7ms/frame budget — Python loses ~3% per frame, the OS recv
        buffer accumulates stale packets monotonically across batches
        (pause/resume freezes both sides but does NOT drain the
        backlog), and once the backlog crosses the ~60-frame staleness
        filter window in HandleAction every action gets rejected.
        Confirmed in 2026-05-04 run: accepted plateaued, dropped
        climbed through 1680+ with gap growing 116 → 129.

        Trade-off: occasional act_max spikes (32ms) cause the game to
        produce ~2 STATEs during one inference; drain discards the
        intermediate frame so the trajectory has occasional 1-frame
        skips.  Slop is bounded; slippi-ai's gym makes the same
        compromise.  The strict fix is synchronous pacing on the C++
        side (game waits for ACTION before advancing).
        """
        if self._closed:
            raise RuntimeError("Env is closed")
        self.send_action(frame_id, btn_probs, stick_vals)
        state = self.recv_state()
        drained: list[StateFrame] = []
        if drain:
            while True:
                ready, _, _ = select.select([self._sock], [], [], 0)
                if not ready:
                    break
                tag, body = protocol.recv_packet(self._sock)
                if tag != protocol.TAG_STATE:
                    raise RuntimeError(
                        f"unexpected tag during drain: 0x{tag:02x}"
                    )
                drained.append(state)   # previous state is now an intermediate
                state = protocol.unpack_state(body)
                self.drained_states += 1
        return state, drained

    def pause(self) -> None:
        """Ask Dolphin to pause the live emulator until ``resume()``.

        Used to freeze gameplay across a slow PPO update so opponents
        can't score against a held-action AI during the gap.  See
        AIController.cpp::ReceiverLoop for the C++ side.
        """
        if self._closed:
            raise RuntimeError("Env is closed")
        protocol.send_pause(self._sock)

    def resume(self) -> None:
        """Resume after a ``pause()``."""
        if self._closed:
            raise RuntimeError("Env is closed")
        protocol.send_resume(self._sock)

    def reset(self, savestate_id: int = 0, timeout_frames: int = 240) -> StateFrame:
        """Ask Dolphin to reload a savestate, blocking until the load lands.

        ``savestate_id`` maps to a .sav file next to the Dolphin binary on
        the C++ side (see Movie.cpp::InitAIControllerIpc).  Currently:
        0=rl_palace.sav, 1=rl_underground.sav, 2=rl_battle_dome.sav.

        The load happens on Dolphin's host thread via QueueHostJob, which
        means an unknown number of pre-load STATE packets (potentially with
        ``match_end=True``) can arrive before the savestate fires.  We
        drain those packets until we see ``reset_context=True``, which
        AIController sets on phase transitions — the post-load frame
        transitions from post-match phase (0/3) into kickoff/active-play
        (1/4/5), so that's our signal that the load completed.
        """
        if self._closed:
            raise RuntimeError("Env is closed")
        body = protocol.pack_reset(savestate_id)
        protocol.send_packet(self._sock, protocol.TAG_RESET, body)
        drained = 0
        while drained < timeout_frames:
            state = self.recv_state()
            if state.reset_context:
                return state
            drained += 1
        raise TimeoutError(
            f"savestate {savestate_id} did not produce a reset_context=True "
            f"STATE within {timeout_frames} frames"
        )

    def close(self) -> None:
        """Stop responding. Doesn't tear down the Dolphin process — that's
        the ``Dolphin`` wrapper's job."""
        self._closed = True
