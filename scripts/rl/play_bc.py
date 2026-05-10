"""Phase B: play Strikers with the PyTorch BC agent over IPC.

Equivalence test for the IPC plumbing: if BC-over-IPC plays the way the
in-process LocalOnnxBackend does, the wire format + KV cache + prev_labels
+ norm stats handling are all wired correctly, and Phase C (PPO with the
same Python policy) can build on top.

Run::

    cd "C:/Users/Brian/Documents/SMS AI/scripts"
    py -3.9 -m rl.play_bc                       # default checkpoint, --keep-alive off
    py -3.9 -m rl.play_bc --keep-alive          # leave Dolphin running on Ctrl+C
    py -3.9 -m rl.play_bc --device cuda         # if PyTorch CUDA is set up

Eyeball test: navigate to a kickoff in the Dolphin window, then watch the
AI-controlled player.  It should pick up the ball, dribble, pass, and look
generally competent — same vibe as the local-ONNX BC.

Press Ctrl+C to exit cleanly.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time

from .bc_agent import BCAgent
from .dolphin import DEFAULT_DOLPHIN_EXE, Dolphin
from .env import Env

DEFAULT_ISO = (
    r"C:\Users\Brian\Downloads\Super Mario Strikers (USA)\Super Mario Strikers (USA).iso"
)
DEFAULT_CHECKPOINT = (
    r"C:\Users\Brian\Documents\SMS AI\runs\transformer_v3\best_model.pt"
)
DEFAULT_NORM_STATS = (
    r"C:\Users\Brian\Documents\SMS AI\runs\transformer_v3\norm_stats.npz"
)


def _wait_for_first_state(env: Env, sock: socket.socket, timeout_s: float):
    """Poll-wait for the first STATE packet with periodic heartbeats so we
    can spot dead sockets and tell the user how long they've been navigating
    menus.  Same shape as ``smoke_test._wait``-style logic; cheap to dupe."""
    deadline = time.monotonic() + timeout_s
    t0 = time.monotonic()
    while time.monotonic() < deadline:
        sock.settimeout(2.0)
        try:
            return env.recv_state()
        except socket.timeout:
            err = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            elapsed = int(time.monotonic() - t0)
            if err != 0:
                raise RuntimeError(
                    f"socket has SO_ERROR={err} after {elapsed}s — connection "
                    f"died (firewall/AV?)"
                )
            print(
                f"[play_bc] still waiting for first STATE... {elapsed}s elapsed; "
                f"navigate to a kickoff if needed",
                flush=True,
            )
    raise TimeoutError(
        f"no STATE packet within {timeout_s}s; AI port never entered phase 1/4/5"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iso", default=DEFAULT_ISO)
    p.add_argument("--exe", default=DEFAULT_DOLPHIN_EXE)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--norm-stats", default=DEFAULT_NORM_STATS)
    p.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="PyTorch device for the policy forward pass",
    )
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--first-state-timeout-s", type=float, default=600.0)
    p.add_argument("--log-every-s", type=float, default=5.0)
    p.add_argument(
        "--keep-alive",
        action="store_true",
        help="leave Dolphin running on exit so you can inspect the log",
    )
    args = p.parse_args()

    print(f"[play_bc] iso={args.iso}")
    print(f"[play_bc] checkpoint={args.checkpoint}")
    print(f"[play_bc] norm_stats={args.norm_stats}")
    print(f"[play_bc] device={args.device}")

    # Construct the agent BEFORE Dolphin so checkpoint-load failures don't
    # leave a Dolphin process orphaned with no client to connect.
    agent = BCAgent(args.checkpoint, args.norm_stats, device=args.device)

    with Dolphin(
        iso_path=args.iso,
        ipc_port=args.port,
        dolphin_exe=args.exe,
        keep_alive=args.keep_alive,
    ) as dolphin:
        env = Env(dolphin.socket)
        sock = dolphin.socket

        print("[play_bc] waiting for first STATE packet (heartbeat every 2s)...")
        try:
            state = _wait_for_first_state(env, sock, args.first_state_timeout_s)
        except (TimeoutError, RuntimeError, OSError) as e:
            print(f"[play_bc] FAIL: {e}", file=sys.stderr)
            return 1
        sock.settimeout(None)

        print(
            f"[play_bc] first STATE: frame_id={state.frame_id} "
            f"reset={state.reset_context} mirror={state.mirror_x} "
            f"score=({state.score_left},{state.score_right})"
        )

        n_frames = 0
        n_resets = 0
        if state.reset_context:
            n_resets += 1
        loop_t0 = time.monotonic()
        last_log_t = loop_t0

        try:
            while True:
                btn, stick = agent.act(state)
                state, _ = env.step(state.frame_id, btn, stick)
                n_frames += 1
                if state.reset_context:
                    n_resets += 1
                    print(
                        f"[play_bc] reset #{n_resets} at frame_id={state.frame_id}",
                        flush=True,
                    )

                now = time.monotonic()
                if now - last_log_t >= args.log_every_s:
                    fps = n_frames / (now - loop_t0)
                    bx = float(state.core_features[0])
                    by = float(state.core_features[1])
                    print(
                        f"[play_bc] frames={n_frames:6d} fps={fps:5.1f} "
                        f"score=({state.score_left},{state.score_right}) "
                        f"ball=({bx:+.2f},{by:+.2f}) resets={n_resets} "
                        f"btn={btn.astype(int).tolist()} "
                        f"stk=[{stick[0]:+.2f},{stick[1]:+.2f},"
                        f"{stick[2]:+.2f},{stick[3]:+.2f}]",
                        flush=True,
                    )
                    last_log_t = now
        except KeyboardInterrupt:
            elapsed = time.monotonic() - loop_t0
            fps = n_frames / elapsed if elapsed > 0 else 0.0
            print(
                f"\n[play_bc] interrupted: played {n_frames} frames in "
                f"{elapsed:.1f}s ({fps:.1f} fps), {n_resets} resets",
                flush=True,
            )
        except (ConnectionError, OSError) as e:
            print(
                f"[play_bc] socket died after {n_frames} frames: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
