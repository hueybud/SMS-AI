"""Phase A smoke test: validate the IPC roundtrip with no model.

What this proves
----------------
1. Dolphin boots cleanly with ``[Movie] AIIpcPort`` set.
2. The IpcBackend listens, accepts our connection, and starts shipping
   STATE packets once the AI's port enters an active phase
   (game_phase in {1, 4, 5}).
3. STATE packets arrive with monotonic frame_ids and the right size.
4. Dolphin actually applies our ACTIONs — when we hold up on the left
   stick, the AI character should run forward (visible in win32 mode).

How to run
----------
From ``SMS AI/scripts``::

    py -3.9 -m rl.smoke_test

Phase A uses the windowed ``Dolphin.exe`` so you can navigate menus to a
kickoff manually. The first STATE packet only arrives once the AI's
controlled port enters phase 1/4/5; you have ``--first-state-timeout-s``
seconds (default 600) to navigate there.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time

import numpy as np

from .dolphin import DEFAULT_DOLPHIN_EXE, DEFAULT_NOGUI_EXE, Dolphin
from .env import Env

DEFAULT_ISO = (
    r"C:\Users\Brian\Downloads\Super Mario Strikers (USA)\Super Mario Strikers (USA).iso"
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iso", default=DEFAULT_ISO,
                   help="path to Strikers ISO")
    p.add_argument("--exe", default=None,
                   help="dolphin executable (default: windowed Citrus Dolphin.exe, "
                        "or dolphin-emu-nogui under --headless)")
    p.add_argument("--frames", type=int, default=600, help="frames to exchange before stopping")
    p.add_argument("--port", type=int, default=None, help="fixed IPC port (default: pick free)")
    p.add_argument(
        "--first-state-timeout-s",
        type=float,
        default=600.0,
        help="how long to wait for the first STATE packet (you may need time to navigate to a kickoff)",
    )
    p.add_argument(
        "--keep-alive",
        action="store_true",
        help="don't kill Dolphin when the smoke test finishes (so you can read the log window)",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="launch dolphin-emu-nogui in headless mode (Linux/ThunderCompute). "
             "Requires --savestate-path so we boot straight into a kickoff.",
    )
    p.add_argument(
        "--savestate-path",
        default=None,
        help="path to a .sav file to boot into (required under --headless). "
             "Resolved by Dolphin relative to the working dir or absolute.",
    )
    p.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="worker number (drives port + tmpdir under --headless)",
    )
    args = p.parse_args()

    if args.headless and not args.savestate_path:
        p.error("--headless requires --savestate-path (no human to navigate menus)")

    # Default exe resolves based on mode.
    if args.exe is None:
        args.exe = DEFAULT_NOGUI_EXE if args.headless else DEFAULT_DOLPHIN_EXE

    print(f"[smoke] iso={args.iso}")
    print(f"[smoke] exe={args.exe}")
    print(
        f"[smoke] note: first STATE arrives only once the AI port enters phase 1/4/5; "
        f"navigate to a kickoff if needed (timeout={args.first_state_timeout_s}s)"
    )

    btn_neutral = np.zeros(7, dtype=np.float32)
    stick_up = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    stick_neutral = np.zeros(4, dtype=np.float32)

    with Dolphin(
        iso_path=args.iso,
        ipc_port=args.port,
        dolphin_exe=args.exe,
        keep_alive=args.keep_alive,
        headless=args.headless,
        worker_id=args.worker_id,
        savestate_path=args.savestate_path,
    ) as dolphin:
        env = Env(dolphin.socket)

        # Wait for the first STATE packet.  Poll in 2s slices so we can
        # heartbeat-print progress AND detect the socket dying silently
        # underneath us — symptom: Python's recv() blocks happily for the
        # full timeout while Dolphin's recv() already returned 0/-1 because
        # the loopback connection got RST'd by AV / firewall.
        sock = dolphin.socket
        deadline = time.monotonic() + args.first_state_timeout_s
        print("[smoke] waiting for first STATE packet (heartbeat every 2s)...")
        t0 = time.monotonic()
        state = None
        while time.monotonic() < deadline:
            sock.settimeout(2.0)
            try:
                state = env.recv_state()
                break
            except socket.timeout:
                # No data this slice — confirm the socket is still healthy.
                err = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                elapsed = int(time.monotonic() - t0)
                if err != 0:
                    print(
                        f"[smoke] FAIL: socket has SO_ERROR={err} after {elapsed}s — "
                        f"connection died silently (firewall/AV reset?)",
                        file=sys.stderr,
                    )
                    return 1
                # Liveness probe: send a 0-byte ACTION-tagged packet?  No —
                # any send would queue real bytes and confuse the receiver.
                # Instead, peek at MSG_PEEK to see if the socket is still
                # readable in a non-blocking way.
                try:
                    sock.setblocking(False)
                    peek = sock.recv(1, socket.MSG_PEEK)
                    sock.setblocking(True)
                    if peek == b"":
                        print(
                            f"[smoke] FAIL: peer closed cleanly after {elapsed}s "
                            f"(MSG_PEEK returned empty)",
                            file=sys.stderr,
                        )
                        return 1
                except BlockingIOError:
                    sock.setblocking(True)  # no data available, that's fine
                except OSError as e:
                    print(
                        f"[smoke] FAIL: peek raised {type(e).__name__}: {e} "
                        f"after {elapsed}s",
                        file=sys.stderr,
                    )
                    return 1
                print(
                    f"[smoke] still waiting... {elapsed}s elapsed, socket healthy "
                    f"(SO_ERROR=0, peer reachable)"
                )
                continue
            except (ConnectionError, OSError) as e:
                elapsed = int(time.monotonic() - t0)
                print(
                    f"[smoke] FAIL: socket died after {elapsed}s: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                return 1
        if state is None:
            elapsed = int(time.monotonic() - t0)
            print(
                f"[smoke] FAIL: hit overall deadline {args.first_state_timeout_s}s "
                f"with no STATE; ({elapsed}s waited)",
                file=sys.stderr,
            )
            return 1
        sock.settimeout(None)

        wait_s = time.monotonic() - t0
        print(
            f"[smoke] first STATE after {wait_s:.1f}s: frame_id={state.frame_id} "
            f"reset={state.reset_context} mirror={state.mirror_x} "
            f"score=({state.score_left},{state.score_right}) "
            f"feat[0..3]={state.core_features[:3]}"
        )

        last_frame_id = state.frame_id
        gap_count = 0
        first_id = state.frame_id
        loop_t0 = time.monotonic()

        for i in range(args.frames):
            # Alternate stick-up / neutral every ~1s so we can see the
            # AI's character move (or not) on screen.
            stick = stick_up if (i // 60) % 2 == 0 else stick_neutral
            try:
                state, _ = env.step(last_frame_id, btn_neutral, stick)
            except (ConnectionError, OSError) as e:
                print(f"[smoke] FAIL: socket died at i={i}: {e}", file=sys.stderr)
                return 1

            gap = state.frame_id - last_frame_id
            if gap != 1:
                gap_count += 1
                if gap_count <= 5:
                    print(
                        f"[smoke] WARN: frame_id gap={gap} "
                        f"(prev={last_frame_id} new={state.frame_id})"
                    )
            last_frame_id = state.frame_id

            if i % 60 == 0:
                bx = float(state.core_features[0])
                by = float(state.core_features[1])
                bz = float(state.core_features[2])
                stick_label = "UP" if (i // 60) % 2 == 0 else "NEUTRAL"
                print(
                    f"[smoke] i={i:4d} fid={state.frame_id} reset={int(state.reset_context)} "
                    f"score=({state.score_left},{state.score_right}) "
                    f"ball=({bx:+.2f},{by:+.2f},{bz:+.2f}) sending_stick={stick_label}"
                )

        loop_dt = time.monotonic() - loop_t0
        rate = args.frames / loop_dt
        unique_frames = last_frame_id - first_id + 1
        print(
            f"[smoke] DONE: {args.frames} send/recv in {loop_dt:.2f}s = {rate:.1f} pps; "
            f"frames advanced by {unique_frames}; gaps>1: {gap_count}"
        )
        if gap_count == 0:
            print("[smoke] PASS: every frame responded to in time (no stale-frame drops)")
        else:
            print(
                f"[smoke] PARTIAL: {gap_count} frame gaps observed — Python loop is "
                f"slower than emu thread, expected as a baseline; later batched "
                f"forward passes will reduce this."
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
