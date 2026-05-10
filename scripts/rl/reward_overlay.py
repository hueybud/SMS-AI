"""Live reward-component overlay window.

Listens on a localhost UDP port for JSON datagrams emitted by
``rl.train``'s reward event sink and displays each event in a small
always-on-top text window so it can be parked next to the running
Dolphin screen for live fact-checking.

Layout::

    Recent events (last 3s, newest first)
    -----------------------------------------------------
      +0.050  SHOT               2026-05-06 14:32:11.412
      +0.150  GAIN_ATK_THIRD     2026-05-06 14:32:09.001
      -0.100  LOSS_MID_THIRD     2026-05-06 14:32:08.890
      ...

    Cumulative this session
    -----------------------------------------------------
      GOAL_FOR        x  3   total +3.000
      GAIN_ATK_THIRD  x 11   total +1.650
      LOSS_MID_THIRD  x  9   total -0.900
      ...

Run::

    py -3.9 -m rl.reward_overlay --port 9876

Then start training with::

    py -3.9 -m rl.train --reward-overlay-port 9876
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
import tkinter as tk
from collections import defaultdict, deque
from datetime import datetime


RECENT_WINDOW_SECS = 5.0
RECENT_MAX_ROWS    = 30
REFRESH_MS         = 100


class OverlayState:
    """Thread-safe shared state between the UDP listener and the Tk loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.recent: deque = deque()  # (ts, name, value)
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, name: str, value: float) -> None:
        ts = time.time()
        with self._lock:
            self.recent.append((ts, name, value))
            self.totals[name] += value
            self.counts[name] += 1

    def snapshot(self) -> tuple[list, list]:
        cutoff = time.time() - RECENT_WINDOW_SECS
        with self._lock:
            while self.recent and self.recent[0][0] < cutoff:
                self.recent.popleft()
            recent = list(reversed(list(self.recent)))[:RECENT_MAX_ROWS]
            cumulative = sorted(
                ((n, self.counts[n], self.totals[n]) for n in self.totals),
                key=lambda r: -abs(r[2]),
            )
        return recent, cumulative


def listener_thread(state: OverlayState, port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    while True:
        try:
            data, _ = sock.recvfrom(4096)
            payload = json.loads(data.decode("utf-8"))
            state.add(str(payload["comp"]), float(payload["val"]))
        except (OSError, ValueError, KeyError):
            continue


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=9876,
                   help="UDP port to listen on (matches --reward-overlay-port "
                        "passed to rl.train)")
    p.add_argument("--alpha", type=float, default=0.92,
                   help="Window opacity 0.0-1.0 (default 0.92)")
    args = p.parse_args()

    state = OverlayState()
    t = threading.Thread(target=listener_thread, args=(state, args.port), daemon=True)
    t.start()

    root = tk.Tk()
    root.title(f"Reward overlay :{args.port}")
    root.attributes("-topmost", True)
    root.attributes("-alpha", args.alpha)
    root.geometry("420x520+40+40")
    root.configure(bg="#101418")

    text = tk.Text(
        root,
        bg="#101418", fg="#e6e6e6",
        font=("Consolas", 10),
        bd=0, padx=8, pady=8,
        wrap="none",
    )
    text.pack(fill="both", expand=True)
    text.tag_configure("pos",    foreground="#9aff9a")
    text.tag_configure("neg",    foreground="#ff9a9a")
    text.tag_configure("header", foreground="#7fb3ff")
    text.tag_configure("dim",    foreground="#7a8290")

    def render() -> None:
        recent, cumulative = state.snapshot()
        text.config(state="normal")
        text.delete("1.0", "end")

        text.insert("end", f"Recent events (last {RECENT_WINDOW_SECS:.0f}s, newest↑)\n", "header")
        text.insert("end", "-" * 48 + "\n", "dim")
        if not recent:
            text.insert("end", "  (none yet)\n", "dim")
        else:
            for ts, name, value in recent:
                tag = "pos" if value >= 0 else "neg"
                stamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]
                text.insert("end", f"  {value:+.4f}  {name:<18s} {stamp}\n", tag)

        text.insert("end", "\n")
        text.insert("end", "Cumulative this session (|sum|↓)\n", "header")
        text.insert("end", "-" * 48 + "\n", "dim")
        if not cumulative:
            text.insert("end", "  (none yet)\n", "dim")
        else:
            for name, count, total in cumulative:
                tag = "pos" if total >= 0 else "neg"
                text.insert(
                    "end",
                    f"  {name:<18s} x{count:4d}   total {total:+.4f}\n",
                    tag,
                )

        text.config(state="disabled")
        root.after(REFRESH_MS, render)

    render()
    root.mainloop()


if __name__ == "__main__":
    main()
