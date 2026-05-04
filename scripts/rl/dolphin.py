"""Dolphin subprocess wrapper for IPC inference.

One ``Dolphin`` instance == one Dolphin process == one env. Multi-env later
batches N of these (with N picked ports). Mirrors slippi-ai's ``dolphin.py``
shape, minus the menu helper / controller objects we don't need (IPC
subsumes both — see ``rl_gym_design.md`` section 4).

Phase A: launches the regular windowed ``Dolphin.exe`` (the build we
already validate against). Headless / DolphinNoGUI is a follow-up once
batched envs and savestate-on-boot are in.

Lifecycle::

    with Dolphin(iso_path=...) as d:
        env = Env(d.socket)
        ...

The context manager handles: pick port -> patch INI -> launch subprocess
-> connect with retry -> [body] -> SHUTDOWN packet -> close socket -> wait
-> restore INI.
"""

from __future__ import annotations

import configparser
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

import portpicker

from . import protocol


# Windowed Dolphin (the build we work with day-to-day). Phase A uses this
# so the user can navigate to a kickoff manually and see the AI react.
DEFAULT_DOLPHIN_EXE = (
    r"C:\Users\Brian\source\repos\Project-Citrus2\Binary\x64\Citrus Dolphin.exe"
)
DEFAULT_DOLPHIN_INI = Path(
    r"C:\Users\Brian\Documents\Dolphin Emulator\Config\Dolphin.ini"
)


class Dolphin:
    """A windowed Dolphin subprocess wired up for IPC.

    Parameters
    ----------
    iso_path
        Game ISO to boot.  Must be readable by Dolphin.
    ipc_port
        TCP port the IpcBackend listens on.  ``None`` => pick a free port.
    dolphin_exe
        Path to ``Dolphin.exe``.  Phase A defaults to the windowed build;
        switch to ``DolphinNoGUI.exe`` once headless validation lands.
    dolphin_ini
        Path to the user-config Dolphin.ini whose ``[Movie] AIIpcPort``
        we'll patch.  Restored on close.
    extra_args
        Extra CLI args to append (e.g. ``["-v", "Null", "-a", "NoAudio"]``
        once you go headless).  Empty by default — the user's INI controls
        video/audio backends.
    connect_timeout_s
        How long to retry-connect before giving up.  Strikers can take
        ~5-15s to boot on a cold cache.
    log_passthrough
        ``True`` => Dolphin's stdout/stderr go to this process's terminal
        (useful for debugging).  ``False`` => silenced.
    keep_alive
        ``True`` => on close, only tear down the IPC socket + restore INI;
        leave the Dolphin subprocess running (the user can keep poking the
        log window, then close manually).  ``False`` (default) => wait for
        graceful exit then kill if needed.
    """

    def __init__(
        self,
        iso_path: str,
        ipc_port: Optional[int] = None,
        dolphin_exe: str = DEFAULT_DOLPHIN_EXE,
        dolphin_ini: Path = DEFAULT_DOLPHIN_INI,
        extra_args: Optional[list[str]] = None,
        connect_timeout_s: float = 30.0,
        log_passthrough: bool = True,
        keep_alive: bool = False,
    ):
        self.iso_path = str(iso_path)
        self.ipc_port = ipc_port if ipc_port is not None else portpicker.pick_unused_port()
        self.dolphin_exe = dolphin_exe
        self.dolphin_ini = Path(dolphin_ini)
        self.extra_args = list(extra_args) if extra_args else []
        self.connect_timeout_s = connect_timeout_s
        self.log_passthrough = log_passthrough
        self.keep_alive = keep_alive

        self._proc: Optional[subprocess.Popen] = None
        self._sock: Optional[socket.socket] = None
        self._original_ipc_port: Optional[str] = None

    # ------------------------------------------------------------------
    # INI patching
    # ------------------------------------------------------------------
    def _patch_ini(self, port: int) -> None:
        if not self.dolphin_ini.exists():
            raise FileNotFoundError(
                f"Dolphin.ini not found at {self.dolphin_ini}; launch Dolphin "
                f"once to create it, or pass dolphin_ini=..."
            )
        cp = configparser.ConfigParser(strict=False, interpolation=None)
        cp.optionxform = str  # preserve original key casing (Dolphin is case-sensitive)
        cp.read(self.dolphin_ini, encoding="utf-8")
        if not cp.has_section("Movie"):
            cp.add_section("Movie")
        self._original_ipc_port = cp["Movie"].get("AIIpcPort", "0")
        cp["Movie"]["AIIpcPort"] = str(port)
        with open(self.dolphin_ini, "w", encoding="utf-8") as f:
            cp.write(f, space_around_delimiters=False)

    def _restore_ini(self) -> None:
        if self._original_ipc_port is None:
            return
        try:
            cp = configparser.ConfigParser(strict=False, interpolation=None)
            cp.optionxform = str
            cp.read(self.dolphin_ini, encoding="utf-8")
            if not cp.has_section("Movie"):
                cp.add_section("Movie")
            cp["Movie"]["AIIpcPort"] = self._original_ipc_port
            with open(self.dolphin_ini, "w", encoding="utf-8") as f:
                cp.write(f, space_around_delimiters=False)
        finally:
            self._original_ipc_port = None

    # ------------------------------------------------------------------
    # Subprocess + socket
    # ------------------------------------------------------------------
    def _build_cmd(self) -> list[str]:
        # Windowed Dolphin: -e auto-boots the ISO.  Video/audio backends
        # come from the user's Dolphin.ini.  We deliberately do NOT pass
        # -b (batch) so the main launcher/log window stays available for
        # debugging during Phase A; teardown is handled via SHUTDOWN
        # packet + subprocess kill on Python side.
        return [self.dolphin_exe, "-e", self.iso_path, *self.extra_args]

    def start(self) -> None:
        if not Path(self.dolphin_exe).exists():
            raise FileNotFoundError(f"Dolphin exe not found at {self.dolphin_exe}")
        if not Path(self.iso_path).exists():
            raise FileNotFoundError(f"ISO not found at {self.iso_path}")

        self._patch_ini(self.ipc_port)

        cmd = self._build_cmd()
        print(f"[Dolphin] launching on port {self.ipc_port}: {' '.join(cmd)}", flush=True)

        # Inherit stdout/stderr by default so the user can see Dolphin's
        # logs (notably "AIController: IPC active on port N").
        stdio = None if self.log_passthrough else subprocess.DEVNULL
        self._proc = subprocess.Popen(cmd, stdout=stdio, stderr=stdio)

    def connect(self) -> socket.socket:
        if self._proc is None:
            raise RuntimeError("call start() before connect()")

        deadline = time.monotonic() + self.connect_timeout_s
        last_err: Optional[Exception] = None
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            # Bail early if Dolphin already crashed.
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"Dolphin exited before IPC accept (rc={self._proc.returncode}); "
                    f"check the Dolphin logs above"
                )
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                # IMPORTANT: blocking connect, no per-attempt timeout.
                #
                # An emulated timeout (settimeout(1.0) before connect) has a
                # nasty race on loopback where the kernel completes the TCP
                # handshake at almost the same instant Python decides to
                # abort.  Symptom: C++ AcceptLoop logs "client connected"
                # for THIS attempt, then ~100ms later sees "peer closed
                # (orderly FIN)" when the timed-out Python socket gets
                # garbage-collected.  The next attempt produces a new TCP
                # connection that succeeds at the kernel level but is
                # orphaned (C++'s AcceptLoop already exited), and Python
                # blocks forever on a half-alive socket.
                #
                # Blocking connect on loopback either succeeds in
                # microseconds (port bound + listening) or fails fast with
                # ConnectionRefusedError (nothing listening yet — kernel
                # sends RST).  No race.
                s.connect(("127.0.0.1", self.ipc_port))
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock = s
                print(
                    f"[Dolphin] connected on port {self.ipc_port} "
                    f"after {attempt} attempts",
                    flush=True,
                )
                return s
            except (ConnectionRefusedError, OSError) as e:
                last_err = e
                # Explicitly close to ensure the kernel tears the fd down
                # immediately, not whenever GC decides to run.
                try:
                    s.close()
                except OSError:
                    pass
                time.sleep(0.1)

        raise TimeoutError(
            f"Could not connect to Dolphin on port {self.ipc_port} within "
            f"{self.connect_timeout_s}s; last error: {last_err}"
        )

    @property
    def socket(self) -> socket.socket:
        if self._sock is None:
            raise RuntimeError("Dolphin not connected; use the context manager or call connect()")
        return self._sock

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc else None

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    def close(self, send_shutdown: bool = True, wait_s: float = 10.0) -> None:
        # 1. Try to tell the IpcBackend we're done so it can join its threads
        #    cleanly.  Best-effort — if the socket is already broken, ignore.
        if send_shutdown and self._sock is not None:
            try:
                protocol.send_packet(self._sock, protocol.TAG_SHUTDOWN, protocol.pack_shutdown())
            except OSError:
                pass

        # 2. Close socket.
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

        # 3. Wait + kill subprocess unless we were asked to leave it running
        #    (e.g. for log inspection after the smoke test).  Note: on
        #    Windows, child processes outlive a Python parent that exits, so
        #    detaching by setting _proc=None is enough — Dolphin keeps running.
        if self._proc is not None:
            if self.keep_alive:
                print(
                    f"[Dolphin] keep_alive=True; leaving pid={self._proc.pid} running. "
                    f"Close the Dolphin window when done.",
                    flush=True,
                )
                self._proc = None
            else:
                try:
                    self._proc.wait(timeout=wait_s)
                except subprocess.TimeoutExpired:
                    print(
                        f"[Dolphin] subprocess did not exit within {wait_s}s; "
                        f"killing pid={self._proc.pid}",
                        flush=True,
                    )
                    self._proc.kill()
                    try:
                        self._proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        pass
                self._proc = None

        # 4. Restore Dolphin.ini AIIpcPort to whatever it was before we patched.
        #    Safe even with keep_alive: the running Dolphin already read the
        #    INI at boot, so restoring now only affects future launches.
        self._restore_ini()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> "Dolphin":
        try:
            self.start()
            self.connect()
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
