"""Dolphin subprocess wrapper for IPC inference.

One ``Dolphin`` instance == one Dolphin process == one env. Multi-env scale-out
on ThunderCompute uses ``BatchedEnvironment`` (rl/batch_env.py) which holds
N of these in parallel, each with a distinct ``worker_id`` driving its port,
user-dir, and log file.

Two run modes share the same surface:

* ``headless=False`` (default) — launches the windowed ``Citrus Dolphin.exe``
  and patches a single shared ``Dolphin.ini``.  Used for Windows BC validation
  and single-env eyeballing during development.  Original ``[Movie] AIIpcPort``
  is restored on close.
* ``headless=True`` — launches ``dolphin-emu-nogui -p headless`` with a
  per-worker ``-u <tmpdir>`` so each worker writes a fresh INI (null video,
  null audio, unlimited emulation speed, IPC port wired).  Boots straight
  into a savestate via ``--save_state=...``, skipping menu nav.  No shared
  INI to fight over → safe for N parallel processes.

Lifecycle::

    with Dolphin(iso_path=..., headless=True, worker_id=0,
                 savestate_path="rl_palace.sav") as d:
        env = Env(d.socket)
        ...

The context manager handles: pick port -> write/patch INI -> launch
subprocess -> connect with retry -> [body] -> SHUTDOWN packet -> close
socket -> wait -> restore INI (windowed only).
"""

from __future__ import annotations

import configparser
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import portpicker

from . import protocol


# Windowed Dolphin (Windows BC validation default).
DEFAULT_DOLPHIN_EXE = (
    r"C:\Users\Brian\source\repos\Project-Citrus2\Binary\x64\Citrus Dolphin.exe"
)
DEFAULT_DOLPHIN_INI = Path(
    r"C:\Users\Brian\Documents\Dolphin Emulator\Config\Dolphin.ini"
)

# Headless Linux defaults — resolved against $HOME so the same script works
# under either the `ubuntu` or `root` ThunderCompute login.  The repo is
# usually cloned to ~/Project-Citrus (no "2" suffix — that's the local
# Windows checkout's name, the remote is hueybud/Project-Citrus).  Override
# via --exe / dolphin_exe= when the build lives elsewhere.
DEFAULT_NOGUI_EXE = os.path.expanduser(
    "~/Project-Citrus/build/Binaries/dolphin-emu-nogui"
)

# Base port for deterministic multi-env allocation.  worker_id N gets
# port BASE + N.  Pinning to a range (vs portpicker's roaming) prevents
# collisions between simultaneous training jobs on the same instance
# (ThunderCompute docs: "container network is NOT isolated").
HEADLESS_BASE_PORT = 51000


class Dolphin:
    """One Dolphin subprocess wired up for IPC.

    Parameters
    ----------
    iso_path
        Game ISO to boot.  Must be readable by Dolphin.
    ipc_port
        TCP port the IpcBackend listens on.  ``None`` => pick.  Picking
        rules: in headless mode, ``HEADLESS_BASE_PORT + worker_id``
        (deterministic, range-pinned).  In windowed mode, ``portpicker``.
    dolphin_exe
        Path to the Dolphin binary.  ``None`` => DEFAULT_DOLPHIN_EXE
        (windowed) or DEFAULT_NOGUI_EXE (headless).
    dolphin_ini
        Path to the shared ``Dolphin.ini`` to patch (windowed mode only).
        Ignored in headless mode — each worker writes its own INI to
        ``user_dir/Config/Dolphin.ini``.
    extra_args
        Extra CLI args appended to the launch command.  Headless mode
        already supplies ``-p headless``, ``-e <iso>``, ``-u <tmpdir>``,
        and ``--save_state=...`` — don't duplicate those.
    connect_timeout_s
        How long to retry-connect.  Strikers boots in ~5-15s windowed;
        headless with a savestate-boot is similar (the savestate applies
        before the game finishes initializing video/audio, but the IPC
        listener binds during Movie::Init regardless).
    log_passthrough
        ``True`` => inherit stdout/stderr (windowed default).
        ``False`` => silenced (or redirected if ``log_file`` is set).
    log_file
        Per-worker stdout/stderr destination (headless multi-env).
        Overrides ``log_passthrough``.  Parent dir must exist.
    keep_alive
        Leave the Dolphin subprocess running on close (used for log
        inspection after a smoke test).
    headless
        ``True`` => nogui + per-worker user dir + boot-via-savestate.
    worker_id
        0-indexed worker number.  Drives port and user-dir choice in
        headless mode; ignored in windowed mode.
    savestate_path
        Path to a .sav file to boot into via ``--save_state=...``.
        Required in headless mode (no human to navigate menus).
        Optional in windowed mode (you'll navigate manually).
    user_dir
        Override the per-worker user directory (headless mode).
        ``None`` => ``/tmp/citrus_env_<worker_id>``.  The directory is
        created if missing; its INI is overwritten every launch.
    ai_controlled_port
        GC port the AI controls (passed through to ``[Movie]
        AIControlledPort``).  Default 0 matches our setup (AI on P1).
    ai_mirror_x
        ``[Movie] AIMirrorX`` value.  Default False matches our captured
        savestates (Daisy/Toad on the LEFT, mirror flipping not needed).
    """

    def __init__(
        self,
        iso_path: str,
        ipc_port: Optional[int] = None,
        dolphin_exe: Optional[str] = None,
        dolphin_ini: Path = DEFAULT_DOLPHIN_INI,
        extra_args: Optional[list[str]] = None,
        connect_timeout_s: float = 30.0,
        log_passthrough: bool = True,
        log_file: Optional[Path] = None,
        keep_alive: bool = False,
        headless: bool = False,
        worker_id: int = 0,
        savestate_path: Optional[str] = None,
        user_dir: Optional[Path] = None,
        ai_controlled_port: int = 0,
        ai_mirror_x: bool = False,
    ):
        self.iso_path = str(iso_path)
        self.headless = headless
        self.worker_id = worker_id
        self.savestate_path = str(savestate_path) if savestate_path else None
        self.ai_controlled_port = ai_controlled_port
        self.ai_mirror_x = ai_mirror_x

        # Port selection.
        if ipc_port is not None:
            self.ipc_port = ipc_port
        elif headless:
            self.ipc_port = HEADLESS_BASE_PORT + worker_id
        else:
            self.ipc_port = portpicker.pick_unused_port()

        # Executable.
        if dolphin_exe is not None:
            self.dolphin_exe = dolphin_exe
        else:
            self.dolphin_exe = DEFAULT_NOGUI_EXE if headless else DEFAULT_DOLPHIN_EXE

        # User directory (headless only).
        if headless:
            if user_dir is not None:
                self.user_dir = Path(user_dir)
            else:
                self.user_dir = Path(f"/tmp/citrus_env_{worker_id}")
        else:
            self.user_dir = None

        self.dolphin_ini = Path(dolphin_ini)
        self.extra_args = list(extra_args) if extra_args else []
        self.connect_timeout_s = connect_timeout_s
        self.log_passthrough = log_passthrough
        self.log_file = Path(log_file) if log_file else None
        self.keep_alive = keep_alive

        self._proc: Optional[subprocess.Popen] = None
        self._sock: Optional[socket.socket] = None
        self._original_ipc_port: Optional[str] = None
        self._log_fh = None

        # Sanity checks.
        if headless and not self.savestate_path:
            raise ValueError(
                "headless=True requires savestate_path (no menu nav possible)"
            )

    # ------------------------------------------------------------------
    # INI handling
    # ------------------------------------------------------------------
    def _patch_windowed_ini(self, port: int) -> None:
        """Windowed mode: patch [Movie] AIIpcPort in the user's shared INI.

        Restored on close so day-to-day BC play isn't disrupted.
        """
        if not self.dolphin_ini.exists():
            raise FileNotFoundError(
                f"Dolphin.ini not found at {self.dolphin_ini}; launch Dolphin "
                f"once to create it, or pass dolphin_ini=..."
            )
        cp = configparser.ConfigParser(strict=False, interpolation=None)
        cp.optionxform = str  # preserve key casing (Dolphin is case-sensitive)
        cp.read(self.dolphin_ini, encoding="utf-8")
        if not cp.has_section("Movie"):
            cp.add_section("Movie")
        self._original_ipc_port = cp["Movie"].get("AIIpcPort", "0")
        cp["Movie"]["AIIpcPort"] = str(port)
        with open(self.dolphin_ini, "w", encoding="utf-8") as f:
            cp.write(f, space_around_delimiters=False)

    def _restore_windowed_ini(self) -> None:
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

    def _write_headless_ini(self, port: int) -> None:
        """Headless mode: write a fresh per-worker INI into user_dir/Config/.

        Each worker's INI is independent (no shared state, no patch race).
        Contents are deterministic per worker — overwriting is the point.

        Config keys set here mirror what BootManager + MovieConfigLoader
        consult on Strikers boot.  Notably ``[Core] GFXBackend=Null`` and
        ``[DSP] Backend=No Audio Output`` cut all rendering / audio work,
        and ``[Core] EmulationSpeed=0`` runs the JIT as fast as the host
        allows.
        """
        assert self.user_dir is not None
        config_dir = self.user_dir / "Config"
        config_dir.mkdir(parents=True, exist_ok=True)
        ini_path = config_dir / "Dolphin.ini"

        cp = configparser.ConfigParser(strict=False, interpolation=None)
        cp.optionxform = str

        cp["Core"] = {
            "GFXBackend": "Null",
            "EmulationSpeed": "0",  # 0 = unlimited (also forced by BootManager
                                    # when AIIpcPort > 0)
            # CPU thread on, JIT64, DSP HLE — defaults that work on Linux x86_64.
            "CPUThread": "True",
            "CPUCore": "1",  # Cached Interpreter on ARM, JIT64 on x86
            "DSPHLE": "True",
            "FastDiscSpeed": "True",
            # Disable Citrus CIT replay auto-recording.  Default is True (the
            # windowed flow captures every match for replay).  In RL we don't
            # want match-end to write a .citframes file (disk balloon) or
            # require base.sav to exist on the worker.  Core.cpp:390 gates
            # auto-recording on this flag.
            "Replays": "False",
        }
        cp["DSP"] = {
            "Backend": "No Audio Output",
        }
        cp["Movie"] = {
            "AIIpcPort": str(port),
            "AIControlledPort": str(self.ai_controlled_port),
            "AIMirrorX": "True" if self.ai_mirror_x else "False",
            "AIPanicAlerts": "False",  # suppress panic dialogs (headless = no GUI to ack)
            # Synchronous pacing: OnFrameEnd blocks until Python returns an
            # action echoing the submitted frame_id (200ms watchdog).  Pins
            # the emulator's frame cadence to the trainer's response rate,
            # which is what makes N parallel headless workers actually scale
            # — without it the emulators free-run, starve the trainer of
            # CPU, and discard ~2/3 of frames (see rl/bench_act.py).  Off
            # by default in C++ to keep windowed BC play unchanged.
            "AISynchronous": "True",
            # UseNullBackend is a Citrus hack that forces Null GFXBackend +
            # unlimited speed during DTM playback (BootManager.cpp:78,
            # MovieConfigLoader.cpp:35).  Our RL flow boots savestates, not
            # DTMs, so this is technically a no-op for us — the [Core]
            # settings above already pin Null/unlimited.  Set anyway as
            # belt-and-suspenders insurance against any runtime path that
            # would otherwise restore a non-Null backend on savestate load.
            "UseNullBackend": "True",
        }
        # Disable analytics prompts and update checks — headless boxes can't
        # answer dialogs.
        cp["Analytics"] = {
            "Enabled": "False",
            "PermissionAsked": "True",
        }
        cp["General"] = {
            "ShowLag": "False",
            "ShowFrameCount": "False",
        }

        with open(ini_path, "w", encoding="utf-8") as f:
            cp.write(f, space_around_delimiters=False)

    # ------------------------------------------------------------------
    # Subprocess + socket
    # ------------------------------------------------------------------
    def _build_cmd(self) -> list[str]:
        if self.headless:
            assert self.user_dir is not None
            assert self.savestate_path is not None
            cmd = [
                self.dolphin_exe,
                "-p", "headless",
                "-u", str(self.user_dir),
                "-e", self.iso_path,
                "-s", self.savestate_path,
                *self.extra_args,
            ]
        else:
            # Windowed: -e auto-boots the ISO.  Video/audio come from the
            # user's INI.  No -b (batch) because that suppresses the
            # launcher/log window we want during interactive Phase A/B work.
            cmd = [self.dolphin_exe, "-e", self.iso_path, *self.extra_args]
        return cmd

    def start(self) -> None:
        if not Path(self.dolphin_exe).exists():
            raise FileNotFoundError(f"Dolphin exe not found at {self.dolphin_exe}")
        if not Path(self.iso_path).exists():
            raise FileNotFoundError(f"ISO not found at {self.iso_path}")
        if self.savestate_path and not Path(self.savestate_path).exists():
            raise FileNotFoundError(f"savestate not found at {self.savestate_path}")

        if self.headless:
            self._write_headless_ini(self.ipc_port)
        else:
            self._patch_windowed_ini(self.ipc_port)

        cmd = self._build_cmd()
        print(
            f"[Dolphin{f'#{self.worker_id}' if self.headless else ''}] "
            f"launching on port {self.ipc_port}: {' '.join(cmd)}",
            flush=True,
        )

        # stdout/stderr routing.  log_file overrides log_passthrough.
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self.log_file, "w", encoding="utf-8", buffering=1)
            stdout = self._log_fh
            stderr = subprocess.STDOUT
        elif self.log_passthrough:
            stdout = None
            stderr = None
        else:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL

        # Detach into its own process group so SIGINT to the trainer
        # doesn't take down a wedged Dolphin we'd otherwise want to clean
        # up explicitly via SHUTDOWN packet + kill.
        popen_kwargs = dict(stdout=stdout, stderr=stderr)
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True

        self._proc = subprocess.Popen(cmd, **popen_kwargs)

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
                    f"check the Dolphin logs"
                    + (f" at {self.log_file}" if self.log_file else " above")
                )
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                # IMPORTANT: blocking connect, no per-attempt timeout — see
                # feedback_ipc_connect_race for the kernel/userspace race
                # that orphans connections under settimeout().
                s.connect(("127.0.0.1", self.ipc_port))
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock = s
                print(
                    f"[Dolphin{f'#{self.worker_id}' if self.headless else ''}] "
                    f"connected on port {self.ipc_port} after {attempt} attempts",
                    flush=True,
                )
                return s
            except (ConnectionRefusedError, OSError) as e:
                last_err = e
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

        # 3. Wait + kill subprocess unless we were asked to leave it running.
        if self._proc is not None:
            if self.keep_alive:
                print(
                    f"[Dolphin{f'#{self.worker_id}' if self.headless else ''}] "
                    f"keep_alive=True; leaving pid={self._proc.pid} running.",
                    flush=True,
                )
                self._proc = None
            else:
                try:
                    self._proc.wait(timeout=wait_s)
                except subprocess.TimeoutExpired:
                    print(
                        f"[Dolphin{f'#{self.worker_id}' if self.headless else ''}] "
                        f"subprocess did not exit within {wait_s}s; killing pid={self._proc.pid}",
                        flush=True,
                    )
                    self._proc.kill()
                    try:
                        self._proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        pass
                self._proc = None

        # 4. Close per-worker log file if we opened one.
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except OSError:
                pass
            self._log_fh = None

        # 5. Windowed: restore shared INI.  Headless: per-worker INI is
        #    disposable, leave it for post-mortem inspection.
        if not self.headless:
            self._restore_windowed_ini()

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
