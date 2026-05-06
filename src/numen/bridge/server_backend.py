from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from numen.compiler.flatten import CompiledSpec
from numen.bridge.runtime import SolveResult

_JULIA_PKG_DIR = Path(__file__).parent.parent.parent.parent / "julia"
_SERVER_JL = _JULIA_PKG_DIR / "src" / "server.jl"

_READY_SIGNAL = "NUMEN_SERVER_READY"
_STARTUP_TIMEOUT = 120.0   # seconds to wait for Julia to boot + load packages


class JuliaServerBackend:
    """Persistent Julia server — pay JIT cost once, then solve repeatedly.

    Starts a single Julia subprocess on first use (or at construction if
    ``eager=True``).  All subsequent ``solve`` calls send a JSON request over
    stdin and read the result from stdout — no subprocess-per-call overhead.

    The server stays alive until ``close()`` is called or the context manager
    exits.  A new server is started automatically if the previous one died.

    Args:
        julia_file: Path to the .jl file that defines dynamics modules.
        method:     Julia solver name (e.g. ``"Tsit5"``, ``"Rodas5P"``).
        rtol:       Relative tolerance.
        atol:       Absolute tolerance.
        eager:      If True, start the Julia process immediately in ``__init__``
                    rather than on the first ``solve`` call.

    Example::

        with JuliaServerBackend(julia_file="dynamics.jl", method="Rodas5P") as server:
            for params in sweep:
                spec   = compile_spec(make_world(params))
                result = server.solve(spec, tspan=(0.0, 3600.0))
    """

    def __init__(
        self,
        julia_file: str | None = None,
        method: str = "Tsit5",
        rtol: float = 1e-6,
        atol: float = 1e-8,
        eager: bool = False,
    ) -> None:
        self._julia_file = str(Path(julia_file).resolve()) if julia_file else None
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.startup_ms: float | None = None   # set after first successful start

        if eager:
            self._ensure_started()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        compiled_spec: CompiledSpec,
        tspan: tuple[float, float],
    ) -> SolveResult:
        """Send one solve request to the running Julia server.

        Starts the server automatically on the first call.  If the server
        process has died unexpectedly it is restarted (paying JIT cost again).
        """
        with self._lock:
            self._ensure_started()
            payload = self._build_payload(compiled_spec, tspan)
            line = json.dumps(payload)
            try:
                self._proc.stdin.write(line.encode() + b"\n")
                self._proc.stdin.flush()
                response_line = self._proc.stdout.readline()
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError("Julia server process died unexpectedly.") from exc

        if not response_line:
            raise RuntimeError("Julia server closed stdout without responding.")

        data = json.loads(response_line)
        if "error" in data:
            raise RuntimeError(f"Julia server error:\n{data['error']}")

        t = np.array(data["t"])
        x = np.array(data["x"])
        return SolveResult(t=t, x=x)

    def close(self) -> None:
        """Shut down the Julia server process."""
        with self._lock:
            if self._proc is not None:
                try:
                    self._proc.stdin.close()
                    self._proc.wait(timeout=10)
                except Exception:
                    self._proc.kill()
                finally:
                    self._proc = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "JuliaServerBackend":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        """Start the server if not running.  Must be called under self._lock."""
        if self.is_running:
            return

        cmd = [
            "julia",
            f"--project={_JULIA_PKG_DIR}",
            str(_SERVER_JL),
        ]
        if self._julia_file:
            cmd.append(self._julia_file)

        t0 = time.perf_counter()
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Read stderr in a background thread so readline() never blocks the
        # main thread.  Set ready_event when NUMEN_SERVER_READY is seen.
        ready_event = threading.Event()
        stderr_lines: list[str] = []

        def _read_stderr() -> None:
            for raw in proc.stderr:
                decoded = raw.decode(errors="replace").rstrip()
                stderr_lines.append(decoded)
                if decoded == _READY_SIGNAL:
                    ready_event.set()

        threading.Thread(
            target=_read_stderr, daemon=True, name="julia-server-stderr"
        ).start()

        if not ready_event.wait(timeout=_STARTUP_TIMEOUT):
            proc.kill()
            proc.wait()
            stderr_dump = "\n".join(stderr_lines[-40:])
            raise RuntimeError(
                f"Julia server did not signal ready within {_STARTUP_TIMEOUT}s.\n"
                f"Last stderr:\n{stderr_dump}"
            )

        self._proc = proc
        self.startup_ms = (time.perf_counter() - t0) * 1000

    def _build_payload(
        self,
        compiled_spec: CompiledSpec,
        tspan: tuple[float, float],
    ) -> dict:
        missing = [s.dynamics_fn for s in compiled_spec.systems if not s.dynamics_fn]
        if missing:
            raise ValueError(
                f"Systems with empty dynamics_fn (required for JuliaServerBackend): {missing}"
            )
        return {
            "spec":   compiled_spec.to_dict(),
            "tspan":  list(tspan),
            "method": self.method,
            "rtol":   self.rtol,
            "atol":   self.atol,
        }
