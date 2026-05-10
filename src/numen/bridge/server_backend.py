from __future__ import annotations

import json
import logging
import queue
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, ClassVar, Sequence

import numpy as np

from numen.compiler.flatten import CompiledSpec
from numen.bridge.runtime import SolveResult
from numen.errors import check_backend_features, check_julia_fns

_log = logging.getLogger("numen.backend.julia_server")

_JULIA_PKG_DIR = Path(__file__).parent.parent.parent.parent / "julia"
_SERVER_JL = _JULIA_PKG_DIR / "src" / "server.jl"

_READY_SIGNAL = "NUMEN_SERVER_READY"
_STARTUP_TIMEOUT = 120.0


def _ensure_julia_env() -> None:
    """Run Pkg.instantiate() the first time, when Manifest.toml is absent."""
    if (_JULIA_PKG_DIR / "Manifest.toml").exists():
        return
    if not (_JULIA_PKG_DIR / "Project.toml").exists():
        return
    print("First-time Julia server setup: installing dependencies (this may take a few minutes)...",
          flush=True)
    subprocess.run(
        ["julia", f"--project={_JULIA_PKG_DIR}", "-e", "using Pkg; Pkg.instantiate()"],
        check=True,
    )


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

    supported_features: ClassVar[frozenset[str]] = frozenset({
        "vector_fields",
        "discrete_fields",
        "continuous_fields",
        "control_callbacks",
        "dae_constraints",
    })

    def __init__(
        self,
        julia_file: str | None = None,
        method: str = "Tsit5",
        rtol: float = 1e-6,
        atol: float = 1e-8,
        eager: bool = False,
        n_save_points: int = 0,
        dtsave: float | None = None,
        dtmax: float | None = None,
    ) -> None:
        if n_save_points > 0 and dtsave is not None:
            raise ValueError("Specify either n_save_points or dtsave, not both.")
        self._julia_file = str(Path(julia_file).resolve()) if julia_file else None
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.n_save_points = n_save_points
        self.dtsave = dtsave
        self.dtmax = dtmax
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.startup_ms: float | None = None

        if eager:
            self._ensure_started()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        compiled_spec: CompiledSpec,
        tspan: tuple[float, float],
        progress: bool = False,
    ) -> SolveResult:
        """Send one solve request to the running Julia server.

        Args:
            compiled_spec: Compiled simulation spec.
            tspan:         (t0, tf) integration interval.
            progress:      If True, show an elapsed-time spinner while solving.
                           Requires tqdm (``pip install tqdm``); silently skipped
                           if not installed.
        """
        check_backend_features(compiled_spec, "JuliaServerBackend", self.supported_features)

        _log.debug(
            "solve: state_size=%d param_size=%d tspan=%s method=%s rtol=%g atol=%g n_save=%d",
            compiled_spec.state_size, compiled_spec.param_size,
            tspan, self.method, self.rtol, self.atol, self.n_save_points,
        )

        t0_wall = time.perf_counter()
        with self._lock:
            self._ensure_started()
            payload = self._build_payload(compiled_spec, tspan)
            line = json.dumps(payload)
            try:
                self._proc.stdin.write(line.encode() + b"\n")
                self._proc.stdin.flush()
                if progress:
                    header_line = _read_with_spinner(self._proc.stdout, label="julia")
                else:
                    header_line = self._proc.stdout.readline()
                header = json.loads(header_line)
                if "error" not in header:
                    # Chunked protocol: header gives counts, then t line, then x rows
                    t_line = self._proc.stdout.readline()
                    x_lines = [self._proc.stdout.readline()
                                for _ in range(header["n_states"])]
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError("Julia server process died unexpectedly.") from exc

        if not header_line:
            raise RuntimeError("Julia server closed stdout without responding.")

        elapsed_ms = (time.perf_counter() - t0_wall) * 1000
        if "error" in header:
            _log.error("Julia server error after %.0f ms: %s", elapsed_ms, header["error"][:200])
            raise RuntimeError(f"Julia server error:\n{header['error']}")

        _log.debug("solve done in %.1f ms  n_t=%d  n_states=%d",
                   elapsed_ms, header["n_t"], header["n_states"])
        t = np.array(json.loads(t_line))
        x = np.array([json.loads(row) for row in x_lines])
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
        if self.is_running:
            return

        _ensure_julia_env()

        cmd = ["julia", f"--project={_JULIA_PKG_DIR}", str(_SERVER_JL)]
        if self._julia_file:
            cmd.append(self._julia_file)

        t0 = time.perf_counter()
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        ready_event = threading.Event()
        stderr_lines: list[str] = []

        def _read_stderr() -> None:
            for raw in proc.stderr:
                decoded = raw.decode(errors="replace").rstrip()
                stderr_lines.append(decoded)
                if decoded == _READY_SIGNAL:
                    ready_event.set()
                elif decoded.startswith("NUMEN_PRECOMPILE_DONE"):
                    _log.info("[julia] %s", decoded)
                elif decoded.strip():
                    _log.debug("[julia] %s", decoded)

        threading.Thread(target=_read_stderr, daemon=True, name="julia-server-stderr").start()

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

    def _build_payload(self, compiled_spec: CompiledSpec, tspan: tuple[float, float]) -> dict:
        check_julia_fns(compiled_spec, "JuliaServerBackend")
        return {
            "spec":           compiled_spec.to_dict(),
            "tspan":          list(tspan),
            "method":         self.method,
            "rtol":           self.rtol,
            "atol":           self.atol,
            "n_save_points":  self.n_save_points,
            "dtsave":         self.dtsave,
            "dtmax":          self.dtmax,
        }


# ---------------------------------------------------------------------------
# JuliaServerPool — N parallel servers for parameter sweeps
# ---------------------------------------------------------------------------

class JuliaServerPool:
    """Pool of N persistent Julia servers for parallel parameter sweeps.

    Each server runs in its own Julia process.  ``solve`` acquires an idle
    server from an internal queue (blocking if all are busy).  ``map``
    distributes an iterable of tasks across the pool using threads.

    Args:
        n_workers:  Number of parallel Julia server processes.
        julia_file: Path to the .jl dynamics file (same for all workers).
        method:     Julia solver name (e.g. ``"Rodas5P"``).
        rtol:       Relative tolerance.
        atol:       Absolute tolerance.

    Example — parallel parameter sweep::

        params = [{"spring_k": k} for k in np.linspace(100, 1000, 50)]

        with JuliaServerPool(n_workers=4, julia_file="dynamics.jl", method="Rodas5P") as pool:
            results = pool.map(
                lambda server, p: server.solve(compile_spec(make_world(p)), tspan),
                params,
                progress=True,
            )
    """

    def __init__(
        self,
        n_workers: int,
        julia_file: str | None = None,
        method: str = "Tsit5",
        rtol: float = 1e-6,
        atol: float = 1e-8,
        n_save_points: int = 0,
        dtsave: float | None = None,
        dtmax: float | None = None,
    ) -> None:
        self._n = n_workers
        self._servers = [
            JuliaServerBackend(
                julia_file=julia_file, method=method, rtol=rtol, atol=atol,
                eager=True, n_save_points=n_save_points, dtsave=dtsave, dtmax=dtmax,
            )
            for _ in range(n_workers)
        ]
        # Queue acts as a semaphore — each server token is available when idle
        self._q: queue.Queue[JuliaServerBackend] = queue.Queue()
        for s in self._servers:
            self._q.put(s)

    def solve(
        self,
        compiled_spec: CompiledSpec,
        tspan: tuple[float, float],
    ) -> SolveResult:
        """Acquire an idle server, solve, release.  Blocks if all servers busy."""
        server = self._q.get()
        try:
            return server.solve(compiled_spec, tspan)
        finally:
            self._q.put(server)

    def map(
        self,
        fn: Callable[["JuliaServerBackend", object], object],
        items: Sequence,
        *,
        progress: bool = False,
    ) -> list:
        """Apply ``fn(server, item)`` for every item, in parallel across workers.

        Results are returned in the same order as ``items``.

        Args:
            fn:       Callable taking ``(JuliaServerBackend, item)`` and returning
                      a result.  Called on a pooled server; do not call
                      ``server.close()`` inside ``fn``.
            items:    Iterable of inputs.
            progress: Show a tqdm progress bar over completed tasks.

        Example::

            results = pool.map(
                lambda server, p: server.solve(compile_spec(make_world(p)), (0, T)),
                param_list,
                progress=True,
            )
        """
        items = list(items)
        n = len(items)
        results: list = [None] * n

        def _worker(idx: int, item) -> None:
            server = self._q.get()
            try:
                results[idx] = fn(server, item)
            finally:
                self._q.put(server)

        with ThreadPoolExecutor(max_workers=self._n) as executor:
            futures = [executor.submit(_worker, i, item) for i, item in enumerate(items)]
            if progress:
                try:
                    from tqdm.auto import tqdm
                    for _ in tqdm(as_completed(futures), total=n, desc="julia pool"):
                        pass
                except ImportError:
                    for f in futures:
                        f.result()
            else:
                for f in futures:
                    f.result()

        return results

    def borrow(self):
        """Context manager that acquires one idle server and releases it on exit.

        Use this to route a single (non-parallelisable) task through the pool
        without bypassing the queue::

            with pool.borrow() as server:
                result = server.solve(spec, tspan)
        """
        from contextlib import contextmanager

        @contextmanager
        def _borrow():
            server = self._q.get()
            try:
                yield server
            finally:
                self._q.put(server)

        return _borrow()

    def close(self) -> None:
        """Shut down all server processes."""
        for s in self._servers:
            s.close()

    def __enter__(self) -> "JuliaServerPool":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def n_workers(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _read_with_spinner(stdout, label: str = "") -> bytes:
    """Read one line from stdout while showing an elapsed-time spinner."""
    try:
        from tqdm.auto import tqdm
        pbar = tqdm(bar_format=f"{label} {{elapsed}}", desc="", total=0, dynamic_ncols=True)
    except ImportError:
        pbar = None

    result: list[bytes] = []
    done = threading.Event()

    def _read():
        result.append(stdout.readline())
        done.set()

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()

    while not done.wait(timeout=0.1):
        if pbar is not None:
            pbar.update(0)   # triggers elapsed refresh

    if pbar is not None:
        pbar.close()

    reader.join()
    return result[0] if result else b""
