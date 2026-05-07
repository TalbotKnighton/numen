from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np

from numen.compiler.flatten import CompiledSpec
from numen.errors import check_backend_features, check_julia_fns

_log = logging.getLogger("numen.backend.julia")

_JULIA_PKG_DIR = Path(__file__).parent.parent.parent.parent / "julia"
_RUNNER_JL = _JULIA_PKG_DIR / "src" / "runner.jl"


@dataclass
class SolveResult:
    t: np.ndarray           # shape (n_steps,)
    x: np.ndarray           # shape (state_size, n_steps)
    timings_ms: list[float] = field(default_factory=list)
    # timings_ms[0]  = first solve  (JIT + dynamics), if reps >= 1
    # timings_ms[1:] = warm solves  (dynamics only),  if reps >  1
    # startup_ms     = wall time − sum(timings_ms)    (subprocess + package load)

    @property
    def startup_ms(self) -> float:
        return self._startup_ms

    def _set_startup(self, wall_ms: float) -> None:
        self._startup_ms = wall_ms - sum(self.timings_ms)

    @property
    def jit_ms(self) -> float | None:
        return self.timings_ms[0] if self.timings_ms else None

    @property
    def warm_ms(self) -> float | None:
        warm = self.timings_ms[1:]
        return min(warm) if warm else None


class JuliaBackend:
    """Julia + OrdinaryDiffEq.jl solver backend via subprocess.

    Spawns a fresh Julia process for each ``solve`` call. Startup (~2–3 s)
    includes Julia boot, package loading, and user dynamics file ``include``.
    Within that process, ``reps`` solves are timed individually: the first
    includes JIT compilation of user dynamics; subsequent ones are warm.

    Args:
        julia_file: Path to the .jl file that defines all ``dynamics_fn`` modules.
        method:     Julia solver name (default ``"Tsit5"``).
        rtol:       Relative tolerance.
        atol:       Absolute tolerance.

    Example::

        backend = JuliaBackend(julia_file="examples/oscillator/dynamics.jl")
        result  = backend.solve(spec, tspan=(0.0, 5.0), reps=6)
        print(f"startup {result.startup_ms:.0f} ms  "
              f"JIT {result.jit_ms:.0f} ms  "
              f"warm {result.warm_ms:.0f} ms")
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
    ) -> None:
        self._julia_file = str(Path(julia_file).resolve()) if julia_file else None
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def solve(
        self,
        compiled_spec: CompiledSpec,
        tspan: tuple[float, float],
        reps: int = 1,
    ) -> SolveResult:
        """Run the ODE solver via Julia subprocess.

        Args:
            compiled_spec: Compiled simulation spec.
            tspan:         (t0, tf) integration interval.
            reps:          Number of solves to run inside the subprocess.
                           reps=1 → single solve (no warm timing).
                           reps>1 → first solve is JIT, rest are warm.
        """
        check_backend_features(compiled_spec, "JuliaBackend", self.supported_features)
        check_julia_fns(compiled_spec, "JuliaBackend")

        _log.debug(
            "solve: state_size=%d param_size=%d tspan=%s method=%s rtol=%g atol=%g reps=%d",
            compiled_spec.state_size, compiled_spec.param_size,
            tspan, self.method, self.rtol, self.atol, reps,
        )

        payload = {
            "spec":   compiled_spec.to_dict(),
            "tspan":  list(tspan),
            "reps":   reps,
            "method": self.method,
            "rtol":   self.rtol,
            "atol":   self.atol,
        }

        payload_path = Path(tempfile.mktemp(suffix=".json"))
        result_path  = Path(tempfile.mktemp(suffix=".json"))
        try:
            payload_path.write_text(json.dumps(payload))
            result_path.touch()

            cmd = [
                "julia",
                f"--project={_JULIA_PKG_DIR}",
                str(_RUNNER_JL),
                str(payload_path),
                str(result_path),
            ]
            if self._julia_file:
                cmd.append(self._julia_file)

            t0 = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            wall_ms = (time.perf_counter() - t0) * 1000

            # Always route Julia output to logger, not just on failure
            for line in proc.stderr.splitlines():
                if line.strip():
                    _log.debug("[julia stderr] %s", line)
            for line in proc.stdout.splitlines():
                if line.strip():
                    _log.debug("[julia stdout] %s", line)

            if proc.returncode != 0:
                _log.error(
                    "Julia subprocess failed (exit %d) after %.0f ms",
                    proc.returncode, wall_ms,
                )
                raise RuntimeError(
                    f"Julia subprocess failed (exit {proc.returncode}):\n"
                    f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                )

            data = json.loads(result_path.read_text())
        finally:
            payload_path.unlink(missing_ok=True)
            result_path.unlink(missing_ok=True)

        _log.debug("solve done in %.0f ms wall time", wall_ms)
        t = np.array(data["t"])
        x = np.array(data["x"])   # shape (state_size, n_steps) — runner.jl serializes row-wise
        result = SolveResult(t=t, x=x, timings_ms=data.get("timings_ms", []))
        result._set_startup(wall_ms)
        return result
