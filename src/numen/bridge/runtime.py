from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from numen.compiler.flatten import CompiledSpec

_JULIA_PKG_DIR = Path(__file__).parent.parent.parent.parent / "julia"
_RUNNER_JL = _JULIA_PKG_DIR / "src" / "runner.jl"


@dataclass
class SolveResult:
    t: np.ndarray   # shape (n_steps,)
    x: np.ndarray   # shape (state_size, n_steps)


class JuliaBackend:
    """Julia + OrdinaryDiffEq.jl solver backend via subprocess.

    Spawns a fresh Julia process for each solve to avoid juliacall's
    in-process shared-library issues.  Julia startup (~500 ms) dominates
    the first call; subsequent calls pay the same cost (no warm state).

    Args:
        julia_file: Path to the .jl file that defines all ``dynamics_fn`` modules.
        method:     Julia solver name (default ``"Tsit5"``).
        rtol:       Relative tolerance.
        atol:       Absolute tolerance.

    Example::

        backend = JuliaBackend(julia_file="examples/oscillator/dynamics.jl")
        result  = backend.solve(spec, tspan=(0.0, 5.0))
    """

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

    def solve(self, compiled_spec: CompiledSpec, tspan: tuple[float, float]) -> SolveResult:
        missing = [s.dynamics_fn for s in compiled_spec.systems if not s.dynamics_fn]
        if missing:
            raise ValueError(
                f"Systems with empty dynamics_fn (required for JuliaBackend): {missing}\n"
                f"Set 'dynamics_fn: str = \"Module.function_name!\"' on each System class."
            )

        payload = {
            "spec": compiled_spec.to_dict(),
            "tspan": list(tspan),
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

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Julia subprocess failed (exit {proc.returncode}):\n"
                    f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                )

            data = json.loads(result_path.read_text())
        finally:
            payload_path.unlink(missing_ok=True)
            result_path.unlink(missing_ok=True)

        t = np.array(data["t"])
        x = np.array(data["x"])   # shape (state_size, n_steps) — runner.jl serializes row-wise
        return SolveResult(t=t, x=x)
