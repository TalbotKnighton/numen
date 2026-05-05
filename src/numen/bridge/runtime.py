from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from numen.compiler.flatten import CompiledSpec

_JULIA_PKG_DIR = Path(__file__).parent.parent.parent.parent / "julia"


@dataclass
class SolveResult:
    t: np.ndarray   # shape (n_steps,)
    x: np.ndarray   # shape (state_size, n_steps)


class NumenRuntime:
    """Low-level juliacall wrapper. Loads Numen.jl and runs solves.

    Prefer ``JuliaBackend`` for the high-level interface.
    """

    def __init__(self, user_julia_file: str | None = None) -> None:
        """
        Args:
            user_julia_file: Path to a .jl file to ``include()`` before solving.
                             Must define the modules referenced by ``dynamics_fn`` on each System.
        """
        from juliacall import Main as jl

        jl.seval(f'import Pkg; Pkg.activate("{_JULIA_PKG_DIR}"); Pkg.instantiate()')
        jl.seval("using Numen")

        if user_julia_file:
            jl.seval(f'include("{user_julia_file}")')

        self._jl = jl

    def solve(self, compiled_spec: CompiledSpec, tspan: tuple[float, float]) -> SolveResult:
        payload = {
            "spec": compiled_spec.to_dict(),
            "tspan": list(tspan),
        }
        result = self._jl.Numen.solve(json.dumps(payload))
        return SolveResult(
            t=np.array(result.t),
            x=np.array(result.x),
        )


class JuliaBackend:
    """Julia + OrdinaryDiffEq.jl solver backend.

    Mirrors the ``ScipyBackend`` interface. Requires Julia and the user's
    dynamics module to be available.

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
        self._runtime = NumenRuntime(julia_file)
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
        return self._runtime.solve(compiled_spec, tspan)
