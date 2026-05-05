from __future__ import annotations

import numpy as np
from typing import Any, Callable

from numen.compiler.flatten import CompiledSpec, DxBuffer
from numen.bridge.runtime import SolveResult


def _build_tstops(discrete_dts: list[float], tspan: tuple[float, float]) -> list[float]:
    t0, tf = tspan
    times: set[float] = set()
    for dt in discrete_dts:
        t = t0 + dt
        while t <= tf + 1e-12:
            times.add(round(t, 12))
            t += dt
    return sorted(times)


class JAXBackend:
    """JAX + diffrax solver backend.

    The *entire* ``diffeqsolve`` call — ODE steps, RHS evaluations, and
    adaptive step-size control — is wrapped in ``jax.jit``.  diffrax uses
    ``lax.while_loop`` internally, which only becomes efficient inside JIT.

    The compiled XLA program is cached per ``(compiled_spec id, tspan)`` so
    repeated solves with the same problem (e.g. Monte Carlo over initial
    conditions) reuse the compiled kernel.  Only ``x0`` is a dynamic input:
    parameters ``p``, save times, and tolerances are static constants baked
    into the compiled program.

    During JIT tracing, Python ``for`` loops over ``entity_groups`` and dict
    lookups in ``state_index_map`` run once; the XLA kernel contains only
    integer-indexed array operations with no dict overhead at execution time.

    Args:
        rtol:    Relative tolerance (diffrax PIDController).
        atol:    Absolute tolerance.
        n_saves: Number of evenly-spaced save points when no discrete fields
                 are present.
    """

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        n_saves: int = 500,
        max_steps: int = 100_000,
        solver: str = "Dopri5",
    ) -> None:
        self.rtol      = rtol
        self.atol      = atol
        self.n_saves   = n_saves
        self.max_steps = max_steps
        self.solver    = solver
        self._cache: dict[tuple, Callable] = {}

    def _build_run_fn(self, compiled_spec: CompiledSpec, tspan: tuple[float, float]) -> Callable:
        """Build and JIT-compile the full ODE solve for this spec + tspan."""
        import jax
        import jax.numpy as jnp
        import diffrax

        t0, tf   = tspan
        p        = jnp.array(compiled_spec.p)
        systems  = compiled_spec.systems

        tstops   = _build_tstops(compiled_spec.discrete_dts, tspan)
        save_ts  = jnp.array(tstops) if tstops else jnp.linspace(t0, tf, self.n_saves)

        max_steps = self.max_steps
        ctrl      = diffrax.PIDController(rtol=self.rtol, atol=self.atol)
        saveat    = diffrax.SaveAt(ts=save_ts)
        _solver   = getattr(diffrax, self.solver)()

        @jax.jit
        def run(x0: jnp.ndarray) -> Any:
            def rhs(t: float, y: jnp.ndarray, _args: None) -> jnp.ndarray:
                dx = DxBuffer(jnp.zeros_like(y))
                for sys in systems:
                    sys.python_fn(dx, y, p, t, compiled_spec, sys)
                return dx.array

            return diffrax.diffeqsolve(
                diffrax.ODETerm(rhs),
                _solver,
                t0=t0,
                t1=tf,
                dt0=None,
                y0=x0,
                args=None,
                saveat=saveat,
                stepsize_controller=ctrl,
                max_steps=max_steps,
            )

        return run

    def solve(self, compiled_spec: CompiledSpec, tspan: tuple[float, float]) -> SolveResult:
        import jax.numpy as jnp

        missing = [s.dynamics_fn for s in compiled_spec.systems if s.python_fn is None]
        if missing:
            raise ValueError(
                f"Systems missing python_fn (required for JAXBackend): {missing}\n"
                f"Declare 'python_fn: ClassVar[DynamicsFn] = staticmethod(your_fn)' on the System class."
            )

        key = (id(compiled_spec), tspan)
        if key not in self._cache:
            self._cache[key] = self._build_run_fn(compiled_spec, tspan)

        run = self._cache[key]
        x0  = jnp.array(compiled_spec.x0)
        sol = run(x0)

        t_out = np.array(sol.ts)
        x_out = np.array(sol.ys).T   # (n_saves, state_size) → (state_size, n_saves)
        return SolveResult(t=t_out, x=x_out)
