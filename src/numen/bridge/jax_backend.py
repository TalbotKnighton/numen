from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from numen.compiler.flatten import CompiledSpec, DxBuffer
from numen.bridge.runtime import SolveResult

if TYPE_CHECKING:
    pass


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

    JIT-compiles the full ODE right-hand side on the first call; subsequent calls
    run pure XLA with no Python overhead.  The user's dynamics functions require
    **no changes** — the ``DxBuffer`` proxy routes JAX functional updates
    (``arr.at[s].set(value)``) transparently while keeping the same ``+=`` API.

    During JIT tracing, all dict lookups in ``spec.state_index_map`` and the Python
    ``for`` loops over ``entity_groups`` run once at trace time.  The compiled XLA
    kernel contains only integer-indexed array operations — matching the
    "baked literal" optimization described in DESIGN.md, without any codegen step.

    Args:
        rtol:    Relative tolerance (passed to diffrax ``PIDController``).
        atol:    Absolute tolerance.
        n_saves: Number of evenly-spaced save points when no discrete fields are
                 present.  Ignored if the spec has discrete update rates.

    Example::

        result = JAXBackend(rtol=1e-9, atol=1e-9).solve(spec, tspan=(0.0, 5.0))
    """

    def __init__(self, rtol: float = 1e-6, atol: float = 1e-8, n_saves: int = 500) -> None:
        self.rtol = rtol
        self.atol = atol
        self.n_saves = n_saves

    def solve(self, compiled_spec: CompiledSpec, tspan: tuple[float, float]) -> SolveResult:
        import jax
        import jax.numpy as jnp
        import diffrax

        missing = [s.dynamics_fn for s in compiled_spec.systems if s.python_fn is None]
        if missing:
            raise ValueError(
                f"Systems missing python_fn (required for JAXBackend): {missing}\n"
                f"Declare 'python_fn: ClassVar[DynamicsFn] = staticmethod(your_fn)' on the System class."
            )

        p     = jnp.array(compiled_spec.p)
        x0    = jnp.array(compiled_spec.x0)
        t0, tf = tspan

        tstops = _build_tstops(compiled_spec.discrete_dts, tspan)
        if tstops:
            save_ts = jnp.array(tstops)
        else:
            save_ts = jnp.linspace(t0, tf, self.n_saves)

        systems = compiled_spec.systems

        def rhs(t: float, y: jnp.ndarray, _args: None) -> jnp.ndarray:
            dx = DxBuffer(jnp.zeros_like(y))
            for sys in systems:
                sys.python_fn(dx, y, p, t, compiled_spec, sys)
            return dx.array

        term   = diffrax.ODETerm(rhs)
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts=save_ts)
        ctrl   = diffrax.PIDController(rtol=self.rtol, atol=self.atol)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=tf,
            dt0=None,
            y0=x0,
            args=None,
            saveat=saveat,
            stepsize_controller=ctrl,
        )

        t_out = np.array(sol.ts)
        x_out = np.array(sol.ys).T   # diffrax: (n_saves, state_size) → (state_size, n_saves)
        return SolveResult(t=t_out, x=x_out)
