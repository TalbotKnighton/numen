from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

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


class ScipyBackend:
    """Pure-Python solver backend using scipy solve_ivp. Good for development and testing."""

    def __init__(self, method: str = "RK45", rtol: float = 1e-6, atol: float = 1e-8):
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def solve(
        self,
        compiled_spec: CompiledSpec,
        tspan: tuple[float, float],
        t_eval: "np.ndarray | None" = None,
        progress: bool = False,
    ) -> SolveResult:
        missing = [s.dynamics_fn for s in compiled_spec.systems if s.python_fn is None]
        if missing:
            raise ValueError(
                f"Systems missing python_fn (required for ScipyBackend): {missing}\n"
                f"Declare 'python_fn: ClassVar[DynamicsFn] = staticmethod(your_fn)' on the System class."
            )

        p = np.array(compiled_spec.p)
        tstops = _build_tstops(compiled_spec.discrete_dts, tspan)
        t0_sim, tf_sim = tspan

        def rhs(t: float, x: np.ndarray) -> np.ndarray:
            dx = DxBuffer(np.zeros_like(x))
            for sys in compiled_spec.systems:
                sys.python_fn(dx, x, p, t, compiled_spec, sys)
            return dx.array

        if progress:
            rhs = _wrap_rhs_progress(rhs, t0_sim, tf_sim)

        if t_eval is not None:
            dense_eval = np.sort(np.unique(np.concatenate([t_eval, tstops]))) if tstops else t_eval
        else:
            dense_eval = tstops if tstops else None

        try:
            sol = solve_ivp(
                rhs,
                tspan,
                compiled_spec.x0,
                method=self.method,
                t_eval=dense_eval,
                rtol=self.rtol,
                atol=self.atol,
            )
        finally:
            if progress:
                _close_rhs_progress(rhs)

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        return SolveResult(t=sol.t, x=sol.y)


def _wrap_rhs_progress(rhs, t0: float, tf: float):
    """Wrap an RHS function so it updates a tqdm bar as integration advances."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return rhs   # tqdm not installed — skip silently

    duration = tf - t0
    pbar = tqdm(
        total=duration,
        unit="s",
        desc="scipy",
        bar_format="{desc}: {percentage:.0f}%|{bar}| {n:.3g}/{total:.3g} s  [{elapsed}<{remaining}]",
        dynamic_ncols=True,
    )
    last_t = [t0]

    def rhs_tracked(t: float, x):
        if t > last_t[0]:
            pbar.update(t - last_t[0])
            last_t[0] = t
        return rhs(t, x)

    rhs_tracked._pbar = pbar   # stashed for cleanup
    return rhs_tracked


def _close_rhs_progress(rhs_tracked) -> None:
    pbar = getattr(rhs_tracked, "_pbar", None)
    if pbar is not None:
        pbar.close()
