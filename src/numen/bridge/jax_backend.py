"""JAX + diffrax solver backend for Numen.

``JAXBackend`` JIT-compiles the full ODE solve (including RHS evaluations and
adaptive step-size control) into an XLA kernel via ``jax.jit`` or
``equinox.filter_jit``.  The compiled kernel is cached per
``(compiled_spec id, tspan)`` so repeated solves with the same problem reuse
the compiled code — only ``x0`` is a dynamic input.

Typical warm-solve speedup: **~1500×** over ``ScipyBackend`` for non-stiff
problems.  Cold (JIT compile) time is ~550 ms on first call.

Dynamics functions must be JAX-traceable: use ``jnp.*`` instead of ``np.*``,
and ``jnp.where`` instead of Python ``if``/``else`` on state values.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, ClassVar

import numpy as np

from numen.compiler.flatten import CompiledSpec, DxBuffer
from numen.bridge.runtime import SolveResult
from numen.errors import check_backend_features, check_python_fns

_log = logging.getLogger("numen.backend.jax")

# Stiff solvers available in diffrax (in addition to explicit Dopri5 / Tsit5):
#   Kvaerno3, Kvaerno4, Kvaerno5  — SDIRK methods, good for mildly-to-moderately stiff
#   ImplicitEuler                 — robust but first-order only
# For highly stiff problems prefer JuliaServerBackend with method="Rodas5P".


def _build_tstops(discrete_dts: list[float], tspan: tuple[float, float]) -> list[float]:
    t0, tf = tspan
    times: set[float] = set()
    for dt in discrete_dts:
        t = t0 + dt
        while t <= tf + 1e-12:
            times.add(round(t, 12))
            t += dt
    return sorted(times)


def _get_jit():
    """Return equinox.filter_jit if available, else jax.jit.

    equinox.filter_jit provides much clearer runtime error messages —
    especially for 'max_steps exceeded' — compared to jax.jit.
    Install with: pip install equinox
    """
    try:
        import equinox as eqx
        return eqx.filter_jit
    except ImportError:
        import jax
        return jax.jit


class JAXBackend:
    """JAX + diffrax solver backend.

    The *entire* ``diffeqsolve`` call — ODE steps, RHS evaluations, and
    adaptive step-size control — is wrapped in ``jax.jit`` (or
    ``equinox.filter_jit`` if equinox is installed, which gives clearer
    runtime error messages).

    The compiled XLA program is cached per ``(compiled_spec id, tspan)`` so
    repeated solves with the same problem (e.g. Monte Carlo over initial
    conditions) reuse the compiled kernel.  Only ``x0`` is a dynamic input:
    parameters ``p``, save times, and tolerances are static constants baked
    into the compiled program.

    During JIT tracing, Python ``for`` loops over ``entity_groups`` and dict
    lookups in ``state_index_map`` run once; the XLA kernel contains only
    integer-indexed array operations with no dict overhead at execution time.

    Args:
        rtol:      Relative tolerance (diffrax PIDController).
        atol:      Absolute tolerance.
        n_saves:   Number of evenly-spaced save points when no discrete fields
                   are present.
        max_steps: Maximum ODE solver steps.  Increase if you get
                   "maximum number of solver steps was reached".  For stiff
                   problems consider switching to an implicit solver
                   (``solver="Kvaerno5"``) or using JuliaServerBackend with
                   ``method="Rodas5P"``.
        solver:    diffrax solver class name.
                   Explicit (non-stiff): ``"Dopri5"`` (default), ``"Tsit5"``
                   Implicit (stiff):     ``"Kvaerno3"``, ``"Kvaerno4"``,
                                         ``"Kvaerno5"``, ``"ImplicitEuler"``
    """

    supported_features: ClassVar[frozenset[str]] = frozenset({
        "vector_fields",
        "discrete_fields",
        "continuous_fields",
    })

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
        _jit      = _get_jit()

        @_jit
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

    def solve(
        self,
        compiled_spec: CompiledSpec,
        tspan: tuple[float, float],
        progress: bool = False,
    ) -> SolveResult:
        import jax.numpy as jnp

        check_backend_features(compiled_spec, "JAXBackend", self.supported_features)
        check_python_fns(compiled_spec, "JAXBackend")

        _log.debug(
            "solve: state_size=%d param_size=%d tspan=%s solver=%s rtol=%g atol=%g max_steps=%d",
            compiled_spec.state_size, compiled_spec.param_size,
            tspan, self.solver, self.rtol, self.atol, self.max_steps,
        )

        key = (id(compiled_spec), tspan)
        is_first_call = key not in self._cache
        if is_first_call:
            _log.debug("JIT compiling for spec id=%d tspan=%s", id(compiled_spec), tspan)
            self._cache[key] = self._build_run_fn(compiled_spec, tspan)

        run = self._cache[key]
        x0  = jnp.array(compiled_spec.x0)

        label = "JAX (JIT compiling...)" if is_first_call else "JAX"
        t0_wall = time.perf_counter()

        if progress:
            from numen.bridge.server_backend import _read_with_spinner as _spinner
            import threading
            _result = [None]
            _exc    = [None]
            def _run():
                try:
                    _result[0] = run(x0)
                except Exception as e:
                    _exc[0] = e
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            try:
                from tqdm.auto import tqdm
                pbar = tqdm(bar_format=f"{label} {{elapsed}}", total=0, dynamic_ncols=True)
            except ImportError:
                pbar = None
            while t.is_alive():
                if pbar is not None:
                    pbar.update(0)
                t.join(timeout=0.1)
            if pbar is not None:
                pbar.close()
            if _exc[0] is not None:
                raise _exc[0]
            sol = _result[0]
        else:
            try:
                sol = run(x0)
            except Exception as e:
                _reraise_jax_error(e, self.max_steps, self.solver)

        elapsed_ms = (time.perf_counter() - t0_wall) * 1000
        _log.debug("solve done in %.1f ms%s", elapsed_ms, " (includes JIT)" if is_first_call else "")

        t_out = np.array(sol.ts)
        x_out = np.array(sol.ys).T   # (n_saves, state_size) → (state_size, n_saves)
        return SolveResult(t=t_out, x=x_out)


def _reraise_jax_error(exc: Exception, max_steps: int, solver: str) -> None:
    """Catch common diffrax failures and re-raise with actionable guidance."""
    msg = str(exc)
    if "maximum number of solver steps" in msg or "max_steps" in msg.lower():
        raise RuntimeError(
            f"JAX solver hit max_steps={max_steps} before reaching tf.\n\n"
            f"Options:\n"
            f"  1. Increase max_steps:  JAXBackend(max_steps={max_steps * 10}, solver='{solver}')\n"
            f"  2. Switch to an implicit solver for stiff problems:\n"
            f"         JAXBackend(solver='Kvaerno5', max_steps={max_steps})\n"
            f"  3. For highly stiff problems (multiple timescales), use Julia:\n"
            f"         JuliaServerBackend(method='Rodas5P', rtol=1e-6, atol=1e-8)\n"
            f"\nOriginal error: {msg}"
        ) from exc
    raise exc
