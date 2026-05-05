"""
Backend performance comparison: ScipyBackend vs JAXBackend vs JuliaBackend

Creates N independent harmonic oscillators (no coupling) and times each backend.
All backends solve the same compiled spec with identical tolerances.

JAX reports two times:
  cold — first call, includes XLA compilation
  warm — best of REPS subsequent calls, pure XLA execution

Julia reports one time (subprocess launch dominates; no warm-up state).

Usage:
    python examples/benchmark.py                 # default N list
    python examples/benchmark.py 1 10 100 500    # custom N values
    python examples/benchmark.py --no-julia      # skip Julia (slow startup)
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

# Resolve the oscillator example directory for local imports
_OSC_DIR = os.path.join(os.path.dirname(__file__), "oscillator")
_DYNAMICS_JL = os.path.join(_OSC_DIR, "dynamics.jl")
sys.path.insert(0, _OSC_DIR)

from components import OscillatorComponent
from dynamics import OscillatorSystem

from numen.compiler.flatten import compile_spec
from numen.spec.world import GenericWorld

TSPAN  = (0.0, 2.0)
OMEGA  = 2 * math.pi       # 1 Hz
REPS   = 5                  # warm-run repetitions per backend

_World = GenericWorld[OscillatorComponent, OscillatorSystem, None]


def make_world(n: int) -> _World:
    """N independent oscillators with staggered initial positions."""
    components = {
        f"osc_{i}": OscillatorComponent(
            position=float(i % 10 + 1) * 0.1,
            velocity=0.0,
            omega=OMEGA,
            damping=0.0,   # undamped — energy conservation is a useful correctness check
        )
        for i in range(n)
    }
    return _World(components=components, systems={"dyn": OscillatorSystem()})


def _timeit(fn, reps: int) -> tuple[float, float]:
    """Return (first_call_s, best_subsequent_s)."""
    t0 = time.perf_counter()
    fn()
    cold = time.perf_counter() - t0

    best = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return cold, best


def run_benchmark(n_list: list[int], include_julia: bool = True) -> None:
    from numen.bridge.scipy_backend import ScipyBackend
    from numen.bridge.jax_backend import JAXBackend

    rtol, atol = 1e-6, 1e-8

    julia_col = "  {'julia ms':>10}" if include_julia else ""
    header = (
        f"\n{'N':>5}  {'state':>5}  "
        f"{'scipy ms':>9}  "
        f"{'JAX cold ms':>12}  {'JAX warm ms':>12}  {'JAX speedup':>11}"
        + (f"  {'julia ms':>10}  {'julia speedup':>13}" if include_julia else "")
    )
    print(header)
    print("-" * (len(header.expandtabs()) + 2))

    for n in n_list:
        world  = make_world(n)
        spec   = compile_spec(world)
        states = spec.state_size

        scipy = ScipyBackend(rtol=rtol, atol=atol)
        jax   = JAXBackend(rtol=rtol, atol=atol, n_saves=200)

        _, scipy_warm = _timeit(lambda: scipy.solve(spec, TSPAN), REPS)
        jax_cold, jax_warm = _timeit(lambda: jax.solve(spec, TSPAN), REPS)

        jax_speedup = scipy_warm / jax_warm if jax_warm > 0 else float("inf")

        julia_str = ""
        if include_julia:
            from numen.bridge.runtime import JuliaBackend
            julia = JuliaBackend(julia_file=_DYNAMICS_JL, rtol=rtol, atol=atol)
            julia_cold, _ = _timeit(lambda: julia.solve(spec, TSPAN), 1)
            julia_speedup = scipy_warm / julia_cold
            julia_str = f"  {julia_cold * 1000:>10.0f}  {julia_speedup:>13.1f}x"

        print(
            f"{n:>5}  {states:>5}  "
            f"{scipy_warm * 1000:>9.1f}  "
            f"{jax_cold * 1000:>12.1f}  {jax_warm * 1000:>12.1f}  {jax_speedup:>11.1f}x"
            + julia_str
        )


def check_accuracy(n: int = 3, include_julia: bool = True) -> None:
    """Spot-check: final states from scipy and JAX should agree to near-tolerance."""
    from numen.bridge.scipy_backend import ScipyBackend
    from numen.bridge.jax_backend import JAXBackend

    world  = make_world(n)
    spec   = compile_spec(world)

    scipy_r = ScipyBackend(rtol=1e-9, atol=1e-9).solve(spec, TSPAN)
    jax_r   = JAXBackend(rtol=1e-9, atol=1e-9, n_saves=len(scipy_r.t)).solve(spec, TSPAN)

    scipy_final = scipy_r.x[:, -1]
    jax_final   = jax_r.x[:, -1]
    max_err = np.max(np.abs(scipy_final - jax_final))
    print(f"\nAccuracy check (N={n}): final-state |scipy − JAX| = {max_err:.2e}")

    if include_julia:
        from numen.bridge.runtime import JuliaBackend
        julia_r = JuliaBackend(julia_file=_DYNAMICS_JL, rtol=1e-9, atol=1e-9).solve(spec, TSPAN)
        julia_final = julia_r.x[:, -1]
        max_err_j = np.max(np.abs(scipy_final - julia_final))
        print(f"Accuracy check (N={n}): final-state |scipy − Julia| = {max_err_j:.2e}")


if __name__ == "__main__":
    args = sys.argv[1:]
    include_julia = "--no-julia" not in args
    n_args = [a for a in args if not a.startswith("--")]
    n_list = [int(x) for x in n_args] if n_args else [1, 5, 20, 50, 100]

    julia_note = "  |  JuliaBackend (subprocess Tsit5)" if include_julia else "  [Julia skipped]"
    print("Numen backend comparison")
    print(f"Problem : independent harmonic oscillators, N={n_list}, tspan=(0, 2 s)")
    print(f"Backends: ScipyBackend (RK45)  |  JAXBackend (diffrax Tsit5){julia_note}")
    print(f"Tolerances: rtol=1e-6, atol=1e-8  |  warm = best of {REPS} runs")
    if include_julia:
        print("Note: Julia time = subprocess launch + JIT + solve (no warm state across calls)")

    check_accuracy(include_julia=include_julia)
    run_benchmark(n_list, include_julia=include_julia)
