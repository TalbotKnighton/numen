"""
Backend benchmark for the fluid poppet check valve example.

Compares ScipyBackend, JAXBackend (expected incompatible), and JuliaBackend
on the same compiled spec.  Results are printed to stdout and saved to
benchmark_results.txt alongside this file.

Julia timing breakdown (single subprocess, multiple internal solves):
  startup_ms  = Julia boot + package load + dynamics include
  jit_ms      = first solve inside the process (JIT compilation of dynamics)
  warm_ms     = best subsequent solve (pure ODE integration)

Usage:
    python examples/fluid_poppet/benchmark.py
    python examples/fluid_poppet/benchmark.py --no-julia
"""
from __future__ import annotations

import datetime
import math
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from run import make_world
from numen.compiler.flatten import compile_spec
from numen.bridge.scipy_backend import ScipyBackend

TSPAN  = (0.0, 0.15)
REPS   = 5
RTOL   = 1e-8
ATOL   = 1e-10
_DIR   = os.path.dirname(__file__)
_DYN_JL = os.path.join(_DIR, "dynamics.jl")


def _timeit(fn, reps: int) -> tuple[float, float]:
    """(first_call_s, best_subsequent_s)"""
    t0   = time.perf_counter(); fn(); cold = time.perf_counter() - t0
    best = math.inf
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return cold, best


def run_benchmark(include_julia: bool = True) -> str:
    world = make_world()
    spec  = compile_spec(world)

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("=" * 70)
    emit("Fluid Poppet Check Valve — Backend Benchmark")
    emit(f"Date:       {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    emit(f"Problem:    pneumatic 4-CV network + 1-DOF spring-mass poppet")
    emit(f"State size: {spec.state_size}   Param size: {spec.param_size}")
    emit(f"tspan:      {TSPAN}  (150 ms)")
    emit(f"Tolerances: rtol={RTOL}  atol={ATOL}")
    emit(f"Reps (warm timing): {REPS}")
    emit("=" * 70)

    # ------------------------------------------------------------------
    # ScipyBackend
    # ------------------------------------------------------------------
    emit("\n--- ScipyBackend (RK45) ---")
    scipy = ScipyBackend(rtol=RTOL, atol=ATOL)
    scipy_cold, scipy_warm = _timeit(lambda: scipy.solve(spec, TSPAN), REPS)
    r_scipy = scipy.solve(spec, TSPAN)
    emit(f"  cold (first call):  {scipy_cold * 1000:8.1f} ms")
    emit(f"  warm (best of {REPS}):  {scipy_warm * 1000:8.1f} ms")
    emit(f"  output steps:       {len(r_scipy.t)}")

    # ------------------------------------------------------------------
    # JAXBackend
    # ------------------------------------------------------------------
    emit("\n--- JAXBackend (diffrax Dopri5) ---")
    try:
        from numen.bridge.jax_backend import JAXBackend
        jax = JAXBackend(rtol=RTOL, atol=ATOL, n_saves=200, max_steps=100_000, solver="Dopri5")
        jax_cold, jax_warm = _timeit(lambda: jax.solve(spec, TSPAN), REPS)
        emit(f"  cold (first call):  {jax_cold * 1000:8.1f} ms")
        emit(f"  warm (best of {REPS}):  {jax_warm * 1000:8.1f} ms")
        emit(f"  scipy/JAX speedup:  {scipy_warm / jax_warm:.1f}x  (warm vs warm, both Dopri5/RK45)")
        jax_ok = True
    except Exception as e:
        first_line = str(e).split("\n")[0][:120]
        emit(f"  FAILED — {type(e).__name__}: {first_line}")
        jax_ok = False

    # ------------------------------------------------------------------
    # JuliaBackend
    # ------------------------------------------------------------------
    if include_julia:
        emit("\n--- JuliaBackend (subprocess OrdinaryDiffEq Tsit5) ---")
        try:
            from numen.bridge.runtime import JuliaBackend
            julia  = JuliaBackend(julia_file=_DYN_JL, rtol=RTOL, atol=ATOL)
            result = julia.solve(spec, TSPAN, reps=REPS + 1)
            emit(f"  subprocess startup: {result.startup_ms:8.0f} ms  (Julia boot + pkgs + include)")
            emit(f"  JIT solve:          {result.jit_ms:8.1f} ms  (first solve, compiles dynamics)")
            emit(f"  warm solve:         {result.warm_ms:8.1f} ms  (subsequent, compiled)")
            emit(f"  scipy/Julia speedup:{scipy_warm / (result.warm_ms / 1000):8.1f}x  (scipy warm vs julia warm)")
            emit(f"  output steps:       {len(result.t)}")
        except Exception:
            emit("  FAILED")
            emit(traceback.format_exc())
    else:
        emit("\n--- JuliaBackend --- SKIPPED (--no-julia)")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    emit("\n" + "=" * 70)
    emit("Summary (warm solve times — steady-state throughput)")
    emit("-" * 70)
    emit(f"  {'Backend':<30} {'warm ms':>10}  {'vs scipy':>10}")
    emit(f"  {'-'*30} {'-'*10}  {'-'*10}")
    emit(f"  {'ScipyBackend (RK45)':<30} {scipy_warm * 1000:>10.1f}  {'(baseline)':>10}")

    if jax_ok:
        emit(f"  {'JAXBackend (Dopri5)':<30} {jax_warm * 1000:>10.1f}  {scipy_warm / jax_warm:>9.1f}x")
    else:
        emit(f"  {'JAXBackend (Dopri5)':<30} {'incompatible':>10}  {'—':>10}")

    if include_julia:
        try:
            emit(f"  {'JuliaBackend warm (Tsit5)':<30} {result.warm_ms:>10.1f}  {scipy_warm / (result.warm_ms/1000):>9.1f}x")
            emit(f"  {'JuliaBackend+startup':<30} {result.startup_ms + result.jit_ms + result.warm_ms:>10.0f}  {'(cold subprocess)':>10}")
        except Exception:
            pass

    emit("=" * 70)

    return "\n".join(lines)


if __name__ == "__main__":
    include_julia = "--no-julia" not in sys.argv

    output = run_benchmark(include_julia=include_julia)

    out_path = os.path.join(_DIR, "benchmark_results.txt")
    with open(out_path, "w") as f:
        f.write(output)
        f.write("\n")
    print(f"\nResults saved to {out_path}")
