# Backends

Numen ships four solver backends. All accept a `CompiledSpec` and return a `SolveResult`.

!!! tip "Use Julia for production"
    For real engineering work, reach for **`JuliaServerBackend`** (single
    long-lived process, JIT amortised across the session) or
    **`JuliaServerPool`** (N pre-warmed workers for parameter sweeps).

    Julia is a thin wrapper over the full
    [`OrdinaryDiffEq.jl`](https://docs.sciml.ai/OrdinaryDiffEq/stable/)
    ecosystem — **~150+ solvers selectable by string name** — and is the only
    Numen backend that supports **stiff problems** (`Rodas5P`, `FBDF`,
    `KenCarp4`) and **DAEs** (algebraic constraints via mass matrix). JAX's
    explicit solvers diverge on stiff systems that real engineering models
    routinely produce. Use JAX only when you need autodiff *through* the
    solve.

## Performance: don't trust toy benchmarks

Timings for the fluid poppet example (150 ms pneumatic transient, 6-state system):

| Backend | Warm solve | vs scipy |
|---|---|---|
| `JuliaServerBackend` (Tsit5) | **14 ms** | **634×** |
| `JAXBackend` (Dopri5) | **6 ms** | **1507×** |
| `ScipyBackend` (RK45) | 9048 ms | baseline |

JAX cold (first call, JIT compile): ~550 ms. Julia cold (subprocess startup +
JIT): ~6700 ms. `JuliaServerBackend` pays the cold start once per session.

!!! warning "These numbers are misleading for real models"
    The fluid poppet is a deliberately small **non-stiff** problem so it can
    run on every backend. JAX wins it. But representative engineering systems
    (fluid networks, electromechanical coupling, thermal/structural systems)
    are typically stiff — and on those, JAX often **fails outright** while
    Julia's `Rodas5P` + sparse-Jacobian path handles them comfortably. Always
    benchmark your own model. The pneumatic dashpot example is a closer
    proxy: stiff, mass-matrix DAE, parameter sweep.

## Solver ecosystem

The Julia backend gives you the entire `OrdinaryDiffEq.jl` solver set,
selected by string name:

| Family | Solvers | Use for |
|---|---|---|
| Non-stiff explicit RK | `Tsit5`, `Dopri5`, `Vern7`, `Vern9`, `BS3` | Most non-stiff ODEs (default `Tsit5`) |
| Stiff Rosenbrock | `Rodas5P`, `Rodas4`, `Rosenbrock23` | Stiff systems, mass-matrix DAEs |
| Stiff implicit RK / multistep | `KenCarp4`, `KenCarp47`, `TRBDF2`, `FBDF`, `QNDF` | Very stiff or large systems |
| Symplectic | `KahanLi6`, `McAte5`, `VelocityVerlet` | Hamiltonian / energy-preserving |
| IMEX | `KenCarp4`, `ARKODE_ERK_BS3` | Mixed stiff/non-stiff splits |

See the [`OrdinaryDiffEq.jl` solver index](https://docs.sciml.ai/OrdinaryDiffEq/stable/solvers/ode_solve/) for the full list (the `method=` argument forwards directly).

## JuliaServerBackend ⭐

Persistent Julia subprocess — pays the cold start once per session, then all subsequent
solves are warm. **The recommended backend for production work.**

```python
from numen.bridge.server_backend import JuliaServerBackend

with JuliaServerBackend(
    julia_file="dynamics.jl",
    method="Rodas5P",        # any OrdinaryDiffEq.jl solver name
    rtol=1e-8,
    atol=1e-10,
    n_save_points=2000,
) as srv:
    # First solve starts the server (~6 s); subsequent calls are warm (~14 ms)
    for params in trial_set:
        result = srv.solve(compile_spec(make_world(**params)), tspan=(0.0, 5.0))
```

**Supported features:** all features including `dae_constraints` and `control_callbacks`.

## JuliaServerPool

N parallel persistent servers for parameter sweeps. Dispatches each design point to
a free worker. All workers `precompile()` dynamics at startup so the first real solve
carries no JIT latency.

```python
from numen.bridge.server_backend import JuliaServerPool

with JuliaServerPool(
    n_workers=4,
    julia_file="dynamics.jl",
    method="Tsit5",
    rtol=1e-8,
    atol=1e-10,
    n_save_points=2000,
) as pool:
    results = pool.map(
        lambda srv, p: srv.solve(compile_spec(make_world(p)), tspan=(0.0, 1.0)),
        param_grid,
    )
```

Available via YAML `backend:` section with `n_workers: 4`.

## JuliaBackend (one-shot)

Spawns a fresh Julia subprocess per `solve()` call. Use this only for one-off
scripts that exit; for any iterative or repeated work, prefer
`JuliaServerBackend` so the JIT cost is paid once.

```python
from numen.bridge.runtime import JuliaBackend

result = JuliaBackend(
    julia_file="dynamics.jl",
    method="Tsit5",       # any OrdinaryDiffEq.jl solver
    rtol=1e-8,
    atol=1e-10,
    n_save_points=2000,
).solve(spec, tspan=(0.0, 5.0), reps=3)

print(f"JIT: {result.jit_ms:.0f} ms   warm: {result.warm_ms:.0f} ms")
```

**Supported features:** all features including `dae_constraints` and `control_callbacks`.

## ScipyBackend

Pure-Python backend using `scipy.integrate.solve_ivp`. Good for development and debugging
when you want to avoid the Julia install. `LSODA` is the only stiff option here.

```python
from numen.bridge.scipy_backend import ScipyBackend

result = ScipyBackend(
    method="RK45",     # "RK23", "DOP853", "LSODA" (stiff)
    rtol=1e-6,
    atol=1e-8,
    n_save_points=500,
).solve(spec, tspan=(0.0, 5.0))
```

**Supported features:** `vector_fields`, `discrete_fields`, `continuous_fields`, `control_callbacks`

## JAXBackend (autodiff / batched only)

JIT-compiles the full ODE solve via `jax.jit` / `diffrax`. Reach for this only
when you need autodiff *through* the solve (e.g. gradient-based optimisation,
ML-adjacent flows) or GPU-batched solves. Does not support stiff problems
robustly and does not support DAEs.

```python
from numen.bridge.jax_backend import JAXBackend

result = JAXBackend(
    solver="Dopri5",     # explicit: "Tsit5"; implicit: "Kvaerno5" (slow JIT)
    rtol=1e-6,
    atol=1e-8,
    max_steps=100_000,
    n_saves=500,
).solve(spec, tspan=(0.0, 5.0))
```

!!! warning "Tsit5 and tight atol"
    With `atol=1e-10` and state values of order `1e5` (e.g. pressure in Pa),
    Tsit5 causes pathological step rejection. Use `Dopri5` instead.

**Supported features:** `vector_fields`, `discrete_fields`, `continuous_fields`

`control_callbacks` and `dae_constraints` are not supported (JAX cannot call
Python mid-solve, and there's no DAE path).

## Solver selection guide

| Problem type | Recommended backend | Avoid |
|---|---|---|
| Non-stiff ODE | `JAXBackend(solver="Dopri5")` | Tsit5 with tight atol |
| Stiff ODE | `JuliaServerBackend(method="Rodas5P")` | JAX implicit (slow JIT) |
| DAE (algebraic constraints) | `JuliaServerBackend(method="Rodas5P")` | scipy, JAX (unsupported) |
| Development / debugging | `ScipyBackend()` | — |
| Parameter sweep | `JuliaServerPool(n_workers=N)` | JuliaBackend (per-call startup) |
| Control callbacks | `ScipyBackend()` or `JuliaServerBackend()` | JAXBackend |

## Output density controls

All backends support these kwargs to cap output density:

| Kwarg | Default | Meaning |
|---|---|---|
| `n_save_points=N` | 0 (save all) | Save N uniformly-spaced output points |
| `dtsave=dt` | None | Save every `dt` time units (exclusive with `n_save_points`) |
| `dtmax=dt` | None | Cap the adaptive step size |

Rule of thumb: `dtmax = dtsave = 1 / (10 × f_max)` for 10 samples per period
of the highest-frequency content you care about.

## Backend feature compatibility

| Feature | Scipy | JAX | JuliaBackend | JuliaServerBackend |
|---|:---:|:---:|:---:|:---:|
| `vector_fields` | ✓ | ✓ | ✓ | ✓ |
| `discrete_fields` | ✓ | ✓ | ✓ | ✓ |
| `continuous_fields` | ✓ | ✓ | ✓ | ✓ |
| `control_callbacks` | ✓ | — | ✓ | ✓ |
| `dae_constraints` | — | — | ✓ | ✓ |

`compile_spec` detects required features from the field types present. Each backend
checks `required_features ⊆ supported_features` before starting and raises
`NumenFeatureError` with an actionable message if the check fails.

## Logging

```python
from numen.logging import configure_logging
import logging

configure_logging(level=logging.DEBUG)   # solve start/finish + timings + Julia stderr
configure_logging(level=logging.INFO)    # solve start/finish only
```

Logger hierarchy: `numen.backend.scipy`, `numen.backend.jax`, `numen.backend.julia`,
`numen.backend.julia_server`.
