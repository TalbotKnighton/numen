# Backends

Numen ships four solver backends. All accept a `CompiledSpec` and return a `SolveResult`.

## Performance comparison

Timings for the fluid poppet example (150 ms pneumatic transient, 6-state system):

| Backend | Warm solve | vs scipy |
|---|---|---|
| `ScipyBackend` (RK45) | 9048 ms | baseline |
| `JAXBackend` (Dopri5) | **6 ms** | **1507×** |
| `JuliaBackend` (Tsit5) | **14 ms** | **634×** |

JAX cold (first call, JIT compile): ~550 ms.
Julia cold (subprocess startup + JIT): ~6700 ms.
`JuliaServerBackend` pays the cold start once per session.

## ScipyBackend

Pure-Python backend using `scipy.integrate.solve_ivp`. Good for development and debugging.

```python
from numen.bridge.scipy_backend import ScipyBackend

result = ScipyBackend(
    method="RK45",     # "RK23", "DOP853", "LSODA"
    rtol=1e-6,
    atol=1e-8,
    n_save_points=500, # optional: limit output density
).solve(spec, tspan=(0.0, 5.0))
```

**Supported features:** `vector_fields`, `discrete_fields`, `continuous_fields`, `control_callbacks`

## JAXBackend

JIT-compiles the full ODE solve via `jax.jit` / `diffrax`. Up to 1500× faster than scipy
for warm solves. The compiled kernel is cached per `(spec, tspan)`.

```python
from numen.bridge.jax_backend import JAXBackend

result = JAXBackend(
    solver="Dopri5",     # explicit: "Tsit5"; implicit: "Kvaerno5"
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

Note: `control_callbacks` and `dae_constraints` are not supported (JAX cannot call Python mid-solve).

## JuliaBackend

Spawns a fresh Julia subprocess per `solve()` call. Best for long single simulations
where the ~6 s cold startup is amortised.

```python
from numen.bridge.runtime import JuliaBackend

result = JuliaBackend(
    julia_file="dynamics.jl",
    method="Tsit5",       # also: "Rodas5P", "FBDF", "Vern7", etc.
    rtol=1e-6,
    atol=1e-8,
    n_save_points=2000,
).solve(spec, tspan=(0.0, 5.0), reps=3)

print(f"JIT: {result.jit_ms:.0f} ms   warm: {result.warm_ms:.0f} ms")
```

**Supported features:** all features including `dae_constraints` and `control_callbacks`.

## JuliaServerBackend

Persistent Julia subprocess — pays the cold start once per session, then all subsequent
solves are warm. Ideal for iterative workflows (parameter studies, interactive sessions).

```python
from numen.bridge.server_backend import JuliaServerBackend

backend = JuliaServerBackend(
    julia_file="dynamics.jl",
    method="Rodas5P",
    rtol=1e-6,
    atol=1e-8,
    n_save_points=2000,
)
# First solve starts the server (~6 s); subsequent calls are warm (~14 ms)
result = backend.solve(spec, tspan=(0.0, 5.0))
```

## JuliaServerPool

N parallel persistent servers for parameter sweeps. Dispatches each design point to
a free worker. All workers precompile dynamics at startup so the first real solve
carries no JIT latency.

```python
from numen.bridge.server_backend import JuliaServerPool

pool = JuliaServerPool(
    n_workers=4,
    julia_file="dynamics.jl",
    method="Tsit5",
    rtol=1e-6,
    atol=1e-8,
    n_save_points=2000,
)
```

Also available via YAML `backend:` section with `n_workers: 4`.

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
