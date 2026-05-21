# Numen — Physics Simulation Framework
## Claude Code Guide

Numen is an ECS-style (Entity-Component-System) physics simulation framework.
Models are **defined in Python** using Pydantic components, then solved by
Python (scipy, dev/debug), Julia (OrdinaryDiffEq, **production**), or JAX
(diffrax, autodiff/batched) backends.

**Backend strategy.** Julia is the strategic backend for production engineering
work — it's a thin wrapper over the full [`OrdinaryDiffEq.jl`](https://docs.sciml.ai/OrdinaryDiffEq/stable/)
ecosystem (~150+ solvers selectable by string), supports stiff problems
(`Rodas5P`, `FBDF`, `KenCarp4`), DAEs (mass-matrix path with
`ContinuousField(algebraic=True)`), and sparse Jacobian with auto-coloring.
Use `JuliaServerBackend` to amortise JIT across the session and `JuliaServerPool`
for parallel parameter sweeps. JAX is only the right answer when you need
autodiff *through* the solve — its explicit solvers diverge on stiff systems
that real engineering models routinely produce.

The `numen` CLI helps you explore, verify, scaffold, and characterize models:

```bash
uv run numen check          # verify all backends work
uv run numen list           # show built-in examples
uv run numen new mymodel    # scaffold a new model directory
uv run numen run oscillator # run a built-in example

uv run numen characterize test_plan.yaml        # run tests + plot
uv run numen characterize test_plan.yaml -c     # characterize only → results.json
uv run numen characterize test_plan.yaml -p     # plot only → reads results.json
```

---

## Architecture in one picture

```
User defines                     Framework provides
─────────────────────────────    ──────────────────────────────
Component  (data)                compile_spec()  →  CompiledSpec
  ├─ IntegratedField (state x)     ├─ x0         (initial state vector)
  └─ ParameterField (param p)      ├─ p          (parameter vector)
                                   ├─ state_index_map
System  (dynamics)                 └─ systems    [CompiledSystem]
  ├─ python_fn    (scipy / JAX)
  └─ dynamics_fn  (Julia name)   Backends: JuliaServerBackend ★ production
                                            ScipyBackend       dev/debug
World  (components + systems)              JAXBackend          autodiff/batch
```

---

## Step-by-step: building a new model

### 1 — Define components

```python
# components.py
from typing import Annotated, Literal
from numen.spec.component import Component
from numen.fields import IntegratedField, ParameterField

class BallComponent(Component):
    kind:     Literal["ball"] = "ball"
    # IntegratedField  →  appears in the state vector x (integrated by the ODE solver)
    position: Annotated[float, IntegratedField()] = 0.0
    velocity: Annotated[float, IntegratedField()] = 0.0
    # ParameterField  →  appears in the parameter vector p (constant during a solve)
    mass:     Annotated[float, ParameterField()]  = 1.0
```

Rules:
- `kind` must be a unique `Literal` string — it's Pydantic's discriminator
- Default values are the initial conditions / parameter values when unspecified
- Multiple `IntegratedField`s on one component are fine (e.g. position + velocity)

#### Field types at a glance

| Field | Vector in | When updated | Backed? |
|---|---|---|---|
| `IntegratedField()` | state `x` | Every ODE step (solver integrates) | ✓ all backends |
| `ParameterField()` | param `p` | Never (constant for the solve) | ✓ all backends |
| `ContinuousField()` | state `x` | Every RHS call (dynamics fn writes) | ✓ (output/algebraic slot) |
| `DiscreteField(dt)` | state `x` | Forces solver tstops at multiples of `dt` | ✓ tstops; controller callback pending |
| `ExcitationPort()` | *(none)* | Pure annotation metadata — NOT compiled | ✓ (characterization only) |

`ExcitationPort` fields are **not** compiled into the parameter vector `p`. They are pure annotation metadata used by the characterization framework to identify which fields accept excitation signals. The compiler ignores them; no slot is allocated in `x` or `p`.

**Injection is unit-agnostic.** When you call `inject_excitation`, `inject_chirp_excitation`, or `inject_table_excitation`, the computed value F(t) is added **directly** to `d(target_field)/dt` — no automatic unit conversion. `port_type` (`"effort"` / `"flow"`) and `units` are metadata for plot labels and bond-graph conventions only; they do **not** affect the math.

This means if your `ExcitationPort` targets a velocity-like state, "F in Newtons" with no scaling is actually being interpreted as an acceleration [m/s²]. To convert correctly, use the `scale_by` kwarg to **divide** by a parameter:

```python
spec = inject_excitation(
    spec,
    entity_id="osc", component_kind="mass_body",
    port_name="force", target_field="velocity",
    amp=10.0,                      # Newtons
    scale_by="osc.mass_body.mass", # full path of the ParameterField to divide by
)
```

In YAML test plans, set `excitation.scale_by` (full path `entity.component.field`) at the top of the campaign. The same divisor applies to all injection helpers, and it composes with `excitation.*` sweeps — sweeping the amplitude in Newtons is preserved, division by mass happens inside the RHS at every step.

#### Vector / array fields — `size=N`

Any field can hold a contiguous array by setting `size=N`. The compiler packs it
as `N` consecutive slots in `x` (or `p`). **This works today across all backends.**

```python
class VibeComponent(Component):
    kind:        Literal["vibe"] = "vibe"
    # N=8 frequency/amplitude/phase bins — packed into p[start:start+8]
    frequencies: Annotated[list[float], ParameterField(size=8)] = [0.0] * 8
    amplitudes:  Annotated[list[float], ParameterField(size=8)] = [0.0] * 8
    phases:      Annotated[list[float], ParameterField(size=8)] = [0.0] * 8
```

- In dynamics: `spec.view(eid, VibeComponent, x, p).frequencies` returns a numpy/JAX slice.
- In Julia: `p[param_idx(spec, "$id.vibe.frequencies") : param_idx(spec, "$id.vibe.frequencies") + 7]`
  or use `param_slice` helper (returns a `UnitRange{Int}`).
- Size is inferred at `compile_spec()` time from the field annotation — no need to
  specify it twice; just make sure the default value has the right length.

### 2 — Define dynamics functions

```python
# dynamics.py
from typing import ClassVar, Literal
import jax.numpy as jnp                      # ALWAYS jnp, never np, inside dynamics
from numen.compiler.flatten import CompiledSpec, CompiledSystem
from numen.fields import EntityGroup
from numen.spec.system import System, DynamicsFn
from components import BallComponent

def gravity_dynamics(dx, x, p, t, spec, system):
    """ẍ = -g  for every ball entity."""
    for (entity_id,) in system.entity_groups:
        ball = spec.view(entity_id, BallComponent, x, p)   # read state + params
        db   = spec.dx_view(entity_id, BallComponent, dx)  # write derivatives
        db.position += ball.velocity
        db.velocity += -9.81

class GravitySystem(System):
    component_types: ClassVar[tuple[type, ...]] = (BallComponent,)  # auto-populates entity_groups
    python_fn:       ClassVar[DynamicsFn]       = staticmethod(gravity_dynamics)
    kind:            Literal["gravity"]         = "gravity"
    dynamics_fn:     str = "MyDynamics.gravity_dynamics!"   # Julia module.function!
```

Key rules for dynamics functions:
- `spec.view(id, ComponentType, x, p)` → **read-only** view of one entity
- `spec.dx_view(id, ComponentType, dx)` → **write** view; use `+=` to accumulate
- Multiple systems writing the same field accumulate correctly via `+=`
- **Never call** `spec.view` with the wrong `ComponentType` — it will silently read garbage

### 3 — Multi-entity systems (topology lives in the system, not components)

When a system couples multiple entities (spring between two masses, orifice between two control volumes), declare the slots in `entity_slots` and provide `entity_groups` at instantiation:

```python
class SpringForceSystem(System):
    component_types: ClassVar[tuple[type, ...]] = ()   # no auto-population
    entity_slots:    ClassVar[EntityGroup]      = EntityGroup(
        MassComponent, SpringComponent, MassComponent  # defines group_size=3
    )
    python_fn:  ClassVar[DynamicsFn] = staticmethod(spring_force_dynamics)
    kind:       Literal["spring_force"] = "spring_force"
    dynamics_fn: str = "SpringDynamics.spring_force_dynamics!"

# Instantiate with explicit topology
SpringForceSystem(entity_groups=[
    ["m1", "spring1", "m2"],
    ["m2", "spring2", "m3"],
])
```

### 4 — Assemble the world

```python
# world.py
from typing import Annotated, Union
from pydantic import Field
from numen.spec.world import GenericWorld
from components import BallComponent
from dynamics import GravitySystem

AnyComponent = Annotated[Union[BallComponent], Field(discriminator="kind")]
AnySystem    = Annotated[Union[GravitySystem], Field(discriminator="kind")]
World        = GenericWorld[AnyComponent, AnySystem, None]

def make_world():
    return World(
        components={"ball": {"ball": BallComponent(position=10.0, velocity=0.0, mass=2.0)}},
        systems={"gravity": GravitySystem()},
    )
```

### 5 — Compile and solve

```python
# run.py
from numen.compiler.flatten import compile_spec
from numen.bridge.scipy_backend import ScipyBackend
from world import make_world

world  = make_world()
spec   = compile_spec(world)
result = ScipyBackend(rtol=1e-8, atol=1e-10).solve(spec, tspan=(0.0, 2.0))

# result.t  : shape (n_steps,)
# result.x  : shape (state_size, n_steps)
# Access individual fields:
idx = spec.state_index_map["ball.ball.position"][0]
position = result.x[idx]   # array of position over time
```

---

## JAX compatibility rules  ⚠️ critical

If your dynamics function uses Python-native operations on state values, JAX will
raise `TracerBoolConversionError` or `TracerArrayConversionError` at trace time.

| Instead of | Use |
|---|---|
| `if P_a > P_b:` | `jnp.where(P_a > P_b, ..., ...)` |
| `np.maximum(0, x)` | `jnp.maximum(0.0, x)` |
| `np.sqrt(x)` | `jnp.sqrt(x)` |
| `np.clip(x, 0, 1)` | `jnp.clip(x, 0.0, 1.0)` |
| `np.where(...)` | `jnp.where(...)` |

**Both branches of every `jnp.where` are always evaluated** (even the false branch).
Guard against NaN/Inf in both:

```python
# BAD — unchoked branch blows up when P_dn > P_up
mdot = jnp.where(choked, mdot_choked, jnp.sqrt(P_up - P_dn))

# GOOD — guard the sqrt
arg  = jnp.maximum(0.0, P_up - P_dn)
mdot = jnp.where(choked, mdot_choked, jnp.sqrt(arg))
```

**Import**: use `import jax.numpy as jnp` at the top of `dynamics.py`.
Both `jnp.*` and plain Python math (`+`, `*`, `**`) work for both numpy and JAX arrays.

---

## Smooth contact / hard-stop forces  ⚠️ solver performance

A sharp `max(0, -pos)` stop force has a C0 discontinuity at contact onset.
ODE solvers (both scipy and JAX) take thousands of rejected steps at that point.

**Always use a smooth ramp over 1 µm when implementing contact forces:**

```python
_STOP_DELTA = 1e-6   # 1 µm

def _soft_pen(pos_from_stop):
    """C1-smooth approximation of max(0, pos_from_stop)."""
    x = pos_from_stop
    return jnp.where(
        x <= 0.0, 0.0,
        jnp.where(x >= _STOP_DELTA, x - 0.5 * _STOP_DELTA, 0.5 * x * x / _STOP_DELTA)
    )

# Velocity damping: blend in over the same distance
alpha = jnp.clip(-pos / _STOP_DELTA, 0.0, 1.0)    # 0 when open, 1 when 1µm in
v_damp = jnp.maximum(0.0, -vel) * alpha

F_stop = k_stop * _soft_pen(-pos) + c_stop * v_damp
```

See `examples/fluid_poppet/dynamics.py` for the complete pattern with both open and closed stops.

---

## Choosing a solver

**Tsit5 is sensitive to absolute tolerance.**
With `atol=1e-10` and state values of order `1e5` (e.g. pressure in Pa),
Tsit5 produces pathological step rejection (>99%). Use Dopri5 instead:

```python
# scipy (development / debugging)
ScipyBackend(rtol=1e-8, atol=1e-10)

# JAX (fast repeated solves — ~1500x faster than scipy warm)
JAXBackend(rtol=1e-8, atol=1e-10, solver="Dopri5", max_steps=100_000)

# Julia single-shot (long simulations, ~600x faster than scipy warm)
JuliaBackend(julia_file="dynamics.jl", rtol=1e-8, atol=1e-10)

# Julia persistent server (pay JIT once; all solves reuse compiled kernel)
JuliaServerBackend(julia_file="dynamics.jl", method="Tsit5",
                   rtol=1e-8, atol=1e-10, n_save_points=2000)

# Julia pool (parallel parameter sweeps — N processes, DOE-level dispatch)
JuliaServerPool(n_workers=4, julia_file="dynamics.jl", method="Tsit5",
                rtol=1e-8, atol=1e-10, n_save_points=2000)
```

**Backend warm timings for the fluid poppet example (150 ms pneumatic + poppet):**

| Backend | Warm solve | vs scipy |
|---|---|---|
| ScipyBackend (RK45) | 9048 ms | baseline |
| JAXBackend (Dopri5) | 6 ms | **1507×** |
| JuliaBackend (Tsit5) | 14 ms | 634× |

JAX cold (first call, JIT compile): ~550 ms. Subsequent calls hit the compiled kernel.
Julia cold (subprocess startup + JIT): ~6700 ms. `JuliaServerBackend` amortises JIT
across all solves in a session. `JuliaServerPool` additionally runs `precompile()` on
all dynamics functions after startup so the first real solve carries no JIT latency.

### Output density and step control

All Julia backends and ScipyBackend accept these kwargs (also available in YAML `backend:`):

| Kwarg | Default | Meaning |
|---|---|---|
| `n_save_points=N` | 0 | Save N uniformly-spaced output points instead of every adaptive step |
| `dtsave=dt` | None | Save every `dt` time units (mutually exclusive with `n_save_points`) |
| `dtmax=dt` | None | Cap the adaptive step size — prevents aliasing or missing brief transients |
| `maxiters=N` | None | Raise OrdinaryDiffEq's iteration cap (default 1e5). Required for long chirps or fine `dtmax` that need millions of steps. Julia backends only. |
| `precompile=False` | True | (`JuliaServerBackend` / `JuliaServerPool` only) Skip the startup `precompile()` pass on dynamics functions. Saves a few seconds at server startup; first solve pays full JIT cost. Useful for short campaigns or development cycles. |

Rule of thumb: `dtmax = dtsave = 1 / (10 × f_max)` for 10 samples per period of the
highest-frequency content you care about.

If a chirp test terminates early with `"Interrupted. Larger maxiters is needed."`,
raise `maxiters` (e.g. `maxiters=10_000_000`) — a 120-second chirp at
`dtmax=1e-4` requires ≥ 1.2M steps, well above the default cap.

### Stiff solvers (Rodas5P, Rodas4, Rosenbrock23, …)

Rosenbrock-type stiff solvers need both a state Jacobian ∂f/∂x and a time
gradient ∂f/∂t.  Numen handles both automatically:

**State Jacobian** — OrdinaryDiffEq uses ForwardDiff with a sparse
`jac_prototype` built from the ECS entity-group graph.  `compile_spec()` computes
the Jacobian sparsity pattern: within each system's entity group all state slots
are treated as fully coupled (dense block); states from different groups are
independent.  Julia passes this as a `SparseMatrixCSC` to `ODEFunction`, and
OrdinaryDiffEq applies SparseDiffTools matrix coloring.  The effective color count
equals the maximum coupled-state width per group — typically 4–16 — regardless of
how many entities the system has.  This makes Jacobian cost O(1) in the number of
entities for models with bounded per-group coupling.

**Time gradient** — OrdinaryDiffEq normally computes ∂f/∂t by calling the
dynamics function with `t::ForwardDiff.Dual`.  This fails because the solver wraps
the dynamics closure in a `FunctionWrapper{…, Float64}` at problem-construction
time (inferred from `tspan::Tuple{Float64,Float64}`), which rejects Dual `t` at
runtime.  Numen provides an explicit `tgrad!` built by `build_tgrad()` in
`solver.jl` using central finite differences: two Float64 RHS calls per step,
constant cost regardless of state size.  This leaves the ForwardDiff Jacobian path
untouched while fully bypassing the FunctionWrapper restriction.

**User dynamics signatures** — the two-parameter convention `{T, S}` is still
required for scalar helper functions:

```julia
function soft_pen(x::T) where T <: Real
    x <= 0.0 && return zero(T)
    ...
end
```

`t :: Real` in user dynamics functions covers both `Float64` (normal/Jacobian
calls) and `ForwardDiff.Dual` (any future path where Dual `t` is needed).

**Why NOT `autodiff=AutoFiniteDiff()`** — this switches the Jacobian to finite
differences too.  For n states it costs n RHS evaluations per Jacobian vs.
O(color_count) with sparse ForwardDiff.  For large models this is catastrophically
slower.  Always use the `jac_prototype` + explicit `tgrad!` combination instead.

---

## Backend feature compatibility

Each backend declares `supported_features`. `compile_spec()` sets `required_features`
on the `CompiledSpec` based on the field types present. Every `solve()` call checks
compatibility **before starting** and raises `NumenFeatureError` with an actionable
message if the backend can't handle the spec.

| Feature flag | Scipy | JAX | JuliaBackend | JuliaServerBackend |
|---|:---:|:---:|:---:|:---:|
| `vector_fields` (size > 1) | ✓ | ✓ | ✓ | ✓ |
| `discrete_fields` (DiscreteField) | ✓ | ✓ | ✓ | ✓ |
| `continuous_fields` (ContinuousField) | ✓ | ✓ | ✓ | ✓ |
| `control_callbacks` (future) | — | — | — | — |

Adding a new feature to a backend: add the string to that backend's `supported_features`
class var in `bridge/*.py`. Removing support: remove it — affected models will fail early
with a clear message instead of a cryptic solver crash.

### Logging

Enable numen's structured logging to see diagnostics, timings, and Julia output:

```python
from numen.logging import configure_logging
import logging
configure_logging(level=logging.DEBUG)   # everything
configure_logging(level=logging.INFO)    # solve start/finish only
```

Logger hierarchy: `numen.backend.scipy`, `numen.backend.jax`, `numen.backend.julia`,
`numen.backend.julia_server`. Julia stderr lines are routed to `numen.backend.julia*`
at DEBUG level in real time (not just on failure).

---

## Writing Julia dynamics

See **[src/numen/init_data/JULIA.md](src/numen/init_data/JULIA.md)** for the
complete reference: high-level helper API (`get_state`, `get_param`,
`add_deriv!`), low-level form for hot loops, performance trade-offs, and
`groups(sys)` destructuring.

Quick template (single-entity gravity):

```julia
# dynamics.jl
module MyDynamics
import Main: CompiledSpec, CompiledSystemSpec, groups,
             get_state, get_param, add_deriv!

function gravity_dynamics!(
    dx :: AbstractVector{T}, x :: AbstractVector{S}, p :: Vector{Float64},
    t  :: Real, spec :: CompiledSpec, sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    for (id_ball,) in groups(sys)
        vel = get_state(spec, x, id_ball, "ball.velocity")
        add_deriv!(spec, dx, id_ball, "ball.position", vel)
        add_deriv!(spec, dx, id_ball, "ball.velocity", -9.81)
    end
end

end  # module MyDynamics
```

Multi-entity coupled system: use tuple destructuring on `groups(sys)`. For
``EntityGroup(MassComponent, SpringComponent, MassComponent)`` write
``for (id_a, id_s, id_b) in groups(sys)``.

The lower-level helpers `state_idx` / `param_idx` / `state_range` /
`param_range` are also available — use them in performance-critical inner loops
where a single index can be cached and reused for both reads and writes.

**Julia smooth contact helper** (mirror the Python `_soft_pen`):

```julia
const STOP_DELTA = 1e-6
function soft_pen(x::Float64)::Float64
    x <= 0.0 && return 0.0
    x >= STOP_DELTA && return x - 0.5 * STOP_DELTA
    return 0.5 * x * x / STOP_DELTA
end
```

---

## Common physics patterns

### Isentropic compressible orifice flow

```python
def _orifice_mdot(P_up, P_dn, T_up, R, Cd, A, gamma):
    safe_P_up = jnp.maximum(P_up, 1e-300)
    beta      = jnp.maximum(0.0, P_dn) / safe_P_up
    beta_crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    choke_exp = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    mdot_choked   = Cd * A * P_up * jnp.sqrt(gamma / (R * T_up)) * (2.0 / (gamma+1.0))**choke_exp
    arg           = beta**(2/gamma) - beta**((gamma+1)/gamma)
    mdot_unchoked = Cd * A * P_up * jnp.sqrt(jnp.maximum(0.0, 2*gamma/((gamma-1)*R*T_up)*arg))
    mdot = jnp.where(beta <= beta_crit, mdot_choked, mdot_unchoked)
    return jnp.where((P_up <= 0.0) | (A <= 0.0), 0.0, mdot)
```

### Isothermal ideal-gas control volume

```python
# dP/dt = (R·T/V) · ṁ_net
da.pressure += -(cv.R_specific * cv.temperature / cv.volume) * mdot_out
db.pressure +=  (cv.R_specific * cv.temperature / cv.volume) * mdot_in
```

### Spring-mass with hard stops

See `examples/fluid_poppet/dynamics.py` — `poppet_mechanics_dynamics()`.

---

## Repository layout

```
src/numen/
├── fields.py              IntegratedField, ParameterField, EntityGroup
├── spec/
│   ├── component.py       Component base class
│   ├── system.py          System base class, DynamicsFn type alias
│   └── world.py           GenericWorld[C, S, E]
├── compiler/
│   └── flatten.py         compile_spec(), CompiledSpec, DxBuffer, DerivativeView
├── bridge/
│   ├── scipy_backend.py   ScipyBackend
│   ├── jax_backend.py     JAXBackend
│   └── runtime.py         JuliaBackend, SolveResult
└── reconstruction/
    ├── collector.py        SnapshotCollector — post-process results by entity
    └── snapshot.py         WorldSnapshot

examples/
├── oscillator/            Minimal: single 1D harmonic oscillator
├── coupled_spring/        Multi-entity: 3-mass spring chain
├── fluid_poppet/          Full: 4-CV pneumatic network + spring-mass poppet
│                          ← Best reference for complex multi-system models
├── nonlinear_oscillator/  Characterization campaign reference (FRF, chirp, parameter sweep)
└── pneumatic_dashpot/     Piston with orifice vents — stiff ODE, Rodas5P, orifice sweep
                           ← Best reference for stiff systems + parameter sweeps
```

---

## Controller callbacks

Callbacks fire at a fixed period `dt` and can read and write state.

```python
# dynamics.py
def pid_controller(t, x, p, spec):
    """scipy-side: returns {field_key: new_value}."""
    err = x[spec.state_idx("sensor.sensor.angle")] - p[spec.param_idx("ctrl.pid.setpoint")]
    return {"actuator.actuator.force": p[spec.param_idx("ctrl.pid.kp")] * err}

class PIDCallback(Callback):
    kind:      Literal["pid"] = "pid"
    dt:        float = 0.01            # fires every 10 ms
    julia_fn:  str   = "MyDyn.pid!"   # resolved from already-loaded Julia scope
    params:    dict[str, float] = {"kp": 1.0, "setpoint": 0.5}
    python_fn: ClassVar = staticmethod(pid_controller)
```

```python
# world.py
from numen.spec.callback import Callback
...
World = GenericWorld[AnyComponent, AnySystem, Annotated[PIDCallback, Field(discriminator="kind")]]
world = World(
    components={...},
    systems={...},
    callbacks={"ctrl": PIDCallback(dt=0.01, params={"kp": 2.0, "setpoint": 0.0})},
)
```

**Julia callback** (in `dynamics.jl`):
```julia
function pid!(integrator, spec, params)
    i_force  = state_idx(spec, "actuator.actuator.force")
    i_angle  = state_idx(spec, "sensor.sensor.angle")
    err = integrator.u[i_angle] - params["setpoint"]
    integrator.u[i_force] = params["kp"] * err
end
```

**Backend behaviour:**
- **Scipy** — segment-solve: stops at each `dt`, runs `python_fn`, restarts. Zero simulation-time jitter.
- **JAX** — `NumenFeatureError("control_callbacks")` — JIT cannot call Python mid-solve.
- **Julia** — `PeriodicCallback` fires inside the integrator at `t0+dt, t0+2dt, …`.

Non-commensurate rates (e.g., 100 Hz + 75 Hz) are handled by merging all fire times into one timeline. Tstops within 1 µs are collapsed so both callbacks fire together.

---

## DAE — algebraic constraints (`ContinuousField(algebraic=True)`)

For hard constraints with no time derivative (pressure equality, joint constraints):

```python
class CoupledVolume(Component):
    kind:       Literal["cv"] = "cv"
    pressure:   Annotated[float, IntegratedField()] = 1e5
    # Algebraic constraint — dynamics fn writes residual g(x)=0
    p_balance:  Annotated[float, ContinuousField(algebraic=True)] = 0.0
```

The dynamics function writes a **residual** (not a derivative) for algebraic slots:
```python
def balance_dynamics(dx, x, p, t, spec, system):
    for id_a, id_b in system.entity_groups:
        a  = spec.view(id_a, CoupledVolume, x, p)
        b  = spec.view(id_b, CoupledVolume, x, p)
        da = spec.dx_view(id_a, CoupledVolume, dx)
        da.p_balance += a.pressure - b.pressure   # residual = 0 enforces P_a = P_b
```

**Requirements:**
- Julia-only: scipy and JAX raise `NumenFeatureError("dae_constraints")`.
- **Must use an implicit solver**: `method="Rodas5P"` (recommended) or `"FBDF"` for very stiff systems. Passing an explicit solver (Tsit5, Dopri5, Vern7) raises a Julia error before solving.

**Why `differential_mask` is always 0/1:**
Physical capacitances (mass, heat capacity, fluid volume) are always divided out
inside the dynamics function — never placed in the mask. The mask is structural
only: 1 = this slot has a time derivative, 0 = algebraic. See `docs/architecture.md`.

---

## SnapshotCollector — accessing results

```python
from numen.reconstruction.collector import SnapshotCollector

collector = SnapshotCollector(world, spec, result)

# Time series for a single field — 3 args: entity_id, component_kind, field_name
t, pos = collector.field_series("ball", "ball", "position")

# World snapshot at a specific time
snap = collector.at(t=1.5)
ball = snap.components["ball"]["ball"]   # typed ComponentView
print(ball.position, ball.velocity)
```

---

## Quick-start checklist for a new model

- [ ] One `components.py` — define `Component` subclasses; use `IntegratedField`, `ParameterField`, or `size=N` for vectors
- [ ] One `dynamics.py` — write dynamics functions using `jnp.*`, define `System` classes
- [ ] One `world.py` — define `World` type alias, write `make_world()` factory
- [ ] One `run.py` — call `compile_spec`, choose backend, post-process with `SnapshotCollector`
- [ ] One `dynamics.jl` — mirror the Python dynamics for `JuliaBackend` (optional but fast)
- [ ] JAX check: no bare `if` on state values, no `np.*` inside dynamics
- [ ] Contacts / stops: use `_soft_pen` smooth ramp, not bare `jnp.maximum`
- [ ] Solver: use `Dopri5` for JAX if using tight absolute tolerances

---

## Design intent and future directions

See `DESIGN.md` for the full architectural rationale. Key open questions:

- `build_mtk_system` — translate `CompiledSpec` to ModelingToolkit.jl for DAE problems
  (joint constraints, algebraic pressure balance). Currently unimplemented.
- Multibody.jl — 3D constrained mechanisms (deployable structures, robot arms).
  Requires the DAE path above.
- Automatic Julia codegen — SymPy → `sympy.julia_code()` could auto-generate `dynamics.jl`
  from Python symbolic expressions. Currently hand-written; see DESIGN.md for the trade-off.

---

## Characterization framework

The characterization system runs test campaigns from a single YAML file and
generates plots without any Python scripting.  See `CHARACTERIZATION.md` (in
every project scaffolded with `numen new`) for the full reference.

### Test types

| Type | What it does |
|---|---|
| `discrete_frequency_sweep` | Stepped sine; one solve per freq; lock-in FRF |
| `continuous_chirp` | Single chirp solve; fast survey |
| `amplitude_sweep` | Fixed freq, varying amplitude; nonlinearity signature |
| `dc_operating_point_sweep` | DC bias sweep; small-signal FRF at each point |
| `parameter_sweep` | Outer loop over one param; inner loop is any test above |
| `parameter_grid` | Outer grid over multiple params (full_factorial or pairs) |
| `doe_sweep` | Space-filling / classical DOE over continuous ranges |

Planned nonlinear test types (see `docs/plan_nonlinear_test_suite.md`):

| Type | What it does |
|---|---|
| `two_tone` | Two simultaneous sinusoids; extracts IM products, IP3, IMD₃ |
| `harmonic_distortion_sweep` | Stepped sine + THD; measures H₂, H₃ vs. frequency |
| `free_decay` | Ring-down + Hilbert transform; backbone curve, amplitude-dep. damping |
| `phase_portrait` | Steady-state limit cycle in (x, ẋ); Poincaré section for chaos detection |
| `broadband_noise` | Band-limited noise; Best Linear Approximation + NL distortion spectrum |

### `excitation.*` parameter paths

`parameter_sweep`, `parameter_grid`, and `doe_sweep` can vary excitation inputs
as the outer dimension — not just model `ParameterField` values:

```yaml
- name: frf_vs_dc
  type: parameter_sweep
  sweep_param: excitation.dc_offset   # also: excitation.amplitude, excitation.frequency
  values: [0.0, 0.3, 0.6, 1.0]
  sub_test: baseline_frf              # any other test in the same plan
```

This is how DC-offset Bode families are generated: outer loop sets equilibrium
bias, inner loop runs the FRF.  The result is a `ParameterFamilyResult` rendered
by the `parameter_family` plot panel (curves coloured by sweep parameter).

⚠️ **Mandatory parameter key format.** Model parameter keys (`sweep_param`,
`params:` dict keys, etc.) must use the full three-level path
`entity_id.component_kind.field_name` — e.g. `piston.pneumatic_dashpot.orifice_area`,
never `piston.orifice_area`. Excitation parameters use the `excitation.*`
prefix. The runner validates every test's keys at `__init__` and raises
`KeyError` with the full list of valid parameters before any backend opens.

A JSON Schema for `test_plan.yaml` ships with the package; reference it from
the top of any test plan for IDE autocomplete:

```yaml
# yaml-language-server: $schema=test_plan.schema.json
```

`numen init` and `numen new` install this file in the project root; regenerate
with `numen schema -o test_plan.schema.json` after upgrading.

### Parallel execution

Add `n_workers: N` to `backend:` to run sweep tests in parallel across N Julia
server processes.  Each worker precompiles all dynamics at startup (via Julia's
`precompile()`) so the first real solve carries no JIT latency.

```yaml
backend:
  type: julia_server
  julia_file: dynamics.jl
  n_workers: 4           # 4 parallel processes
  n_save_points: 2000    # cap output size to avoid huge JSON payloads
  maxiters: 10_000_000   # raise iteration cap for long chirps / fine dtmax
  precompile: false      # skip startup precompile() (faster boot, slower first solve)
```

Set `precompile: false` when the campaign is too short for the precompile pass
to amortise — e.g. during dynamics development, or campaigns with only a handful
of solves.  The default is `true`.

CLI override: `numen characterize test_plan.yaml --workers 4`

### Custom excitation functions

Built-in `inject_excitation` (sine), `inject_chirp_excitation`, and
`inject_table_excitation` cover the common cases.  For arbitrary user-defined
forcing — gated sinusoids, step responses, shocks, bursts, pulse trains — use
`inject_custom_excitation`:

```python
from numen.characterization.excitation import inject_custom_excitation
import math

def my_gate(t, amp, freq, t_on, t_off):
    if t_on <= t < t_off:
        return amp * math.sin(2 * math.pi * freq * t)
    return 0.0

spec = inject_custom_excitation(
    spec,
    entity_id      = "osc",
    component_kind = "squeeze_film_osc",
    port_name      = "force",
    target_field   = "velocity",
    params         = {"amp": 4.9, "freq": 1250.0, "t_on": 0.1, "t_off": 0.4},
    python_fn      = my_gate,            # f(t, amp, freq, t_on, t_off) -> float
    julia_fn       = "MyDyn.my_gate_dyn!",
    scale_by       = "osc.squeeze_film_osc.mass",   # optional divisor
)
```

All entries of `params` live in the parameter vector `p`, so they are
sweepable via `sweep_param: excitation.amp` (etc.).  The helper inspects
`python_fn`'s signature and raises `ValueError` if it doesn't match
`("t",) + tuple(params.keys())`.

For Julia backends, the user must define `julia_fn` in their `dynamics.jl`.
The cleanest pattern uses `make_custom_excitation_dyn` to wrap a
human-readable function:

```julia
module MyDyn
function my_gate(t, amp, freq, t_on, t_off)
    (t < t_on || t >= t_off) && return 0.0
    return amp * sin(2π * freq * t)
end

# One-line wrapper — type-stable, zero per-step dispatch overhead
const my_gate_dyn! = Main.NumenCharacterization.make_custom_excitation_dyn(
    my_gate, ("amp", "freq", "t_on", "t_off"),
)
end
```

**Constraints** (same as built-in `inject_*`):
- The function reads `t` and its own params only — it cannot read state `x`.
  For feedback / closed-loop behaviour, use a `Callback`, not excitation.
- One target field per injection.  Call the helper twice for two-port forcing.

- `parameter_sweep`, `parameter_grid`, `doe_sweep` dispatch each design point to
  a free worker in parallel; results are returned in input order.
- Top-level leaf tests (chirp, FRF, amplitude sweep) run on one worker sequentially.

### Key implementation detail

Inner test runners (`freq_sweep`, `amplitude_sweep`, `chirp_sweep`) do NOT apply
`dc_offset` themselves.  `CharacterizationRunner._run_leaf_test` pre-applies the
test's own `dc_offset` once before calling the runner (standalone path).  The
sub-runner path passes `spec_v` (outer-sweep DC already set) straight through.
This is why outer `excitation.*` sweeps work correctly.

### Julia server response protocol

Server responses use a chunked line protocol (since the last major update):
- Line 1: `{"n_t": <int>, "n_states": <int>}` — header
- Line 2: `[t0, t1, …]` — time vector
- Lines 3…: one line per state row

This prevents single huge `readline()` calls for long simulations.  Use
`n_save_points` or `dtsave` in the backend config to cap output density.

### Plot panel types

`bode`, `chirp_timeseries`, `amplitude_sweep`, `dc_sweep`, `parameter_family`,
`doe_scatter`, `parameter_grid_heatmap`.  All configured in the `plots:` section
of the same YAML file as the tests.  Use `enabled: false` on any test or panel
to skip it without deleting it.

---

## Documentation workflow

Active design work lives in `docs/plan_*.md` files. These are **living documents**:
every time a design decision is refined during implementation, update the relevant
plan file in the same commit as the code change. Never let the plan drift from the
code.

When a feature is complete, its design rationale graduates:
- Architectural decisions → `DESIGN.md`
- Model-authoring guidance (field types, patterns, gotchas) → this file (`CLAUDE.md`)

Current active plans:

| File | Status | Contents |
|---|---|---|
| `docs/plan_characterization_framework.md` | Complete | Characterization test framework: ExcitationPort, test types, DOE, bond graph abstraction, Julia-first architecture |
| `docs/plan_characterization_plots.md` | Complete | YAML-driven plots: `-c`/`-p` CLI, `plots:` schema, all panel types, excitation.* outer-loop sweeps |
| `docs/plan_parallel_characterization.md` | Complete | Parallel characterization: `n_workers`, DOE-level dispatch, `JuliaServerPool`, `precompile()`, chunked response protocol |
| `docs/plan_symbolic_codegen.md` | Planned | SymPy → Julia auto-codegen to eliminate dual Python/Julia dynamics authoring |
| `docs/plan_nonlinear_test_suite.md` | Planned | Two-tone/IMD, harmonic distortion sweep, free-decay backbone, phase portrait, broadband noise — five new test types |
| `docs/plan_random_vibe_testing.md` | Planned | Random vibe testing: PSD-driven stochastic excitation via table lookup in p; psd_profile / psd_file / multisine / time_series_file input formats; --seed CLI override; BLA / coherence / crest-factor analysis |

When starting a new session, check `docs/` for active plans before writing any
code — they contain the full context of prior design decisions that should not
be relitigated.
