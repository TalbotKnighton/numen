# Numen — Physics Simulation Framework
## Claude Code Guide

Numen is an ECS-style (Entity-Component-System) physics simulation framework.
Models are **defined in Python** using Pydantic components, then solved by
Python (scipy), JAX (diffrax), or Julia (OrdinaryDiffEq) backends.

The `numen` CLI helps you explore, verify, and scaffold new models:

```bash
uv run numen check          # verify all backends work
uv run numen list           # show built-in examples
uv run numen new mymodel    # scaffold a new model directory
uv run numen run oscillator # run a built-in example
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
  └─ dynamics_fn  (Julia name)   Backends: ScipyBackend
                                            JAXBackend
World  (components + systems)              JuliaBackend
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
        components={"ball": BallComponent(position=10.0, velocity=0.0, mass=2.0)},
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
idx = spec.state_index_map["ball.position"][0]
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
ScipyBackend(rtol=1e-8, atol=1e-10)       # uses RK45 (Dormand-Prince)

# JAX (fast repeated solves — ~1500x faster than scipy warm)
JAXBackend(rtol=1e-8, atol=1e-10, solver="Dopri5", max_steps=100_000)

# Julia (long simulations — ~600x faster than scipy warm, subprocess)
JuliaBackend(julia_file="dynamics.jl", rtol=1e-8, atol=1e-10)
```

**Backend warm timings for the fluid poppet example (150 ms pneumatic + poppet):**

| Backend | Warm solve | vs scipy |
|---|---|---|
| ScipyBackend (RK45) | 9048 ms | baseline |
| JAXBackend (Dopri5) | 6 ms | **1507×** |
| JuliaBackend (Tsit5) | 14 ms | 634× |

JAX cold (first call, JIT compile): ~550 ms. Subsequent calls hit the compiled kernel.
Julia cold (subprocess startup + JIT): ~6700 ms. Warm calls reuse compiled dynamics if
you batch multiple solves via `reps` parameter.

---

## Writing Julia dynamics

For the Julia backend, each Python `System` has a corresponding Julia function.
The naming convention is `ModuleName.function_name!` (matching `dynamics_fn` on the System).

```julia
# dynamics.jl
module MyDynamics
import Main: CompiledSpec, CompiledSystemSpec, state_idx, param_idx

function gravity_dynamics!(
    dx::Vector{Float64}, x::Vector{Float64}, p::Vector{Float64},
    t::Float64, spec::CompiledSpec, sys::CompiledSystemSpec,
)
    for id_ball in sys.entity_ids
        i_pos = state_idx(spec, id_ball * ".position")
        i_vel = state_idx(spec, id_ball * ".velocity")
        dx[i_pos] += x[i_vel]
        dx[i_vel] += -9.81
    end
end

end  # module MyDynamics
```

Julia uses the same `state_idx` / `param_idx` helpers that index into the flat
`x` and `p` vectors. The index maps are serialized in the `CompiledSpec` JSON
that the subprocess receives.

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
└── fluid_poppet/          Full: 4-CV pneumatic network + spring-mass poppet
                           ← Best reference for new models
```

---

## SnapshotCollector — accessing results

```python
from numen.reconstruction.collector import SnapshotCollector

collector = SnapshotCollector(world, spec, result)

# Time series for a single field
t, pos = collector.field_series("ball", "position")

# World snapshot at a specific time
snap = collector.at(t=1.5)
ball = snap.components["ball"]   # typed ComponentView
print(ball.position, ball.velocity)
```

---

## Quick-start checklist for a new model

- [ ] One `components.py` — define `Component` subclasses, choose `IntegratedField` vs `ParameterField`
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
