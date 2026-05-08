# {project_name} — Numen Physics Simulation Project

This project uses the **Numen** framework for engineering dynamics simulation.
Models are defined in Python and solved by scipy, JAX, or Julia backends.

```bash
numen check                   # verify scipy + JAX + Julia
numen info                    # framework quick-reference
numen characterize test_plan.yaml        # run tests + plot
numen characterize test_plan.yaml -c     # characterize only
numen characterize test_plan.yaml -p     # plot only
```

---

## Framework overview

```
Component  (Pydantic data)          compile_spec(world) → CompiledSpec
  ├─ IntegratedField → state x                ├─ x0, p  (initial vectors)
  ├─ ParameterField  → param p                ├─ state_index_map
  ├─ ExcitationPort  → param p (input)        └─ systems [CompiledSystem]
  └─ ContinuousField → state x (output/alg)

System  (stateless dynamics)        Backends:  ScipyBackend  (dev)
  └─ python_fn / dynamics_fn                   JAXBackend    (~1500× warm)
                                               JuliaBackend  (~600× warm)
World  (components + systems)
```

---

## Field types

| Field | Vector | Updated | Use for |
|---|---|---|---|
| `IntegratedField()` | state `x` | Every ODE step | Position, velocity, pressure, temperature |
| `ParameterField()` | param `p` | Never (constant) | Material properties, geometry, gains |
| `ContinuousField()` | state `x` | Every RHS call | Output / algebraic slots |
| `DiscreteField(dt)` | state `x` | Forces tstops | Sampled sensors, event-driven state |
| `ExcitationPort(...)` | param `p` | Via inject_excitation | Characterization input ports |

Vector fields: add `size=N` to any field to store an array of N values.

---

## Minimal example

```python
# components.py
from numen.spec.component import Component
from numen.fields import IntegratedField, ParameterField
from typing import Annotated, Literal

class MassComponent(Component):
    kind:     Literal["mass"] = "mass"
    position: Annotated[float, IntegratedField()] = 0.0
    velocity: Annotated[float, IntegratedField()] = 0.0
    mass:     Annotated[float, ParameterField()]  = 1.0

# dynamics.py
import jax.numpy as jnp
from numen.spec.system import System, DynamicsFn
from typing import ClassVar

def gravity(dx, x, p, t, spec, system):
    for (eid,) in system.entity_groups:
        c  = spec.view(eid, MassComponent, x, p)
        dc = spec.dx_view(eid, MassComponent, dx)
        dc.position += c.velocity
        dc.velocity += -9.81

class GravitySystem(System):
    component_types: ClassVar = (MassComponent,)
    python_fn:       ClassVar = staticmethod(gravity)
    kind:     Literal["gravity"] = "gravity"
    dynamics_fn: str = "MyDyn.gravity!"

# world.py + solve
from numen.spec.world import GenericWorld
from numen.compiler.flatten import compile_spec
from numen.bridge.scipy_backend import ScipyBackend

World  = GenericWorld[MassComponent, GravitySystem, None]
world  = World(components={"ball": MassComponent(position=10.0)},
               systems={"g": GravitySystem()})
spec   = compile_spec(world)
result = ScipyBackend().solve(spec, tspan=(0.0, 2.0))
```

---

## JAX rules  ⚠️ critical

Inside dynamics functions, always use `jnp.*` (never `np.*`).
Never use `if`/`else` on state values; use `jnp.where(cond, a, b)`.
Guard both branches of every `jnp.where` against NaN/Inf.
Use solver `Dopri5` (not Tsit5) when absolute tolerance is tight.

## Smooth contact  ⚠️ critical

Hard-stop forces with a C0 kink cause catastrophic step rejection (>99%).
Always use a C1-smooth 1 µm ramp:

```python
_D = 1e-6
def _soft_pen(x):
    return jnp.where(x <= 0, 0.0, jnp.where(x >= _D, x - 0.5*_D, 0.5*x*x/_D))
```

---

## Backends

| Backend | Warm speed | Use when |
|---|---|---|
| `ScipyBackend(rtol, atol)` | baseline | Development, debugging |
| `JAXBackend(solver="Dopri5", max_steps=100_000)` | ~1500× faster | Repeated solves, Monte Carlo, differentiable |
| `JuliaBackend(julia_file=..., method=..., rtol, atol)` | ~600× faster | Long runs, stiff systems |
| `JuliaServerBackend(julia_file=..., method=..., rtol, atol)` | ~600× faster | Parameter sweeps (pays JIT once) |

Stiff problems: use `method="Rodas5P"` (Julia) or `solver="Kvaerno5"` (JAX).

---

## Accessing results

```python
from numen.reconstruction.collector import SnapshotCollector

collector = SnapshotCollector(world, spec, result)
t, pos = collector.field_series("entity_id", "position")   # time series
snap   = collector.at(t=1.5)                                # typed snapshot
```

---

## Characterization

See **[CHARACTERIZATION.md](CHARACTERIZATION.md)** for the complete guide.
Quick-start once your component has an `ExcitationPort`:

```bash
uv run numen characterize test_plan.yaml        # run tests + generate plot
uv run numen characterize test_plan.yaml -c     # characterize only
uv run numen characterize test_plan.yaml -p     # re-plot without re-running
```

**Test types:** `discrete_frequency_sweep`, `continuous_chirp`, `amplitude_sweep`,
`dc_operating_point_sweep`, `parameter_sweep`, `parameter_grid`, `doe_sweep`.

**`parameter_sweep` sweep params** — model param OR excitation input:

```yaml
- name: frf_vs_dc
  type: parameter_sweep
  sweep_param: excitation.dc_offset   # or: excitation.amplitude, excitation.frequency
  values: [0.0, 0.3, 0.6, 1.0]       # or: osc.c1 for a model ParameterField
  sub_test: baseline_frf              # any other test name in the same plan
```

**Plot panel types:** `bode`, `chirp_timeseries`, `amplitude_sweep`, `dc_sweep`,
`parameter_family` (family of curves coloured by sweep param), `doe_scatter`,
`parameter_grid_heatmap`.

The `plots:` section lives in the same YAML file as the tests — one file controls
both the campaign and the figure layout.

---

## Multi-entity topology

```python
from numen.fields import EntityGroup

class SpringSystem(System):
    entity_slots: ClassVar[EntityGroup] = EntityGroup(
        MassComponent, SpringComponent, MassComponent
    )
    ...

SpringSystem(entity_groups=[["m1", "spring", "m2"]])
```

---

## Controller callbacks

```python
from numen.spec.callback import Callback

class MyCallback(Callback):
    kind:      Literal["my"] = "my"
    dt:        float = 0.01          # fires every dt seconds
    julia_fn:  str   = "Dyn.ctrl!"
    params:    dict  = {"kp": 1.0}
    python_fn: ClassVar = staticmethod(lambda t, x, p, spec: {})

# Scipy: segment-solve (zero jitter).  JAX: unsupported.  Julia: PeriodicCallback.
```

---

## DAE — algebraic constraints

`ContinuousField(algebraic=True)` marks a slot with no time derivative.
Julia-only; requires `method="Rodas5P"` or `"FBDF"`.

---

## Logging

```python
from numen.logging import configure_logging
import logging
configure_logging(level=logging.DEBUG)   # per-solve timing + Julia output
```

---

## Project models

<!-- Update this table when you add or modify a model. -->
<!-- This is the most important section for a new AI session to read. -->

| Entity | Component | Key parameters | Notes |
|---|---|---|---|
| *(add your models here)* | | | |

---

## Analysis notes

<!-- Record findings from characterization campaigns here. -->
<!-- Example: "oscillator Q≈63 at ω=2π; softening onset at amp≈0.1 N" -->

---

## Known issues and workarounds

<!-- Track bugs, in-progress work, or non-obvious gotchas specific to this project. -->
