# Systems & Dynamics

## Overview

A system declares **which entities it operates on** and provides the dynamics
functions (one for Python backends, one reference for Julia).

Topology lives in the system — not in the components. A spring doesn't know
which masses it connects; the `SpringForceSystem` declares the coupling via
`entity_groups`.

## Single-entity system

The simplest case: one system operates on all entities with a given component type.

```python
from typing import ClassVar, Literal
from numen.spec.system import System, DynamicsFn
from components import MassComponent

def mass_kinematics(dx, x, p, t, spec, system):
    for (eid,) in system.entity_groups:
        m  = spec.view(eid, MassComponent, x, p)   # read
        dm = spec.dx_view(eid, MassComponent, dx)  # write derivatives
        dm.position += m.velocity                  # ẋ = v

class MassKinematicsSystem(System):
    component_types: ClassVar[tuple[type, ...]] = (MassComponent,)
    python_fn:       ClassVar[DynamicsFn]       = staticmethod(mass_kinematics)
    kind:            Literal["mass_kinematics"] = "mass_kinematics"
    dynamics_fn:     str = "Dynamics.mass_kinematics!"
```

`component_types` tells `compile_spec` to auto-populate `entity_ids` with all
entities that have a `MassComponent`. The compiled system's `entity_groups` is
then `(("m1",), ("m2",), ("m3",))` for a 3-mass model.

## Multi-entity (coupled) system

When a system couples multiple entities — a spring between two masses, an
orifice between two control volumes — declare the slots in `entity_slots` and
provide `entity_groups` at instantiation.

```python
from numen.fields import EntityGroup

class SpringForceSystem(System):
    component_types: ClassVar[tuple[type, ...]] = ()   # no auto-population
    entity_slots:    ClassVar[EntityGroup]      = EntityGroup(
        MassComponent, SpringComponent, MassComponent   # group_size = 3
    )
    python_fn:  ClassVar[DynamicsFn] = staticmethod(spring_force_dynamics)
    kind:       Literal["spring_force"] = "spring_force"
    dynamics_fn: str = "Dynamics.spring_force!"

# Instantiate with explicit topology
SpringForceSystem(entity_groups=[
    ["m1", "s1", "m2"],
    ["m2", "s2", "m3"],   # m2 appears in both groups — += accumulates forces correctly
])
```

`compile_spec` validates each group's component types against `entity_slots.slot_types`.

## Dynamics function signature

All Python dynamics functions use the same calling convention:

```python
def my_dynamics(
    dx:     np.ndarray,          # derivative accumulator (write via dx_view)
    x:      np.ndarray,          # current state vector (read via view)
    p:      np.ndarray,          # parameter vector (read via view)
    t:      float,               # current time
    spec:   CompiledSpec,        # compiled spec (index maps, view helpers)
    system: CompiledSystem,      # compiled system (entity_groups, python_fn)
) -> None:
    ...
```

Key rules:

- `spec.view(eid, ComponentType, x, p)` returns a **read-only** `ComponentView`.
- `spec.dx_view(eid, ComponentType, dx)` returns a **write** `DerivativeView`.
- Use `+=` to accumulate derivatives — multiple systems writing the same slot add correctly.
- Never call `spec.view` with the wrong `ComponentType` — it will silently read garbage.

## JAX compatibility

The JAX backend JIT-traces dynamics functions. Python `if`/`else` on state
values will raise `TracerBoolConversionError` at trace time.

| Instead of | Use |
|---|---|
| `if P_a > P_b:` | `jnp.where(P_a > P_b, ..., ...)` |
| `np.sqrt(x)` | `jnp.sqrt(x)` |
| `np.maximum(0, x)` | `jnp.maximum(0.0, x)` |
| `np.clip(x, 0, 1)` | `jnp.clip(x, 0.0, 1.0)` |

Both branches of every `jnp.where` are **always evaluated** — guard the non-selected branch:

```python
# BAD — sqrt blows up when arg is negative in the false branch
result = jnp.where(choked, mdot_choked, jnp.sqrt(P_up - P_dn))

# GOOD — guard the argument
arg    = jnp.maximum(0.0, P_up - P_dn)
result = jnp.where(choked, mdot_choked, jnp.sqrt(arg))
```

Import: always `import jax.numpy as jnp` at the top of `dynamics.py`.
Both `jnp.*` and plain Python arithmetic (`+`, `*`, `**`) work for both numpy and JAX arrays.

## Smooth contact / hard-stop forces

A sharp `max(0, -pos)` stop force has a C0 discontinuity that causes thousands
of rejected solver steps. Use a C1-smooth ramp over 1 µm:

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
alpha  = jnp.clip(-pos / _STOP_DELTA, 0.0, 1.0)
v_damp = jnp.maximum(0.0, -vel) * alpha

F_stop = k_stop * _soft_pen(-pos) + c_stop * v_damp
```

## DynamicsFn protocol

```python
class DynamicsFn(Protocol[GroupT]):
    def __call__(
        self,
        dx: np.ndarray,
        x: np.ndarray,
        p: np.ndarray,
        t: float,
        spec: CompiledSpec,
        system: CompiledSystem[GroupT],
    ) -> None: ...
```

Declare `python_fn` as a `ClassVar` with `staticmethod` to avoid Pydantic
trying to serialize the callable:

```python
class MySystem(System):
    python_fn: ClassVar[DynamicsFn] = staticmethod(my_dynamics_fn)
```
