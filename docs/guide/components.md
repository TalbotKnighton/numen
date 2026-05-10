# Components & Fields

## Overview

Components are frozen Pydantic models that declare the **data** of your simulation.
They carry no topology, no solver logic — just fields annotated with their role in the solver.

```python
from typing import Annotated, Literal
from numen.spec.component import Component
from numen.fields import IntegratedField, ParameterField

class TankComponent(Component):
    kind:        Literal["tank"] = "tank"
    pressure:    Annotated[float, IntegratedField()] = 101_325.0   # Pa
    volume:      Annotated[float, ParameterField()]  = 1e-3        # m³
    temperature: Annotated[float, ParameterField()]  = 293.15      # K
```

Rules:

- `kind` must be a unique `Literal` string — it's Pydantic's discriminator.
- Default values are the initial conditions / parameter values when not overridden.
- Components are **frozen** (immutable after construction).

## Field types

| Field | Vector | When updated | All backends? |
|---|---|---|---|
| `IntegratedField()` | state `x` | Every ODE step (solver integrates) | Yes |
| `ParameterField()` | param `p` | Never — constant for the entire solve | Yes |
| `ContinuousField()` | state `x` | Every RHS call (dynamics fn writes) | Yes |
| `DiscreteField(dt)` | state `x` | Forces solver tstops at multiples of `dt` | Yes |
| `ExcitationPort()` | *(none)* | Pure annotation metadata — NOT compiled | Characterization only |

### IntegratedField

The most common field type. The ODE solver integrates `dx/dt = f(x, p, t)` for these slots.

```python
position: Annotated[float, IntegratedField()] = 0.0
velocity: Annotated[float, IntegratedField()] = 0.0
```

### ParameterField

Constant during a solve. Useful for physical parameters like mass, stiffness, area.

```python
mass:        Annotated[float, ParameterField()] = 1.0     # kg
spring_k:    Annotated[float, ParameterField()] = 5000.0  # N/m
```

### ContinuousField

Written by the dynamics function on every RHS call. Use for computed outputs
(forces, flow rates, power) that you want to read out of the result.

```python
class ForceComponent(Component):
    kind:        Literal["force"] = "force"
    force_value: Annotated[float, ContinuousField()] = 0.0  # N — written each step
```

With `algebraic=True`: the dynamics function writes a **residual** `g(x) = 0`
instead of a derivative. This is a DAE algebraic constraint, supported only by
Julia implicit solvers (Rodas5P, FBDF).

```python
pressure_balance: Annotated[float, ContinuousField(algebraic=True)] = 0.0
```

### DiscreteField

Zero-order-hold variable updated at a fixed rate. Numen injects required solver
stop times at multiples of `dt` so a controller callback can fire at exact times.

```python
class ControllerState(Component):
    kind:     Literal["ctrl"] = "ctrl"
    setpoint: Annotated[float, DiscreteField(dt=0.01)] = 0.0  # updated at 100 Hz
```

### ExcitationPort

Marks a field as an injectable excitation input for the characterization framework.
`ExcitationPort` is **not** compiled into the parameter vector — it's pure metadata
used by `inject_excitation()` to add a time-varying forcing system.

```python
class MassComponent(Component):
    kind:     Literal["mass"] = "mass"
    position: Annotated[float, IntegratedField()] = 0.0
    velocity: Annotated[float, IntegratedField()] = 0.0
    mass:     Annotated[float, ParameterField()]  = 1.0
    force:    Annotated[float, ExcitationPort(
                  targets   = "velocity",   # IntegratedField whose derivative gets F(t)
                  port_type = "effort",     # "effort" or "flow"
                  units     = "N",
              )] = 0.0
```

## Vector fields (`size=N`)

Any field type can hold a contiguous array by setting `size=N`. The compiler packs
it as `N` consecutive slots in `x` (or `p`).

```python
class SignalComponent(Component):
    kind:        Literal["signal"] = "signal"
    frequencies: Annotated[list[float], ParameterField(size=8)] = [0.0] * 8
    amplitudes:  Annotated[list[float], ParameterField(size=8)] = [0.0] * 8
```

Accessing a vector field via `spec.view()` returns a numpy/JAX slice:

```python
sig = spec.view(eid, SignalComponent, x, p)
f0  = sig.frequencies[0]    # scalar
all_freqs = sig.frequencies  # array slice
```

## Multiple components per entity

An entity can have more than one component (keyed by their `kind` string):

```python
world = World(
    components={
        "robot": {
            "body":   BodyComponent(mass=10.0),
            "sensor": SensorComponent(noise=0.01),
        },
    },
    ...
)
```

State/param keys use the full path `entity_id.component_kind.field_name`, so
`"robot.body.mass"` and `"robot.sensor.noise"` are independent parameter slots.
