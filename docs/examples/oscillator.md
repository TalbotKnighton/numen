# Oscillator Example

The oscillator is the minimal end-to-end Numen example — a damped 1D harmonic oscillator.
It demonstrates the complete model-to-result workflow using only the scipy backend.

Source: `src/numen/examples/oscillator/`

## Physics

The equation of motion is:

```
ẍ + 2ζω·ẋ + ω²x = 0
```

where `ω` is the natural frequency [rad/s] and `ζ` is the damping ratio.

## Component

```python
# components.py
from typing import Annotated, Literal
from numen.spec.component import Component
from numen.fields import IntegratedField, ParameterField

class OscillatorComponent(Component):
    """Simple 1D harmonic oscillator: ẍ + 2ζω·ẋ + ω²x = 0."""

    kind:     Literal["oscillator"] = "oscillator"
    position: Annotated[float, IntegratedField()] = 1.0   # initial x = 1 m
    velocity: Annotated[float, IntegratedField()] = 0.0   # initial v = 0 m/s
    omega:    Annotated[float, ParameterField()]  = 1.0   # natural frequency [rad/s]
    damping:  Annotated[float, ParameterField()]  = 0.0   # damping ratio ζ
```

Two `IntegratedField` slots → state vector `x` has two elements per entity.
Two `ParameterField` slots → parameter vector `p` has two elements.

## Dynamics

```python
# dynamics.py
def oscillator_dynamics(dx, x, p, t, spec, system):
    """ẋ = v, v̇ = -ω²x - 2ζωv"""
    for (entity_id,) in system.entity_groups:
        c  = spec.view(entity_id, OscillatorComponent, x, p)   # read
        dc = spec.dx_view(entity_id, OscillatorComponent, dx)  # write
        dc.position += c.velocity
        dc.velocity += -c.omega**2 * c.position - 2 * c.damping * c.omega * c.velocity

class OscillatorSystem(System):
    component_types: ClassVar = (OscillatorComponent,)
    python_fn:       ClassVar[DynamicsFn] = staticmethod(oscillator_dynamics)
    kind:            Literal["oscillator_system"] = "oscillator_system"
    dynamics_fn:     str = "OscillatorDynamics.oscillator_dynamics!"
```

`component_types = (OscillatorComponent,)` tells `compile_spec` to auto-populate
`entity_ids` with all entities that have an `OscillatorComponent`. The `entity_groups`
tuple on the compiled system will be `(("osc",),)` for a single-entity world.

## World

```python
# world.py
World = GenericWorld[OscillatorComponent, OscillatorSystem, None]

def make_world(x0=1.0, v0=0.0, omega=1.0, damping=0.0, entity_id="osc"):
    return World(
        components={entity_id: {"oscillator": OscillatorComponent(
            position=x0, velocity=v0, omega=omega, damping=damping
        )}},
        systems={"osc_sys": OscillatorSystem()},
    )
```

## Run

```python
# run.py
import math
from numen.compiler.flatten import compile_spec
from numen.bridge.scipy_backend import ScipyBackend
from numen.reconstruction.collector import SnapshotCollector
from world import make_world

world  = make_world(x0=1.0, omega=2 * math.pi, damping=0.05)
spec   = compile_spec(world)

# State/param index maps after compilation:
# state_index_map: {"osc.oscillator.position": (0, 1), "osc.oscillator.velocity": (1, 2)}
# param_index_map: {"osc.oscillator.omega": (0, 1), "osc.oscillator.damping": (1, 2)}

result = ScipyBackend(rtol=1e-9, atol=1e-9).solve(spec, tspan=(0.0, 5.0))

collector = SnapshotCollector(world, spec, result)
t, position = collector.field_series("osc", "oscillator", "position")
```

## Run it

```bash
uv run numen run oscillator
```

## Key observations

- The model has exactly **4 scalars**: 2 state (`position`, `velocity`) + 2 param (`omega`, `damping`).
- `compile_spec` is a pure function — call it once, reuse the spec for all backends.
- `SnapshotCollector.field_series` is faster than `collector.at(t)` for plotting —
  it directly slices `result.x` without rebuilding world objects.
