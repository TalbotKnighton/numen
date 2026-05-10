# Numen

A Python-first framework for engineering dynamics simulation.
Define your physics model in Python, then solve it fast — using JAX (1500× speedup) or Julia (600× speedup) as drop-in backends.

```
     Python                          Backends
┌──────────────────────┐      ┌─────────────────────────┐
│  Component (data)    │      │  ScipyBackend   — RK45  │
│  System    (physics) │─────▶│  JAXBackend  — Dopri5   │
│  World     (model)   │      │  JuliaBackend — Tsit5   │
└──────────────────────┘      └─────────────────────────┘
```

## Why Numen?

Most simulation tools make you choose between **ease of modeling** and **speed of solving**.
Numen separates the two: write your model once in readable Python, then pick the solver that fits your workflow.

| Backend | Warm solve | Use for |
|---|---|---|
| `ScipyBackend` | ~9 s | Development, debugging, first runs |
| `JAXBackend` | **~6 ms** (1500×) | Monte Carlo, parameter sweeps, repeated solves |
| `JuliaBackend` | **~14 ms** (600×) | Long simulations, production batch runs |

*Timings from the fluid poppet example — 150 ms pneumatic transient, 6-state system.*

## Installation

```bash
pip install numen
```

Optional extras:

```bash
pip install "numen[jax]"              # JAX backend (~1500× faster warm solves)
pip install "numen[characterization]" # DOE sweeps, parameter grids
pip install "numen[dev]"              # pytest + coverage
```

For the Julia backend, install [Julia ≥ 1.10](https://julialang.org/downloads/) and add it to your PATH.

**Requirements:** Python ≥ 3.12

## Quick Start

### 1. Define a component

```python
from typing import Annotated, Literal
from numen.spec.component import Component
from numen.fields import IntegratedField, ParameterField

class BallComponent(Component):
    kind:     Literal["ball"] = "ball"
    position: Annotated[float, IntegratedField()] = 0.0   # integrated state
    velocity: Annotated[float, IntegratedField()] = 0.0   # integrated state
    mass:     Annotated[float, ParameterField()]  = 1.0   # constant parameter
```

### 2. Define the dynamics

```python
import jax.numpy as jnp
from typing import ClassVar, Literal
from numen.spec.system import System, DynamicsFn

def gravity_dynamics(dx, x, p, t, spec, system):
    for (eid,) in system.entity_groups:
        ball = spec.view(eid, BallComponent, x, p)   # read
        db   = spec.dx_view(eid, BallComponent, dx)  # write
        db.position += ball.velocity
        db.velocity += -9.81

class GravitySystem(System):
    component_types: ClassVar = (BallComponent,)
    python_fn:       ClassVar[DynamicsFn] = staticmethod(gravity_dynamics)
    kind:            Literal["gravity"] = "gravity"
    dynamics_fn:     str = "MyDynamics.gravity_dynamics!"
```

### 3. Assemble and solve

```python
from numen.spec.world import GenericWorld
from numen.compiler.flatten import compile_spec
from numen.bridge.scipy_backend import ScipyBackend

World  = GenericWorld[BallComponent, GravitySystem, None]
world  = World(
    components={"ball": {"ball": BallComponent(position=100.0)}},
    systems={"gravity": GravitySystem()},
)
spec   = compile_spec(world)
result = ScipyBackend().solve(spec, tspan=(0.0, 5.0))
```

### 4. Access results

```python
from numen.reconstruction.collector import SnapshotCollector

collector = SnapshotCollector(world, spec, result)
t, position = collector.field_series("ball", "ball", "position")

snap = collector.at(t=2.5)
print(snap.components["ball"]["ball"].position)
```

## Built-in Examples

```bash
numen check          # verify all backends work
numen list           # show built-in examples
numen run oscillator # run the oscillator example
```

| Example | Demonstrates |
|---|---|
| `oscillator` | Minimal end-to-end model |
| `coupled_spring` | Multi-entity topology, spring chain |
| `fluid_poppet` | Pneumatic network + poppet valve, all backends |
| `nonlinear_oscillator` | ExcitationPort, characterization campaign |
| `pneumatic_dashpot` | Stiff ODE, Rodas5P, parameter sweeps |
