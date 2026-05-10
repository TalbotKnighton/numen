# Getting Started

## Installation

Install Numen with pip or uv:

```bash
pip install numen
# or
uv add numen
```

Optional extras:

```bash
pip install "numen[jax]"              # JAX backend
pip install "numen[characterization]" # DOE sweeps, parameter grids
```

For the Julia backend, install [Julia ≥ 1.10](https://julialang.org/downloads/) and add it to your PATH.
The first solve will automatically install the required Julia packages.

## Verify your installation

```bash
numen check
```

Expected output:

```
Numen backend check
==================================================
  scipy   ✓  (RK45, oscillator x(1s) = 1.000000)
  JAX     ✓  (Dopri5, oscillator x(1s) = 1.000000)
  Julia   ✓  julia version 1.12.0
```

## Start a new project

```bash
numen init my_project --model first_model --domain mechanical
cd my_project
```

This creates:

```
my_project/
├── CLAUDE.md           (AI assistant context)
└── first_model/
    ├── components.py   (define state and parameter fields)
    ├── dynamics.py     (write physics — JAX-compatible)
    ├── dynamics.jl     (Julia translation for fast backend)
    ├── world.py        (set initial conditions and topology)
    └── run.py          (solve and plot)
```

Run it immediately:

```bash
cd my_project/first_model
python run.py
```

## Your first model: 1D oscillator

This walkthrough builds the oscillator example from scratch.

### Step 1 — Define a component

Components are frozen Pydantic models that declare fields using `Annotated` type hints.

```python
# components.py
from typing import Annotated, Literal
from numen.spec.component import Component
from numen.fields import IntegratedField, ParameterField

class OscillatorComponent(Component):
    kind:     Literal["oscillator"] = "oscillator"
    position: Annotated[float, IntegratedField()] = 1.0   # initial position = 1 m
    velocity: Annotated[float, IntegratedField()] = 0.0   # initial velocity = 0 m/s
    omega:    Annotated[float, ParameterField()]  = 1.0   # natural frequency [rad/s]
    damping:  Annotated[float, ParameterField()]  = 0.05  # damping ratio ζ
```

- `IntegratedField` — solver integrates this; it's part of the state vector `x`.
- `ParameterField` — constant during the solve; part of the parameter vector `p`.

### Step 2 — Define the dynamics

```python
# dynamics.py
import math
from typing import ClassVar, Literal
from numen.spec.system import System, DynamicsFn
from components import OscillatorComponent

def oscillator_dynamics(dx, x, p, t, spec, system):
    """ẋ = v, v̇ = -ω²x - 2ζωv"""
    for (eid,) in system.entity_groups:
        c  = spec.view(eid, OscillatorComponent, x, p)   # read state + params
        dc = spec.dx_view(eid, OscillatorComponent, dx)  # write derivatives
        dc.position += c.velocity
        dc.velocity += -c.omega**2 * c.position - 2 * c.damping * c.omega * c.velocity

class OscillatorSystem(System):
    component_types: ClassVar = (OscillatorComponent,)
    python_fn:       ClassVar[DynamicsFn] = staticmethod(oscillator_dynamics)
    kind:            Literal["oscillator_system"] = "oscillator_system"
    dynamics_fn:     str = "OscDyn.oscillator_dynamics!"
```

### Step 3 — Assemble the world

```python
# world.py
from numen.spec.world import GenericWorld
from components import OscillatorComponent
from dynamics import OscillatorSystem

World = GenericWorld[OscillatorComponent, OscillatorSystem, None]

def make_world(omega=2*math.pi, damping=0.05):
    return World(
        components={"osc": {"oscillator": OscillatorComponent(omega=omega, damping=damping)}},
        systems={"osc_sys": OscillatorSystem()},
    )
```

### Step 4 — Compile and solve

```python
# run.py
from numen.compiler.flatten import compile_spec
from numen.bridge.scipy_backend import ScipyBackend
from numen.reconstruction.collector import SnapshotCollector
from world import make_world

world  = make_world()
spec   = compile_spec(world)
result = ScipyBackend(rtol=1e-9, atol=1e-9).solve(spec, tspan=(0.0, 5.0))

collector = SnapshotCollector(world, spec, result)
t, position = collector.field_series("osc", "oscillator", "position")
print(f"Final position: {position[-1]:.4f} m")
```

### Step 5 — Switch backends

Switch to JAX for repeated solves — no code changes required:

```python
from numen.bridge.jax_backend import JAXBackend
result = JAXBackend(solver="Dopri5").solve(spec, tspan=(0.0, 5.0))
```

## Running built-in examples

```bash
numen list                 # list all examples
numen run oscillator       # run the oscillator
numen run coupled_spring   # run the spring chain
numen run fluid_poppet     # run the poppet valve (scipy only)
```

## CLI reference

```
numen init [dir] [--model NAME] [--domain DOMAIN]
    Bootstrap a new project.

numen check
    Smoke-test scipy, JAX, and Julia backends.

numen new NAME [--domain DOMAIN]
    Scaffold a new model directory inside an existing project.

numen list
    List built-in example models.

numen run EXAMPLE
    Run a built-in example.

numen characterize PLAN [--output FILE] [--verbose]
    Run a YAML test campaign against a model.
```
