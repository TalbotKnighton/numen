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
Numen separates the two: write your model once in readable Python, then pick the solver
that fits your workflow — from quick scipy runs during development to compiled JAX or Julia
kernels for production.

| Backend | Warm solve¹ | Use for |
|---|---|---|
| `ScipyBackend` | 9048 ms | Development, debugging, first runs |
| `JAXBackend` | **6 ms** (1507×) | Monte Carlo, parameter sweeps, repeated solves |
| `JuliaBackend` | **14 ms** (634×) | Long simulations, production batch runs |

¹ Fluid poppet check valve example — 150 ms pneumatic transient, 6-state system.

---

## Installation

```bash
pip install numen
```

**Optional extras:**

```bash
pip install "numen[jax]"   # JAX backend (diffrax, ~1500× faster warm solves)
```

For the Julia backend, install [Julia ≥ 1.10](https://julialang.org/downloads/) and add it to your PATH.
The first solve will automatically install the required Julia packages.

**Requirements:** Python ≥ 3.12

---

## Quick start

### Verify your installation

```bash
numen check
```

```
Numen backend check
==================================================
  scipy   ✓  (RK45, oscillator x(1s) = 1.000000)
  JAX     ✓  (Dopri5, oscillator x(1s) = 1.000000)
  Julia   ✓  julia version 1.12.0
```

### Start a new project

```bash
numen init my_project --model first_model --domain mechanical
cd my_project
```

This creates:
```
my_project/
├── CLAUDE.md           (AI assistant context — explains the framework)
└── first_model/
    ├── components.py   (define state and parameter fields)
    ├── dynamics.py     (write physics — JAX-compatible)
    ├── dynamics.jl     (Julia translation for fast backend)
    ├── world.py        (set initial conditions and topology)
    └── run.py          (solve and plot)
```

Run it immediately:

```bash
cd first_model
python run.py
```

### Scaffold additional models

```bash
numen new heat_pipe --domain fluid
numen new deployment_arm --domain mechanical
numen new sensor_loop --domain generic
```

---

## How it works

A Numen model has three parts:

### 1. Components — your data

```python
from numen.spec.component import Component
from numen.fields import IntegratedField, ParameterField
from typing import Annotated, Literal

class BallComponent(Component):
    kind:     Literal["ball"] = "ball"
    position: Annotated[float, IntegratedField()] = 0.0   # state: solved by ODE
    velocity: Annotated[float, IntegratedField()] = 0.0   # state: solved by ODE
    mass:     Annotated[float, ParameterField()]  = 1.0   # param: constant
```

### 2. Systems — your physics

```python
import jax.numpy as jnp
from numen.spec.system import System, DynamicsFn
from typing import ClassVar

def gravity_dynamics(dx, x, p, t, spec, system):
    for (eid,) in system.entity_groups:
        ball = spec.view(eid, BallComponent, x, p)   # read state + params
        db   = spec.dx_view(eid, BallComponent, dx)  # write derivatives
        db.position += ball.velocity
        db.velocity += -9.81

class GravitySystem(System):
    component_types: ClassVar[tuple[type, ...]] = (BallComponent,)
    python_fn:       ClassVar[DynamicsFn]       = staticmethod(gravity_dynamics)
    kind:            Literal["gravity"]         = "gravity"
    dynamics_fn:     str = "MyDynamics.gravity_dynamics!"
```

### 3. Solve

```python
from numen.spec.world import GenericWorld
from numen.compiler.flatten import compile_spec
from numen.bridge.scipy_backend import ScipyBackend

World  = GenericWorld[BallComponent, GravitySystem, None]
world  = World(
    components={"ball": BallComponent(position=100.0, mass=2.0)},
    systems={"gravity": GravitySystem()},
)
spec   = compile_spec(world)
result = ScipyBackend().solve(spec, tspan=(0.0, 5.0))
```

Switch to the JAX backend for repeated solves with no code changes:

```python
from numen.bridge.jax_backend import JAXBackend
result = JAXBackend(solver="Dopri5").solve(spec, tspan=(0.0, 5.0))
```

---

## Accessing results

```python
from numen.reconstruction.collector import SnapshotCollector

collector = SnapshotCollector(world, spec, result)

# Time series
t, position = collector.field_series("ball", "position")

# Snapshot at a specific time
snap = collector.at(t=2.5)
print(snap.components["ball"].position)
```

---

## Built-in examples

| Example | Domain | Demonstrates |
|---|---|---|
| `oscillator` | Mechanical | Minimal end-to-end model, damped harmonic oscillator |
| `coupled_spring` | Mechanical | Multi-entity topology, spring chain, energy conservation |
| `fluid_poppet` | Fluid + Mechanical | Isentropic orifice flow, poppet valve, all three backends |

```bash
numen list              # show all examples
numen run oscillator    # run one (no plot window)
```

---

## JAX compatibility

For the JAX backend to work, dynamics functions must be traceable by JAX:

```python
# ✗ Python if/else on state values
if P_a > P_b:
    mdot = flow(P_a, P_b)

# ✓ Use jnp.where
mdot = jnp.where(P_a > P_b, flow(P_a, P_b), -flow(P_b, P_a))

# ✗ numpy operations
f = np.sqrt(np.maximum(0, x))

# ✓ jax.numpy operations
f = jnp.sqrt(jnp.maximum(0.0, x))
```

The scaffold templates from `numen new` are already JAX-compatible.

---

## Julia backend

For each Python `System`, write a matching Julia function in a `.jl` file:

```julia
# dynamics.jl
module MyDynamics
import Main: CompiledSpec, CompiledSystemSpec, state_idx, param_idx

function gravity_dynamics!(dx, x, p, t, spec, sys)
    for id_ball in sys.entity_ids
        dx[state_idx(spec, id_ball * ".position")] += x[state_idx(spec, id_ball * ".velocity")]
        dx[state_idx(spec, id_ball * ".velocity")] += -9.81
    end
end

end  # module MyDynamics
```

```python
from numen.bridge.runtime import JuliaBackend
result = JuliaBackend(julia_file="dynamics.jl").solve(spec, tspan=(0.0, 5.0))
```

The scaffolded `dynamics.jl` from `numen new` gives you a working starting point.

---

## CLI reference

```
numen init [dir] [--model NAME] [--domain DOMAIN]
    Bootstrap a new project. Creates CLAUDE.md and optionally a first model.
    Domains: mechanical, fluid, generic

numen check
    Smoke-test scipy, JAX, and Julia backends.

numen new NAME [--domain DOMAIN]
    Scaffold a new model directory inside an existing project.

numen list
    List built-in example models.

numen run EXAMPLE
    Run a built-in example (oscillator, coupled_spring, fluid_poppet).

numen info
    Print a quick-reference cheat-sheet.
```

---

## Design

See [DESIGN.md](DESIGN.md) for architectural decisions, the ODE vs. DAE boundary,
Multibody.jl plans for 3D mechanisms, and open questions.
