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
pip install "numen[jax]"              # JAX backend (diffrax, ~1500× faster warm solves)
pip install "numen[characterization]" # pandas, pyDOE3, SALib — required for DOE sweeps
pip install "numen[dev]"              # pytest + coverage
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
| `nonlinear_oscillator` | Mechanical | ExcitationPort, characterization campaign, FRF + amplitude sweep |

```bash
numen list              # show all examples
numen run oscillator    # run one (no plot window)
```

---

## Characterization framework

Numen includes a domain-agnostic test campaign engine for characterizing model behavior.
Write a YAML test plan and run it against any model with an `ExcitationPort`:

```bash
numen characterize test_plan.yaml --output results.json
```

### Test types

| Type | Description |
|---|---|
| `discrete_frequency_sweep` | Stepped sine — most accurate FRF, lock-in detection |
| `continuous_chirp` | Single-solve frequency sweep — fast survey |
| `amplitude_sweep` | Fixed frequency, varying amplitude — reveals nonlinearity |
| `dc_operating_point_sweep` | Small-signal FRF at each DC bias level |
| `parameter_sweep` | Repeat a sub-test for each value of one model parameter |
| `parameter_grid` | Full factorial or pairwise grid over multiple parameters |
| `doe_sweep` | Space-filling DOE (LHS, Sobol, Halton) or classical designs (CCD, BBD) |

### Quick start

```python
# 1. Add an ExcitationPort to your component
from numen.fields import ExcitationPort

class OscComponent(Component):
    ...
    force: Annotated[float, ExcitationPort(
        targets="velocity",   # IntegratedField whose derivative gets F(t)
        port_type="effort",
        units="N",
    )] = 0.0
```

```yaml
# 2. Write a test_plan.yaml
version: "1.0"
backend: { type: scipy }
model:   { module: world, factory: make_world }
excitation: { entity: osc, port: force, output_state: position }
tests:
  - { name: frf, type: discrete_frequency_sweep,
      frequencies: { spacing: log, f_start: 0.1, f_end: 10.0, n_points: 30 },
      amplitude: 0.01, settle_periods: 50, measure_periods: 10 }
```

```bash
# 3. Run
numen characterize test_plan.yaml --output results.json
```

DOE sweeps (`latin_hypercube`, `sobol`, `halton`, `central_composite`, `box_behnken`) require:

```bash
pip install "numen[characterization]"
```

See `examples/nonlinear_oscillator/` for a complete worked example, and the
`CHARACTERIZATION.md` file generated by `numen init` for the full guide.

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

function gravity_dynamics!(
    dx  :: AbstractVector{T},
    x   :: AbstractVector{S},
    p   :: Vector{Float64},
    t   :: Real,
    spec:: CompiledSpec,
    sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    for id_ball in sys.entity_ids
        dx[state_idx(spec, id_ball * ".ball.velocity")] += -9.81
        dx[state_idx(spec, id_ball * ".ball.position")] += x[state_idx(spec, id_ball * ".ball.velocity")]
    end
end

end  # module MyDynamics
```

The `{T, S}` signature is required for stiff solvers (Rodas5P): during Jacobian evaluation OrdinaryDiffEq calls the function with `x::Vector{Dual}`, and both type parameters must be present so helper functions can correctly type their return values.

```python
from numen.bridge.runtime import JuliaBackend
result = JuliaBackend(julia_file="dynamics.jl").solve(spec, tspan=(0.0, 5.0))
```

The scaffolded `dynamics.jl` from `numen new` gives you a working starting point.

---

## CLI reference

```
numen init [dir] [--model NAME] [--domain DOMAIN]
    Bootstrap a new project. Creates CLAUDE.md, CHARACTERIZATION.md, and
    optionally a first model. Domains: mechanical, fluid, generic.

numen check
    Smoke-test scipy, JAX, and Julia backends.

numen new NAME [--domain DOMAIN]
    Scaffold a new model directory inside an existing project.

numen list
    List built-in example models.

numen run EXAMPLE
    Run a built-in example (oscillator, coupled_spring, fluid_poppet,
    nonlinear_oscillator).

numen characterize PLAN [--output FILE] [--verbose]
    Run a YAML/JSON test campaign against a model.
    PLAN is the path to a test_plan.yaml.
    --output saves results to a JSON file.
    --verbose enables DEBUG logging (per-solve timing, lock-in values).

numen info
    Print a quick-reference cheat-sheet.
```

---

## Design

See [DESIGN.md](DESIGN.md) for architectural decisions, the ODE vs. DAE boundary,
Multibody.jl plans for 3D mechanisms, and open questions.
