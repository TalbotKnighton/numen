# Numen

[![Documentation](https://img.shields.io/badge/docs-online-2563eb?logo=mkdocs&logoColor=white)](https://TalbotKnighton.github.io/numen/)
[![Deploy Docs](https://github.com/TalbotKnighton/numen/actions/workflows/docs.yml/badge.svg)](https://github.com/TalbotKnighton/numen/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/downloads/)
[![Julia 1.10+](https://img.shields.io/badge/julia-1.10+-9558B2?logo=julia&logoColor=white)](https://julialang.org/downloads/)

A Python-first framework for engineering dynamics simulation, with **Julia as the strategic production backend**.
Define your physics model in Python, then solve it with the full [SciML / `OrdinaryDiffEq.jl`](https://docs.sciml.ai/OrdinaryDiffEq/stable/) solver ecosystem — including stiff implicit solvers and DAE support that JAX simply can't provide.

```
         Python                          Backends
  ┌──────────────────────┐      ┌─────────────────────────────────┐
  │  Component (data)    │      │  JuliaServerBackend  ★ default  │
  │  System    (physics) │─────▶│  ScipyBackend — dev/debug       │
  │  World     (model)   │      │  JAXBackend  — diff/batch only  │
  └──────────────────────┘      └─────────────────────────────────┘
```

## Use Julia for real work

The Julia backend is a thin wrapper over [`OrdinaryDiffEq.jl`](https://docs.sciml.ai/OrdinaryDiffEq/stable/) — **~150+ solvers** selectable by string name (`method="Rodas5P"`, `"Tsit5"`, `"FBDF"`, …). The integration is intentionally shallow: you get the full SciML universe, not a curated subset.

What makes Julia the right answer for engineering dynamics:

- **Stiff problems work.** Real engineering systems (fluid networks, electromechanical systems, thermal/structural coupling) are routinely stiff. Julia ships state-of-the-art stiff solvers (`Rodas5P`, `Rosenbrock23`, `KenCarp4`, `FBDF`, `QNDF`, `TRBDF2`). JAX's explicit solvers diverge on these; its implicit solvers are slow to JIT and don't ship a comparable solver set.
- **DAEs work.** Algebraic constraints (pressure equality, joint constraints, conservation residuals) via the mass-matrix path with `ContinuousField(algebraic=True)`. **Julia-only** — scipy and JAX raise `NumenFeatureError`.
- **Sparse Jacobian with auto-coloring.** Numen builds a `jac_prototype` from the entity-group graph; OrdinaryDiffEq applies SparseDiffTools matrix coloring. Jacobian cost stays roughly O(group_coupling_width), not O(state_size). Large multi-entity models remain fast.
- **JIT amortisation.** `JuliaServerBackend` keeps a hot Julia process across the entire session — pays compilation **once**, every subsequent solve is warm. `JuliaServerPool` runs N pre-warmed workers in parallel for parameter sweeps and DOEs.
- **Future-proof.** Opens [Multibody.jl](https://docs.sciml.ai/Multibody/stable/) integration for 3D constrained mechanisms (see [DESIGN.md](DESIGN.md)).

### Backend honesty

| Backend | Use for | Stiff? | DAE? | Notes |
|---|---|:---:|:---:|---|
| **`JuliaServerBackend`** ⭐ | **Production work, stiff problems, parameter sweeps** | ✓ | ✓ | Full `OrdinaryDiffEq.jl` solver set; sparse Jacobian; JIT amortised across session |
| `ScipyBackend` | Development, debugging, first runs | (LSODA only) | — | Pure Python, no Julia install required |
| `JAXBackend` | **Only when you need autodiff through the solve** | weak | — | Fast on small non-stiff problems; explicit solvers diverge on stiff systems; implicit solvers slow to JIT |

**About performance numbers.** A small non-stiff benchmark in this repo (the fluid poppet example) shows JAX at ~6 ms warm, Julia at ~14 ms, scipy at ~9 s. **Don't trust this for your real model.** That benchmark is intentionally tiny and non-stiff so it can run on every backend; representative engineering problems are stiff, and JAX often **fails outright** on them while Julia handles them comfortably with `Rodas5P` and the sparse-Jacobian path. Always benchmark your own model.

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

## Julia backend (recommended for production)

For each Python `System`, write a matching Julia function in a `.jl` file using
the readable helper API:

```julia
# dynamics.jl
module MyDynamics
import Main: CompiledSpec, CompiledSystemSpec, groups,
             get_state, get_param, add_deriv!

function gravity_dynamics!(
    dx  :: AbstractVector{T},
    x   :: AbstractVector{S},
    p   :: Vector{Float64},
    t   :: Real,
    spec:: CompiledSpec,
    sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    for (eid,) in groups(sys)
        vel = get_state(spec, x, eid, "ball.velocity")
        add_deriv!(spec, dx, eid, "ball.position", vel)
        add_deriv!(spec, dx, eid, "ball.velocity", -9.81)
    end
end

end  # module MyDynamics
```

The `{T, S}` signature lets the same function serve normal solves (`Float64`)
and stiff Jacobian evaluation (`ForwardDiff.Dual`) without modification.
The scaffolded `dynamics.jl` from `numen new` is a working starting point.
See [JULIA.md](src/numen/init_data/JULIA.md) for the full API reference and
performance notes.

### Solver selection — pick by string

`method=` accepts any solver name from `OrdinaryDiffEq.jl`. A few common choices:

| Family | Solvers | Use for |
|---|---|---|
| **Non-stiff explicit RK** | `Tsit5`, `Dopri5`, `Vern7`, `Vern9`, `BS3` | Most ODEs (default: `Tsit5`) |
| **Stiff Rosenbrock** | `Rodas5P`, `Rodas4`, `Rosenbrock23` | Stiff systems, DAEs (mass-matrix) |
| **Stiff implicit RK / multistep** | `KenCarp4`, `KenCarp47`, `TRBDF2`, `FBDF`, `QNDF` | Very stiff or large systems |
| **Symplectic** | `KahanLi6`, `McAte5`, `VelocityVerlet` | Hamiltonian / energy-preserving |
| **IMEX** | `KenCarp4`, `ARKODE_ERK_BS3` | Mixed stiff/non-stiff |

See the [`OrdinaryDiffEq.jl` solver index](https://docs.sciml.ai/OrdinaryDiffEq/stable/solvers/ode_solve/) for the complete list.

### Fast iteration: server backend + pool

```python
from numen.bridge.server_backend import JuliaServerBackend, JuliaServerPool

# Persistent process — pays JIT once per session
with JuliaServerBackend(julia_file="dynamics.jl", method="Rodas5P",
                        rtol=1e-8, atol=1e-10) as srv:
    for params in trial_set:
        result = srv.solve(compile_spec(make_world(**params)), tspan=(0.0, 5.0))

# Parallel parameter sweep — N pre-warmed workers
with JuliaServerPool(n_workers=4, julia_file="dynamics.jl",
                     method="Tsit5", rtol=1e-8, atol=1e-10) as pool:
    results = pool.map(
        lambda srv, p: srv.solve(compile_spec(make_world(p)), tspan=(0.0, 1.0)),
        param_grid,
    )
```

Single-shot solves (one-off computations, scripts that exit) can use plain
`JuliaBackend(julia_file=...)` and pay the ~6 s startup once per call.

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
