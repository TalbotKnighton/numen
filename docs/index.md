# Numen

A Python-first framework for engineering dynamics simulation, with **Julia as the strategic production backend**.
Define your physics model in Python, then solve it with the full [SciML / `OrdinaryDiffEq.jl`](https://docs.sciml.ai/OrdinaryDiffEq/stable/) ecosystem — including stiff implicit solvers and DAE support that JAX simply can't provide.

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
- **Future-proof.** Opens [Multibody.jl](https://docs.sciml.ai/Multibody/stable/) integration for 3D constrained mechanisms.

### Backend honesty

| Backend | Use for | Stiff? | DAE? | Notes |
|---|---|:---:|:---:|---|
| **`JuliaServerBackend`** ⭐ | **Production, stiff problems, parameter sweeps** | ✓ | ✓ | Full `OrdinaryDiffEq.jl` solver set; sparse Jacobian; JIT amortised across session |
| `ScipyBackend` | Development, debugging | (LSODA only) | — | Pure Python, no Julia install required |
| `JAXBackend` | **Only when you need autodiff through the solve** | weak | — | Fast on small non-stiff problems; explicit solvers diverge on stiff systems |

!!! warning "About performance numbers"
    A small non-stiff benchmark in this repo (the fluid poppet) shows JAX at ~6 ms warm, Julia at ~14 ms, scipy at ~9 s. **Don't trust this for your real model.** That benchmark is intentionally tiny and non-stiff so it runs on every backend; representative engineering problems are stiff, and JAX often **fails outright** on them while Julia handles them comfortably with `Rodas5P` and the sparse-Jacobian path. Always benchmark your own model.

## Installation

```bash
pip install numen
```

Optional extras:

```bash
pip install "numen[jax]"              # JAX backend (differentiable solves, GPU batches)
pip install "numen[characterization]" # DOE sweeps, parameter grids
pip install "numen[dev]"              # pytest + coverage
```

For the Julia backend (recommended), install [Julia ≥ 1.10](https://julialang.org/downloads/) and add it to your PATH. The first solve auto-installs required Julia packages.

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
from numen.bridge.scipy_backend import ScipyBackend       # dev/debug
# from numen.bridge.server_backend import JuliaServerBackend  # production

World  = GenericWorld[BallComponent, GravitySystem, None]
world  = World(
    components={"ball": {"ball": BallComponent(position=100.0)}},
    systems={"gravity": GravitySystem()},
)
spec   = compile_spec(world)
result = ScipyBackend().solve(spec, tspan=(0.0, 5.0))
```

For production / repeated solves, the JuliaServerBackend is a drop-in replacement:

```python
from numen.bridge.server_backend import JuliaServerBackend
with JuliaServerBackend(julia_file="dynamics.jl", method="Tsit5",
                        rtol=1e-8, atol=1e-10) as srv:
    result = srv.solve(spec, tspan=(0.0, 5.0))
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
| `pneumatic_dashpot` | Stiff ODE, Rodas5P, parameter sweeps, DAE example |
