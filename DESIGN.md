# Numen Design Document

## Vision

A general-purpose, hybrid Python/Julia simulation framework for engineering dynamics. Python owns the spec, validation, and serialization layer. Julia owns the numerical solver. The two communicate through a thin bridge.

Target simulation domains:
- Lumped-parameter fluid/thermal networks
- 1D, 2D, and 6DOF rigid body dynamics
- Arbitrary coupled ODE/DAE systems

---

## Why hybrid Python + Julia?

| Concern | Python | Julia |
|---|---|---|
| Spec authoring, validation | Pydantic — excellent | Weak ecosystem |
| Serialization / snapshots | JSON, human-readable | Verbose, manual |
| ECS ergonomics | Natural | Awkward |
| ODE/DAE solving | Diffrax — good | DiffEq.jl — best in class |
| DAE + algebraic constraints | Limited | Native in MTK |
| Event / precision timing | Workable | Mature, well-documented |
| Solver variety | ~15 | ~100 |

Python is where users write models. Julia is where math happens. Rust is not in scope — MTK already provides the performance and solver quality that Rust would require rebuilding from scratch.

---

## Repo Structure

```
numen/
  pyproject.toml          # Python package
  src/numen/
    fields.py             # Field annotations + EntityGroup
    spec/                 # Pydantic ECS layer
      component.py        # Component base class
      system.py           # System base class + DynamicsFn Protocol
      world.py            # GenericWorld
    compiler/
      flatten.py          # compile_spec, CompiledSpec, CompiledSystem
    bridge/
      scipy_backend.py    # Pure-Python development backend
      julia_backend.py    # juliacall bridge (skeleton)
      runtime.py          # SolveResult
    reconstruction/
      snapshot.py         # Flat arrays → Pydantic snapshots
      collector.py        # SnapshotCollector
  julia/
    Project.toml          # Julia package
    src/
      Numen.jl
      solver.jl
      events.jl
  examples/
    oscillator/           # Single-entity 1D harmonic oscillator
    coupled_spring/       # Multi-entity spring chain (coupled system)
```

---

## Layer 1: Field Annotations (Python)

Fields on a Component are annotated to declare their role in the solver. The annotation is the sole source of truth for how a field enters the flat arrays.

```python
from typing import Annotated
from numen.fields import IntegratedField, DiscreteField, ContinuousField, ParameterField

class TankComponent(Component):
    pressure:    Annotated[float, IntegratedField()]       # continuous state, solver integrates ṗ
    temperature: Annotated[float, IntegratedField()]       # continuous state
    valve_cmd:   Annotated[float, DiscreteField(dt=0.01)] # ZOH, updates every 10ms
    volume:      Annotated[float, ParameterField()]        # constant, goes into p not x
    flow_out:    Annotated[float, ContinuousField()]       # algebraic output, computed not integrated
```

| Annotation | Solver role |
|---|---|
| `IntegratedField(size=N)` | Enters state vector `x`; solver integrates `ẋ = f(...)` |
| `ContinuousField(size=N)` | Algebraic / output; computed from `x` at runtime |
| `DiscreteField(dt=T, size=N)` | ZOH input/state; updated at fixed intervals, injects required solver times |
| `ParameterField(size=N)` | Constant; enters parameter vector `p`, never integrated |

Array-valued fields use `size=N` explicitly (e.g. `IntegratedField(size=3)` for a 3D vector). Scalar `float` defaults to `size=1`.

---

## Layer 2: ECS Spec (Python)

### Components

Components are frozen Pydantic models — pure data, no topology. Each field is annotated with its solver role. Components never store references to other entities.

```python
class MassComponent(Component):
    kind:     Literal["mass"] = "mass"
    position: Annotated[float, IntegratedField()] = 0.0
    velocity: Annotated[float, IntegratedField()] = 0.0
    mass:     Annotated[float, ParameterField()]  = 1.0

class SpringComponent(Component):
    kind:        Literal["spring"] = "spring"
    k:           Annotated[float, ParameterField()] = 1.0
    rest_length: Annotated[float, ParameterField()] = 0.0
    # No mass_a / mass_b here — topology belongs to the system, not the component.
```

### Systems

Systems declare which entities they operate on and the dynamics function that runs on them. The system carries both the behavior type and the connection topology.

```python
from numen.fields import EntityGroup
from numen.spec.system import System, DynamicsFn

class SpringForceSystem(System):
    # Class variables — statically declared, not Pydantic fields, not serialized:
    component_types: ClassVar[tuple[type, ...]] = ()
    entity_slots:    ClassVar[EntityGroup]      = EntityGroup(MassComponent, SpringComponent, MassComponent)
    python_fn:       ClassVar[DynamicsFn]       = staticmethod(spring_force_dynamics)

    # Pydantic fields — serialized:
    kind:         Literal["spring_force"] = "spring_force"
    dynamics_fn:  str = "SpringDynamics.spring_force_dynamics!"
    entity_groups: list[list[str]] = []   # declared at world-construction time
```

#### Auto-populated single-type systems

When `component_types` is set and `entity_slots` is `None`, `compile_spec` finds all matching entities automatically:

```python
class MassKinematicsSystem(System):
    component_types: ClassVar[tuple[type, ...]] = (MassComponent,)
    python_fn:       ClassVar[DynamicsFn]       = staticmethod(mass_kinematics_dynamics)
    kind:            Literal["mass_kinematics"] = "mass_kinematics"
    dynamics_fn:     str = "Dynamics.mass_kinematics!"

# Instantiate with no args — compile_spec discovers all MassComponent entities.
MassKinematicsSystem()
```

#### Multi-slot coupled systems

When `entity_slots` is set, `compile_spec` validates each group in `entity_groups` against the declared slot types at compile time. Topology lives in the system, not in components.

```python
SpringForceSystem(entity_groups=[
    ["m1", "s1", "m2"],   # slot 0: MassComponent, slot 1: SpringComponent, slot 2: MassComponent
    ["m2", "s2", "m3"],   # compile_spec validates each entry's component type
])
```

`entity_slots.size` becomes `CompiledSystem.group_size`. Wrong types in the wrong slots raise `TypeError` at `compile_spec` time.

### EntityGroup

`EntityGroup` is a metadata descriptor analogous to `IntegratedField(size=N)`. It bundles the slot type declarations with an implicit group size:

```python
EntityGroup(MassComponent, SpringComponent, MassComponent)
# .slot_types = (MassComponent, SpringComponent, MassComponent)
# .size       = 3
```

### World

The world is a generic typed container. `CC`, `SC`, `BC` are discriminated unions — every component, system, and callback has a `kind` literal discriminator.

```python
class GenericWorld(BaseModel, Generic[CC, SC, BC]):
    components: dict[str, CC]
    systems:    dict[str, SC]
    callbacks:  dict[str, BC]
```

The entity map is **static**: adding or removing entities requires stopping the simulation, updating the world, and recompiling. This is a deliberate design choice — the static topology is what allows Julia to bake entity indices into generated code as compile-time constants.

### Model-level validators

Pydantic `@model_validator` hooks are where domain rules live:

```python
@model_validator(mode="after")
def check_network_bidirectional(self) -> "FluidNetwork":
    for node_id, node in self.components.items():
        for neighbor_id in node.connects_to:
            if node_id not in self.components[neighbor_id].connects_to:
                raise ValueError(f"Connection {node_id}→{neighbor_id} is not bidirectional")
    return self
```

Validation runs on construction and on JSON load.

---

## Layer 3: Compilation (Python)

`compile_spec(world)` walks a `GenericWorld` once and produces a flat `CompiledSpec`. This is the boundary between the Python spec layer and the numerical layer.

### Steps

1. **Field scan** — iterate all components; `IntegratedField` / `DiscreteField` / `ContinuousField` entries enter state vector `x` in entity insertion order; `ParameterField` entries enter parameter vector `p`.
2. **Parameter scan** — same pass populates `param_index_map`.
3. **Discrete schedule** — collect `DiscreteField` update rates; these become required solver evaluation times.
4. **System resolution** — for each system:
   - If `entity_groups` is set: validate each group against `entity_slots.slot_types`, flatten into `entity_ids`.
   - If `entity_ids` is set manually: validate against `component_types`.
   - If only `component_types` is set: auto-populate by scanning world components.
5. **Group caching** — build `entity_groups: tuple[tuple[str, ...], ...]` on `CompiledSystem` once; the solver hot loop reads the cached tuple, never re-slices.

### CompiledSpec

```python
@dataclass
class CompiledSpec:
    state_size:      int
    param_size:      int
    state_index_map: dict[str, tuple[int, int]]   # "entity.field" → (start, end) in x
    param_index_map: dict[str, tuple[int, int]]   # "entity.field" → (start, end) in p
    discrete_dts:    list[float]
    x0:              list[float]
    p:               list[float]
    systems:         list[CompiledSystem]
```

### CompiledSystem

```python
@dataclass
class CompiledSystem(Generic[GroupT]):
    dynamics_fn:   str                           # Julia function reference
    entity_ids:    list[str]                     # flat; serialized to JSON for Julia
    group_size:    int                           # entities per group (from EntityGroup.size)
    entity_groups: tuple[GroupT, ...]            # pre-grouped, immutable; Python backend only
    python_fn:     DynamicsFn | None             # Python callable; not serialized
```

`GroupT` encodes the group arity: `CompiledSystem[tuple[str]]` for single-entity systems, `CompiledSystem[tuple[str, str, str]]` for 3-slot systems. The type checker validates unpacking at each call site.

`CompiledSpec.to_dict()` serializes everything except `python_fn` and `entity_groups` — only `entity_ids` and `group_size` cross the Julia bridge.

---

## Layer 4: Dynamics Functions (Python + Julia)

Dynamics functions are **separate artifacts** for each backend. Python and Julia have different calling conventions because they have fundamentally different execution models.

### Python dynamics functions (scipy backend)

Python functions are used during development via the `ScipyBackend`. The calling convention is:

```python
def spring_force_dynamics(
    dx:     np.ndarray,
    x:      np.ndarray,
    p:      np.ndarray,
    t:      float,
    spec:   CompiledSpec,
    system: CompiledSystem[tuple[str, str, str]],
) -> None:
    for id_a, id_s, id_b in system.entity_groups:
        a = spec.view(id_a, MassComponent,   x, p)   # returns MassComponent to the type checker
        b = spec.view(id_b, MassComponent,   x, p)
        s = spec.view(id_s, SpringComponent, x, p)

        stretch = (b.position - a.position) - s.rest_length
        force   = s.k * stretch

        da = spec.dx_view(id_a, MassComponent, dx)
        db = spec.dx_view(id_b, MassComponent, dx)
        da.velocity +=  force / a.mass   # += because multiple systems may contribute
        db.velocity += -force / b.mass
```

Key design points:

- **`spec.view(id, ComponentType, x, p) -> ComponentType`** — returns a `ComponentView` at runtime but is typed as `ComponentType` for the type checker, so `a.mass`, `a.position`, etc. are known fields.
- **`spec.dx_view(id, ComponentType, dx) -> ComponentType`** — same trick for the derivative write proxy (`DerivativeView`).
- **`+=` (linearly additive)** — `dx` is zeroed before each RHS call. Multiple systems contribute independently to the same derivative slots by accumulating. This matches the pathfinder design.
- **`DynamicsFn` is a `Protocol[GroupT]`** — static type checkers validate every dynamics function against the declared calling convention.

`ComponentView` and `DerivativeView` are lightweight proxy objects. They do string-keyed dict lookups at Python runtime, but for the Julia backend this layer does not exist — integer indices are baked in directly.

### Julia dynamics functions (production backend)

Julia functions have no `entity_groups` argument. Entity indices are resolved at **code generation time** from `CompiledSpec` and written as integer literals into the generated function body:

```julia
# Generated by the Julia codegen — no dict, no string, no entity_groups
function spring_force_dynamics!(dx, x, p, t)
    # group 0: m1=state[0:2], s1=params[0:2], m2=state[2:4]
    stretch = (x[3] - x[1]) - p[2]
    force   = p[1] * stretch
    dx[2]  +=  force / p[0]
    dx[4]  += -force / p[5]
    # group 1: ...
end
```

`dynamics_fn: str = "MyModule.function!"` on each `System` is the reference the bridge uses to locate the Julia function. Python and Julia functions are declared in the same file alongside the system class to keep them co-located.

---

## Layer 5: Coordinate Frames

A reference frame is a component like any other:

```python
class ReferenceFrame(Component):
    position:   Annotated[Array3, IntegratedField(size=3)]
    quaternion: Annotated[Array4, IntegratedField(size=4)]  # [w, x, y, z]
    parent_id:  str | None = None   # resolved to int index at compile time
```

At compile time, all frames are topologically sorted. At runtime (Julia), forward kinematics is a sequential scan — no recursion, no dict lookups:

```julia
function forward_kinematics(local_transforms, parent_indices)
    world = similar(local_transforms)
    for i in eachindex(parent_indices)
        pidx = parent_indices[i]
        parent_world = pidx < 0 ? identity_transform : world[pidx]
        world[i] = compose(parent_world, local_transforms[i])
    end
    world
end
```

---

## Layer 6: Bridge (Python ↔ Julia)

The bridge uses **juliacall** (Python package) / **PythonCall.jl** (Julia package), which allows Python to call Julia functions directly in-process with minimal overhead.

### ScipyBackend (development)

A pure-Python backend using `scipy.integrate.solve_ivp`. Requires `python_fn` to be declared on every system. No Julia required.

```python
result = ScipyBackend(rtol=1e-9, atol=1e-9).solve(spec, tspan=(0.0, 10.0))
```

The RHS loop:
```python
def rhs(t, x):
    dx = np.zeros_like(x)
    for sys in compiled_spec.systems:
        sys.python_fn(dx, x, p, t, compiled_spec, sys)  # sys is the CompiledSystem
    return dx
```

### JuliaBackend (production)

```python
spec_json = json.dumps(spec.to_dict())
result = jl.Numen.solve(spec_json, tspan[0], tspan[1])
```

`spec.to_dict()` serializes `entity_ids` and `group_size` for each system (not `entity_groups` or `python_fn` — those are Python-only).

---

## Layer 7: Julia Solver

Julia receives the `CompiledSpec` JSON and owns everything numerical.

```julia
module Numen
using ModelingToolkit, DifferentialEquations, JSON3

function solve(spec_json::String, t0::Float64, tf::Float64)
    spec = JSON3.read(spec_json, CompiledSpec)
    sys  = build_mtk_system(spec)
    prob = ODEProblem(sys, spec.x0, (t0, tf), spec.p)
    sol  = solve(prob, Rodas5(), saveat=spec.discrete_times)
    return (t=sol.t, x=hcat(sol.u...))
end
end
```

MTK handles:
- DAE index reduction (algebraic constraints in fluid networks)
- Adaptive stepping with required evaluation times for discrete events
- Jacobian sparsity detection
- Choice of integrator (Rodas5 for stiff, Tsit5 for non-stiff, etc.)

---

## Layer 8: Reconstruction (Python)

After solving, Python uses the index maps to reconstruct structured snapshots:

```python
collector = SnapshotCollector(world, spec, result)
snap = collector.at(2.5)          # interpolates to nearest solver step
osc  = snap.components["osc"]    # full typed Pydantic component
print(osc.position, osc.velocity)

t, positions = collector.field_series("m1", "position")  # direct array extraction
```

The snapshot is a full Pydantic object — serializable to JSON, validatable, diffable.

---

## Event / Timing System

Discrete systems inject their required evaluation times directly into the solver rather than LCM chunking. This is MTK's native `tstops` / `saveat` mechanism.

For continuous events (zero-crossings, threshold triggers), MTK/DiffEq.jl has a mature `ContinuousCallback` / `DiscreteCallback` system.

**Open question:** How do callbacks communicate back to the Python layer mid-simulation?
- Callbacks run entirely in Julia (fast, limited to Julia logic)
- Solver pauses at callback time, yields to Python, resumes (flexible, slower)
- Hybrid: Julia callbacks update state, Python post-processes at snapshots

---

## Callback Architecture

### Tier 1 — Julia callbacks (tight loop)

For state modifications that must happen inside the integrator loop with full precision timing:

```julia
valve_event = ContinuousCallback(
    (x, t, integrator) -> x[idx.tank_a.pressure] - p[idx.params.valve_threshold],
    (integrator) -> integrator.u[idx.valve.open] = 1.0
)
```

### Tier 2 — Python callbacks (discrete / infrequent)

For control logic updates, logging, and snapshots. Called at `DiscreteField` update times injected into `tstops` — the solver naturally pauses at these points.

```python
class ControlCallback(Callback):
    kind: Literal["control_callback"] = "control_callback"
    dt: float = 0.01

    def __call__(self, world: GenericWorld, t: float) -> dict[str, float]:
        ...
```

---

## CLI

All framework operations exposed through a Python CLI (Typer + Rich):

```
numen run   <world.json> --tspan 0 100 --output results/
numen check <world.json>
numen snap  <results/> --time 42.5 --output snap.json
numen plot  <results/>
numen init  <project-name>
```

---

## Open Questions

1. **Stiffness detection** — automatic solver selection (inspect Jacobian sparsity) or user-specified integrator?
2. **Batching / Monte Carlo** — multiple initial conditions in parallel via DiffEq.jl ensemble problems?
3. **Callback mid-simulation protocol** — how Python callbacks yield control back to the Julia integrator.
