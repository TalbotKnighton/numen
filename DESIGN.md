# Numen Design Document

## Vision

A general-purpose, hybrid Python/Julia simulation framework for engineering
dynamics.  Python owns the spec, validation, and serialization layer.  Julia
owns the numerical solver.  The two communicate through a thin JSON bridge.

Target simulation domains:
- Lumped-parameter fluid/thermal networks
- 1D and 6DOF rigid body dynamics
- Arbitrary coupled ODE/DAE systems

---

## Why hybrid Python + Julia?

| Concern | Python | Julia |
|---|---|---|
| Spec authoring, validation | Pydantic — excellent | Weak ecosystem |
| Serialization / snapshots | JSON, human-readable | Verbose, manual |
| ECS ergonomics | Natural | Awkward |
| ODE/DAE solving | Diffrax (JAX) — good | DifferentialEquations.jl — best in class |
| DAE + algebraic constraints | Limited | Native mass-matrix + implicit solvers |
| Event / precision timing | Workable (segment-solve) | Mature PeriodicCallback / DiscreteCallback |
| Solver variety | ~15 | ~100 |

Python is where users write models.  Julia is where math happens.

---

## Bridge architecture (current)

Python → Julia communication is via **subprocess JSON**, not juliacall / PythonCall.

```
Python: compile_spec(world) → CompiledSpec → .to_dict() → JSON payload
            ↓ subprocess (JuliaBackend) or stdin/stdout (JuliaServerBackend)
Julia:  JSON3.read(payload, SolvePayload) → solve() → JSON result
            ↓
Python: np.array(data["t"]), np.array(data["x"])
```

Two modes:
- `JuliaBackend` — fresh subprocess per `solve()` call.  ~6 s cold start, ~14 ms warm.
- `JuliaServerBackend` — persistent subprocess, pay JIT once, warm-solve forever.
- `JuliaServerPool` — N parallel servers for parameter sweeps.

No `juliacall` dependency.  Julia is a black box that reads JSON and writes JSON.

---

## Repo structure

```
src/numen/
  fields.py              IntegratedField, ContinuousField, DiscreteField, ParameterField, EntityGroup
  spec/
    component.py         Component base (frozen Pydantic)
    system.py            System base + DynamicsFn Protocol
    callback.py          Callback base — dt, julia_fn, params, python_fn ClassVar
    world.py             GenericWorld[CC, SC, BC]
  compiler/
    flatten.py           compile_spec() → CompiledSpec, CompiledSystem, CompiledCallback
  bridge/
    scipy_backend.py     ScipyBackend — segment-solve for callbacks
    jax_backend.py       JAXBackend   — JIT via diffrax
    runtime.py           JuliaBackend + SolveResult
    server_backend.py    JuliaServerBackend + JuliaServerPool
  errors.py              NumenError, NumenFeatureError, NumenMissingFnError
  logging.py             configure_logging()
  reconstruction/
    collector.py         SnapshotCollector
    snapshot.py          WorldSnapshot

julia/src/
  Numen.jl               Module entry
  types.jl               CompiledSpec, CompiledCallbackSpec, SolvePayload structs
  solver.jl              solve(), build_dynamics(), build_tstops()
  events.jl              build_callbacks(), check_dae_solver(), _resolve_fn()

docs/
  architecture.md        Field types, differential_mask convention, callback architecture

examples/
  oscillator/            Minimal 1D harmonic oscillator
  coupled_spring/        Multi-entity spring chain
  fluid_poppet/          Pneumatic 4-CV network + spring-mass poppet  ← best reference
```

---

## Layer 1: Field annotations

Every field annotation declares its role in the solver.  The annotation is the
**sole source of truth** for how the field enters the flat arrays.

| Annotation | Vector | Role |
|---|---|---|
| `IntegratedField(size=N)` | state `x` | `ẋ = f(...)` — solver integrates |
| `ParameterField(size=N)` | param `p` | constant for the solve |
| `ContinuousField(size=N, algebraic=False)` | state `x` | output variable — fn writes derived quantity each RHS |
| `ContinuousField(size=N, algebraic=True)` | state `x` | algebraic constraint — fn writes residual `g(x)=0`; `differential_mask=0` |
| `DiscreteField(dt, size=N)` | state `x` | ZOH — updated by controller callback, injects tstops |

`size=N` works on all field types.  See `docs/architecture.md` for the `differential_mask` convention.

---

## Layer 2: ECS spec

### Components

Frozen Pydantic models.  Pure data, no topology, no solver knowledge.

```python
class MassComponent(Component):
    kind:     Literal["mass"] = "mass"
    position: Annotated[float, IntegratedField()] = 0.0
    velocity: Annotated[float, IntegratedField()] = 0.0
    mass:     Annotated[float, ParameterField()]  = 1.0
```

### Systems

Declare which entities they operate on and the dynamics function.  Topology
lives in the system, not in components.

```python
class SpringForceSystem(System):
    component_types: ClassVar[tuple[type, ...]] = ()
    entity_slots:    ClassVar[EntityGroup]      = EntityGroup(Mass, Spring, Mass)
    python_fn:       ClassVar[DynamicsFn]       = staticmethod(spring_force_dynamics)
    kind:            Literal["spring_force"] = "spring_force"
    dynamics_fn:     str = "SpringDyn.spring_force!"
    entity_groups:   list[list[str]] = []
```

### Callbacks

Controller callbacks fire at fixed period `dt` between ODE segments (scipy)
or inside the integrator via `PeriodicCallback` (Julia).

```python
class PIDCallback(Callback):
    kind:      Literal["pid"] = "pid"
    dt:        float = 0.01
    julia_fn:  str   = "MyDyn.pid!"
    params:    dict[str, float] = {"kp": 1.0}
    python_fn: ClassVar = staticmethod(pid_fn)
```

### World

```python
World = GenericWorld[AnyComponent, AnySystem, AnyCallback]
```

The entity map is **static** — topology is baked in at `compile_spec()`.

---

## Layer 3: Compilation

`compile_spec(world)` walks the world once and produces a flat `CompiledSpec`.

Steps:
1. **Field scan** — build `state_index_map`, `param_index_map`, `x0`, `p`, `differential_mask`.
2. **Feature detection** — populate `required_features` frozenset from field types present.
3. **System resolution** — validate entity groups, cache `entity_groups` tuple.
4. **Callback compilation** — validate `dt > 0`, build `CompiledCallback` list.

`differential_mask[i]` = 1.0 for integrated/discrete slots, 0.0 for algebraic
`ContinuousField` slots.  Always length `state_size`.

`CompiledSpec.to_dict()` serializes everything except `python_fn` and `entity_groups`.

---

## Layer 4: Backend compatibility

Each backend declares `supported_features: ClassVar[frozenset[str]]`.
`solve()` checks `required_features ⊆ supported_features` before starting.

| Feature | Scipy | JAX | JuliaBackend | JuliaServerBackend |
|---|:---:|:---:|:---:|:---:|
| `vector_fields` | ✓ | ✓ | ✓ | ✓ |
| `discrete_fields` | ✓ | ✓ | ✓ | ✓ |
| `continuous_fields` | ✓ | ✓ | ✓ | ✓ |
| `control_callbacks` | ✓ | ✗ | ✓ | ✓ |
| `dae_constraints` | ✗ | ✗ | ✓ | ✓ |

---

## Layer 5: Dynamics functions

### Python (scipy / JAX)

```python
def spring_force_dynamics(dx, x, p, t, spec, system):
    for id_a, id_s, id_b in system.entity_groups:
        a  = spec.view(id_a, MassComponent,   x, p)
        b  = spec.view(id_b, MassComponent,   x, p)
        s  = spec.view(id_s, SpringComponent, x, p)
        da = spec.dx_view(id_a, MassComponent, dx)
        db = spec.dx_view(id_b, MassComponent, dx)
        force = s.k * ((b.position - a.position) - s.rest_length)
        da.velocity +=  force / a.mass
        db.velocity += -force / b.mass
```

- `spec.view()` returns a `ComponentView` (read-only, attribute access)
- `spec.dx_view()` returns a `DerivativeView` (write-only via `+=`)
- `+=` is additive — multiple systems accumulate independently into the same slots
- JAX: always use `jnp.*`, never `np.*`, never bare `if` on state values

### Julia

```julia
function spring_force!(dx, x, p, t, spec, sys)
    gs = sys.group_size
    for i in 1:gs:length(sys.entity_ids)
        id_a = sys.entity_ids[i];  id_s = sys.entity_ids[i+1];  id_b = sys.entity_ids[i+2]
        pos_a = x[state_idx(spec, id_a * ".position")]
        pos_b = x[state_idx(spec, id_b * ".position")]
        k     = p[param_idx(spec, id_s * ".k")]
        # ...
    end
end
```

Use `state_range` / `param_range` for vector fields (`UnitRange{Int}`).

### Julia callbacks

```julia
function pid!(integrator, spec, params)
    i = state_idx(spec, "actuator.force")
    err = integrator.u[state_idx(spec, "sensor.angle")] - params["setpoint"]
    integrator.u[i] = params["kp"] * err
end
```

---

## Layer 6: Julia solver

`runner.jl` (subprocess) and `server.jl` (persistent) both call `Numen.solve(json)`.

`solver.jl` dispatch:
1. Resolve `dynamics_fn` strings → Julia callables via `_resolve_fn()`.
2. Build `PeriodicCallback` for each compiled callback via `build_callbacks()`.
3. If any `differential_mask` entry is 0.0: build `ODEFunction(dynamics!, mass_matrix=Diagonal(mask))`.
4. Call `OrdinaryDiffEq.solve(prob, solver; tstops, saveat, callback=cb_set)`.

`_resolve_fn("Module.fn!")` does symbol lookup in already-loaded scope —
never `eval` of arbitrary code.

---

## Performance notes

**Warm timings (fluid poppet example, 150 ms simulation):**

| Backend | Warm solve | vs scipy |
|---|---|---|
| ScipyBackend | 9048 ms | baseline |
| JAXBackend (Dopri5) | 6 ms | 1507× |
| JuliaBackend (Tsit5) | 14 ms | 634× |

JAX cold (JIT compile): ~550 ms.
Julia cold (subprocess startup + JIT): ~6700 ms.
JuliaServerBackend amortizes JIT over many solves.

---

## Solver selection guide

| Problem type | Recommended | Avoid |
|---|---|---|
| Non-stiff ODE | `JAXBackend(solver="Dopri5")` | Tsit5 with tight atol |
| Stiff ODE | `JuliaServerBackend(method="Rodas5P")` | JAX implicit (slow JIT) |
| DAE (algebraic constraints) | `JuliaServerBackend(method="Rodas5P")` | scipy, JAX (unsupported) |
| Development / debugging | `ScipyBackend()` | — |
| Parameter sweep | `JuliaServerPool(n_workers=N)` | JuliaBackend (per-call startup) |

---

## Open questions / roadmap

### `build_mtk_system` (unimplemented)

For **joint-constrained multibody** (revolute, prismatic, spherical joints),
the current mass-matrix DAE approach covers index-1 algebraic constraints but
not the higher-index DAEs that arise from kinematic chains.  ModelingToolkit.jl
with index reduction is the correct path.  Not needed until 3D constrained
mechanisms are in scope.

### Multibody.jl

3D rigid body mechanisms with joint constraints.  Requires `build_mtk_system`.
Deferred until `build_mtk_system` is implemented.

### Symbolic codegen

Writing dynamics in Python and Julia separately is the current approach.
A future `symbolic_rhs` method on System could auto-generate both the Python
callable (via `sympy.lambdify`) and the Julia function (via `sympy.julia_code`).
Not implemented; hand-authoring is acceptable given the derivations only happen
once.

### Continuous events (zero-crossings)

Julia's `ContinuousCallback` for threshold triggers (valve open/close, contact
onset) is not yet wired from the Python spec layer.  Currently handled by
smooth ramp functions (see `_soft_pen` in CLAUDE.md).

### Batching / ensemble

Multiple initial conditions in parallel via DifferentialEquations.jl
`EnsembleProblem`.  Currently handled at the Python level via `JuliaServerPool`.
