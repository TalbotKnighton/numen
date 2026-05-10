# {project_name} — Design Notes

This file captures architectural decisions for this project.
The framework architecture below is fixed; the **Project decisions** section is yours.

---

## Framework architecture (read-only reference)

### How the stack fits together

```
Python (spec + validation)          Julia (solver)
──────────────────────────          ──────────────────────────────────
Component  → IntegratedField  ──┐   build_dynamics() assembles closure
           → ParameterField    │   build_tgrad()    FD time gradient
                                │   jac_prototype   sparse ∂f/∂x
compile_spec(world)            ├──► solve() → ODEFunction → ODEProblem
  → CompiledSpec (JSON)        │   OrdinaryDiffEq.solve(prob, Rodas5P())
      x0, p, index maps        │
      jac_sparsity_rows/cols  ─┘
      differential_mask
```

Python owns: model definition, compilation, result post-processing.
Julia owns: ODE/DAE integration, Jacobian, time gradient, callbacks.
Bridge: JSON over subprocess stdin/stdout (no juliacall dependency).

### Field types

| Field | Vector | Role |
|---|---|---|
| `IntegratedField(size=N)` | state `x` | Solver integrates ẋ = f(…) |
| `ParameterField(size=N)` | param `p` | Constant for the solve |
| `ContinuousField(algebraic=False)` | state `x` | Output — fn writes derived quantity |
| `ContinuousField(algebraic=True)` | state `x` | DAE residual g(x)=0; Julia only |
| `DiscreteField(dt)` | state `x` | ZOH — tstops injected; controller updates |
| `ExcitationPort(...)` | *(not compiled)* | Characterization port discovery only |

### Julia dynamics signature (required)

```julia
function my_dynamics!(
    dx  :: AbstractVector{T},
    x   :: AbstractVector{S},
    p   :: Vector{Float64},
    t   :: Real,
    spec:: CompiledSpec,
    sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
```

Two separate type parameters `{T, S}` are required because stiff solvers (Rodas5P)
call the function in two distinct AD passes:
- Jacobian pass (∂f/∂x): `dx::Vector{Dual}`, `x::Vector{Dual}`, `t::Float64`
- Normal pass: `dx::Vector{Float64}`, `x::Vector{Float64}`, `t::Float64`

The time gradient (∂f/∂t) is handled by the framework via central finite differences
(`build_tgrad` in `solver.jl`) — user functions are never called with `t::Dual`.

**Helper functions** deriving values from state must also be generic:
```julia
function soft_pen(x::T)::T where T <: Real
    x <= 0.0 && return zero(T)          # zero(T) not 0.0
    x >= STOP_DELTA && return x - 0.5 * STOP_DELTA
    return 0.5 * x * x / STOP_DELTA
end
```

### Jacobian sparsity (automatic)

`compile_spec()` computes the Jacobian sparsity pattern from the ECS entity-group
graph: all state slots for entities in the same system group are treated as fully
coupled (dense block per group); states across groups are independent.  Julia
converts this to a `SparseMatrixCSC` `jac_prototype` for `ODEFunction`, enabling
SparseDiffTools matrix coloring.  For models with N independent entities the
Jacobian cost is O(max_group_width), not O(N).

### Backends

| Backend | Warm speed | Use when |
|---|---|---|
| `ScipyBackend` | 1× | Development, debugging |
| `JAXBackend(solver="Dopri5")` | ~1500× | Repeated solves, Monte Carlo |
| `JuliaServerBackend(method="Tsit5")` | ~600× | Long runs, single trajectory |
| `JuliaServerBackend(method="Rodas5P")` | ~600× | Stiff ODEs / DAEs |
| `JuliaServerPool(n_workers=N)` | ~600× | Parallel parameter sweeps |

Stiff solver checklist:
- Use `Rodas5P` for stiff ODEs and all DAE systems
- DAE (`ContinuousField(algebraic=True)`) requires Julia + implicit solver
- `FBDF` is an alternative for very stiff or index-2+ DAEs

### Smooth contacts — always use the C¹ ramp

Sharp `max(0, -pos)` forces have a C⁰ kink that causes >99% step rejection.

```python
_STOP_DELTA = 1e-6
def _soft_pen(pos_from_stop):
    x = pos_from_stop
    return jnp.where(x <= 0, 0.0,
           jnp.where(x >= _STOP_DELTA, x - 0.5*_STOP_DELTA,
                     0.5 * x * x / _STOP_DELTA))
```

---

## Project decisions

> Record non-obvious architectural choices specific to this project here.
> Examples: why a particular solver was chosen, why a component is split a certain
> way, why a parameter is fixed vs. integrated, trade-offs accepted.

| Decision | Rationale | Revisit if… |
|---|---|---|
| *(add entries)* | | |

---

## Model inventory

| Model directory | Domain | Key components | Stiff? |
|---|---|---|---|
| *(add entries)* | | | |

---

## Known trade-offs and deferred work

<!-- List any shortcuts taken, approximations made, or features left for later. -->
