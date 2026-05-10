# Julia Reference

Numen uses Julia (via [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/)) as its fastest solver backend — typically **600× faster** than scipy warm, without JIT cost after the first solve.

---

## How it works

When you call `JuliaBackend` or `JuliaServerBackend`, Numen:

1. Serialises the `CompiledSpec` to JSON
2. Starts a Julia subprocess (or reuses a server)
3. `include`s your `dynamics.jl` file
4. Calls your module's dynamics functions inside `ODEProblem` / `DAEProblem`

The Julia side receives the same `CompiledSpec` the Python backends use, so every index lookup (`state_idx`, `param_idx`) returns the same slot as `spec.state_idx(key)` on the Python side.

---

## Dynamics function signature

Every dynamics function must have this exact signature:

```julia
function my_dynamics!(
    dx  :: AbstractVector{T},   # write: derivative accumulator
    x   :: AbstractVector{S},   # read:  current state
    p   :: Vector{Float64},     # read:  parameter vector (constant)
    t   :: Real,                # read:  current time
    spec:: CompiledSpec,        # read:  index maps
    sys :: CompiledSystemSpec,  # read:  entity IDs for this system
) where {T <: Real, S <: Real}
    # ...
end
```

**Why two type parameters `{T, S}`?**
Stiff solvers (Rodas5P, Rosenbrock23) evaluate the state Jacobian ∂f/∂x using
ForwardDiff. During those calls `x` contains `Dual` numbers — `S` covers both
`Float64` (normal solve steps) and `ForwardDiff.Dual` (Jacobian steps). The
separate `T` parameter for `dx` lets helper functions return the correct type in
both paths.

**Why `t :: Real`?**
Any future path that passes a `Dual` time will work without changes. For the
current explicit `tgrad!`, `t` is always `Float64` — `Real` covers both.

**The golden rule: always use `+=` on `dx`, never `=`.**
Multiple systems accumulate into the same `dx` slot. Direct assignment silently
zeros out contributions from other systems.

---

## Module layout

Every `dynamics.jl` file defines one or more `module` blocks. Numen `include`s
the file once, then resolves function names from `dynamics_fn` strings like
`"MyModule.my_fn!"`:

```julia
# dynamics.jl
module MyDynamics
import Main: CompiledSpec, CompiledSystemSpec, state_idx, param_idx, groups

function gravity_dynamics!(
    dx :: AbstractVector{T}, x :: AbstractVector{S}, p :: Vector{Float64},
    t  :: Real, spec :: CompiledSpec, sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    for (eid,) in groups(sys)
        i_pos = state_idx(spec, "$eid.ball.position")
        i_vel = state_idx(spec, "$eid.ball.velocity")
        dx[i_pos] += x[i_vel]
        dx[i_vel] += -9.81
    end
end

end  # module MyDynamics
```

In Python, set `dynamics_fn = "MyDynamics.gravity_dynamics!"` on the `System`.

---

## Helpers available in dynamics files

These five functions are always available via `import Main: ...`:

| Function | Returns | Description |
|---|---|---|
| `state_idx(spec, key)` | `Int` (1-based) | First Julia index for a state field |
| `param_idx(spec, key)` | `Int` (1-based) | First Julia index for a parameter field |
| `state_range(spec, key)` | `UnitRange{Int}` | Full slice for vector state fields (`size=N`) |
| `param_range(spec, key)` | `UnitRange{Int}` | Full slice for vector parameter fields |
| `groups(sys)` | iterator | Yields entity-id groups for destructuring |

Keys use the full path `"entity_id.component_kind.field_name"`, matching Python's `spec.state_index_map`.

```julia
# Scalar field
i_pos = state_idx(spec, "osc.oscillator.position")

# Vector field (size=8)
r = state_range(spec, "node.beam.displacement")   # UnitRange 1:8
x[r]                                               # 8-element view
```

---

## Iterating entity groups

Use `groups(sys)` and tuple-destructure each group — this mirrors the Python
`for entity_group in system.entity_groups:` pattern and gives semantic names to
the entities in each group:

```julia
# group_size = 1 — one entity per group
for (eid,) in groups(sys)
    i_pos = state_idx(spec, "$eid.oscillator.position")
    # ...
end

# group_size = 3 — coupled triplet [cv_a, orifice, cv_b]
for (cv_a, orifice, cv_b) in groups(sys)
    P_a = x[state_idx(spec, "$cv_a.control_volume.pressure")]
    A   = p[param_idx(spec, "$orifice.orifice.area")]
    P_b = x[state_idx(spec, "$cv_b.control_volume.pressure")]
    # ...
end
```

Internally, `groups(sys) === Iterators.partition(sys.entity_ids, sys.group_size)`
— a zero-allocation iterator over fixed-size slices of `sys.entity_ids`.
Destructure into whatever names match your topology's slot order (defined by the
`entity_slots = EntityGroup(...)` declaration on the Python `System` class).

---

## Smooth contact / hard-stop helper

A bare `max(0, -pos)` stop force has a C0 kink that causes thousands of tiny
rejected steps. Use this C1-smooth ramp instead:

```julia
const STOP_DELTA = 1e-6   # 1 µm — matches Python _STOP_DELTA

function soft_pen(x::T) where T <: Real
    x <= 0.0        && return zero(T)
    x >= STOP_DELTA && return x - 0.5 * STOP_DELTA
    return 0.5 * x * x / STOP_DELTA
end
```

Piston with both-end stops:

```julia
pen_close    = soft_pen(-pos)
pen_open     = soft_pen(pos - max_travel)
alpha_close  = clamp(-pos / STOP_DELTA, zero(S), one(S))
alpha_open   = clamp((pos - max_travel) / STOP_DELTA, zero(S), one(S))
v_damp_close = max(zero(S), -vel) * alpha_close
v_damp_open  = max(zero(S),  vel) * alpha_open
F_stop = (k_stop * pen_close + c_stop * v_damp_close
         - k_stop * pen_open  - c_stop * v_damp_open)
```

---

## Isentropic orifice flow

Standard compressible orifice helper (from the fluid examples):

```julia
function orifice_mdot(
    P_up::T, P_dn::T, T_up::Float64,
    R::Float64, Cd::Float64, A::Real, gamma::Float64,
) where T <: Real
    (P_up <= 0.0 || A <= 0.0) && return zero(T)

    beta      = max(zero(T), P_dn) / P_up
    beta_crit = (2.0 / (gamma + 1.0))^(gamma / (gamma - 1.0))

    if beta <= beta_crit
        choke_exp = (gamma + 1.0) / (2.0 * (gamma - 1.0))
        return Cd * A * P_up * sqrt(gamma / (R * T_up)) *
               (2.0 / (gamma + 1.0))^choke_exp
    else
        arg = beta^(2.0 / gamma) - beta^((gamma + 1.0) / gamma)
        return Cd * A * P_up * sqrt(
            max(zero(T), 2.0 * gamma / ((gamma - 1.0) * R * T_up) * arg))
    end
end
```

`P_up`, `P_dn`, and state-derived areas may be Dual numbers; scalar parameters
from `p` are always `Float64`.

---

## Helper function type signatures

Any helper that receives state values must be generic in `T`:

```julia
# Correct — works for Float64 and Dual
function my_helper(x::T) where T <: Real
    x > 0.0 && return x * x
    return zero(T)        # zero(T), not 0.0
end

# Wrong — crashes during stiff Jacobian evaluation
function my_helper(x::Float64)::Float64
    return x > 0.0 ? x * x : 0.0
end
```

---

## Calling from Python

```python
from numen.bridge.runtime import JuliaBackend, JuliaServerBackend

# Single solve
result = JuliaBackend(
    julia_file = "dynamics.jl",
    method     = "Tsit5",         # or "Rodas5P" for stiff systems
    rtol       = 1e-8,
    atol       = 1e-10,
).solve(spec, tspan=(0.0, 1.0))

# Repeated solves — pays JIT cost once
with JuliaServerBackend(
    julia_file    = "dynamics.jl",
    method        = "Tsit5",
    rtol          = 1e-8,
    atol          = 1e-10,
    n_save_points = 2000,
) as srv:
    for p_val in param_sweep:
        result = srv.solve(compile_spec(make_world(p_val)), tspan)
```

See [Backends](guide/backends.md) for `JuliaServerPool` (parallel sweeps).

---

## Framework types reference

Defined in `julia/src/types.jl`; available via `import Main:` in every dynamics file.

### `CompiledSpec`

```julia
struct CompiledSpec
    state_size        :: Int
    param_size        :: Int
    state_index_map   :: Dict{String, Vector{Int}}   # key → [start, stop] (0-based)
    param_index_map   :: Dict{String, Vector{Int}}
    discrete_dts      :: Vector{Float64}
    x0                :: Vector{Float64}
    p                 :: Vector{Float64}
    differential_mask :: Vector{Float64}             # 1.0 = ODE slot, 0.0 = algebraic
    systems           :: Vector{CompiledSystemSpec}
    callbacks         :: Vector{CompiledCallbackSpec}
    jac_sparsity_rows :: Vector{Int}                 # COO sparsity (0-based)
    jac_sparsity_cols :: Vector{Int}
end
```

### `CompiledSystemSpec`

```julia
struct CompiledSystemSpec
    dynamics_fn :: String            # "Module.function_name!"
    entity_ids  :: Vector{String}    # flat; stride = group_size
    group_size  :: Int
end
```

### Index helper source

```julia
# 0-based Python [start, stop) → 1-based Julia start:stop
state_range(spec, key) = let e = spec.state_index_map[key]; (e[1]+1):e[2] end
param_range(spec, key) = let e = spec.param_index_map[key]; (e[1]+1):e[2] end

state_idx(spec, key) = first(state_range(spec, key))
param_idx(spec, key) = first(param_range(spec, key))
```
