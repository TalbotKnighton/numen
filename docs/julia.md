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
import Main: CompiledSpec, CompiledSystemSpec, groups,
             get_state, get_param, add_deriv!

function gravity_dynamics!(
    dx :: AbstractVector{T}, x :: AbstractVector{S}, p :: Vector{Float64},
    t  :: Real, spec :: CompiledSpec, sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    for (eid,) in groups(sys)
        vel = get_state(spec, x, eid, "ball.velocity")
        add_deriv!(spec, dx, eid, "ball.position", vel)
        add_deriv!(spec, dx, eid, "ball.velocity", -9.81)
    end
end

end  # module MyDynamics
```

In Python, set `dynamics_fn = "MyDynamics.gravity_dynamics!"` on the `System`.

---

## Two API levels

Numen provides two layers of helpers. Pick one based on the situation.

### High-level (default — readable)

These take the entity ID and a `"kind.field"` path; they read like Python's
`spec.view(eid, ComponentType, x, p).field`.

| Function | Description |
|---|---|
| `get_state(spec, x, eid, "kind.field")` | Read a scalar state value |
| `get_param(spec, p, eid, "kind.field")` | Read a scalar parameter |
| `get_state_vec(spec, x, eid, "kind.field")` | Read a vector state field as a `SubArray` |
| `get_param_vec(spec, p, eid, "kind.field")` | Read a vector parameter field |
| `add_deriv!(spec, dx, eid, "kind.field", value)` | Accumulate `value` into `dx` |
| `groups(sys)` | Iterate entity groups for tuple destructuring |

```julia
# group_size = 3 — coupled triplet [cv_a, orifice, cv_b]
for (cv_a, orifice, cv_b) in groups(sys)
    P_a = get_state(spec, x, cv_a,    "control_volume.pressure")
    A   = get_param(spec, p, orifice, "orifice.area")
    P_b = get_state(spec, x, cv_b,    "control_volume.pressure")
    # ...
    add_deriv!(spec, dx, cv_a, "control_volume.pressure", -(R_a*T_a/V_a) * mdot)
end
```

### Low-level (performance-tuned)

When you need to minimise dictionary lookups inside a hot inner loop, drop down
to the raw index helpers and cache the index for reuse:

| Function | Description |
|---|---|
| `state_idx(spec, key)` | First 1-based Julia index for a state field |
| `param_idx(spec, key)` | First 1-based Julia index for a parameter field |
| `state_range(spec, key)` | `UnitRange{Int}` — full slice for vector fields (`size=N`) |
| `param_range(spec, key)` | `UnitRange{Int}` — full slice for vector parameter fields |

Keys use the full path `"entity_id.component_kind.field_name"`.

```julia
i_pos = state_idx(spec, "$eid.oscillator.position")
i_vel = state_idx(spec, "$eid.oscillator.velocity")
pos = x[i_pos];  vel = x[i_vel]
dx[i_pos] += vel
dx[i_vel] += -omega^2 * pos - 2 * damping * omega * vel
```

### Performance trade-off

The high-level helpers do a `Dict{String, ...}` lookup on every call. Caching
the index (low-level form) and reusing it for read+write is roughly 2× faster
per state field per RHS call.

**Default to the high-level helpers** — the cost is microseconds and is dwarfed
by ODE solver work on any realistic problem. Drop to the low-level form only
when:

1. You're solving a stiff problem (Rodas5P, Rosenbrock23, FBDF) where Jacobian
   evaluation calls dynamics `O(state_size × color_count)` times per step
2. You've measured a meaningful speedup with `BenchmarkTools.@btime`

Don't pre-optimise. Write the readable form first, profile if it's slow.

---

## Iterating entity groups

`groups(sys)` mirrors the Python `for entity_group in system.entity_groups:`
pattern. Each iteration yields a slice of `sys.entity_ids` of length
`sys.group_size`, suitable for tuple destructuring:

```julia
# group_size = 1 — one entity per group (note the trailing comma!)
for (eid,) in groups(sys)
    pos = get_state(spec, x, eid, "oscillator.position")
    # ...
end

# group_size = 3 — three entities per group
for (cv_a, orifice, cv_b) in groups(sys)
    # ...
end
```

The destructured names should match the slot order declared in the Python
`System` via `entity_slots = EntityGroup(...)`.

Internally, `groups(sys) === Iterators.partition(sys.entity_ids, sys.group_size)`
— a zero-allocation iterator over fixed-size slices.

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
