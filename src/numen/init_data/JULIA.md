# Numen — Julia dynamics conventions

This file documents the Julia API and performance trade-offs. It's a context
file for AI assistants working on Numen models — when writing or refactoring
`dynamics.jl`, use these conventions.

---

## Two API levels

Numen exposes Julia helpers at two levels. Pick one based on the situation.

### High-level (default — readable)

Use these in 95% of dynamics functions:

```julia
get_state(spec, x, eid, "kind.field")              # → scalar
get_param(spec, p, eid, "kind.field")              # → Float64
get_state_vec(spec, x, eid, "kind.field")          # → SubArray (vector fields)
get_param_vec(spec, p, eid, "kind.field")          # → SubArray
add_deriv!(spec, dx, eid, "kind.field", value)     # accumulate into dx
```

Reads like Python's `spec.view(eid, ComponentType, x, p).field`. Each call does
a single `Dict{String, ...}` lookup internally.

### Low-level (performance-tuned)

When you need to minimise dictionary lookups inside a hot inner loop, drop down
to the raw index helpers:

```julia
state_idx(spec, key)        # → Int (1-based)
param_idx(spec, key)        # → Int
state_range(spec, key)      # → UnitRange{Int}  (vector fields)
param_range(spec, key)      # → UnitRange{Int}
```

Cache the index in a local once, then reuse it for both reads and writes:

```julia
i_pos = state_idx(spec, "$eid.oscillator.position")
i_vel = state_idx(spec, "$eid.oscillator.velocity")
pos = x[i_pos];   vel = x[i_vel]
dx[i_pos] += vel
dx[i_vel] += -omega^2 * pos - 2 * damping * omega * vel
```

This is the form the example dynamics files used historically; it's about 2×
faster per state field per RHS call than the high-level form.

---

## When to optimise

**Default to the high-level helpers.** The performance penalty is dwarfed by
ODE solver cost on any realistic problem. For a typical small-to-medium ODE
solved with `Tsit5` or `Dopri5`, the difference is microseconds — irrelevant.

**Drop to the low-level form when**:

- You're solving a **stiff** problem with `Rodas5P` / `Rosenbrock23` / `FBDF`,
  AND
- The Jacobian sweep dominates the total runtime, AND
- You've measured a meaningful speedup with `@btime` from `BenchmarkTools.jl`

Stiff Jacobian evaluation calls dynamics `O(state_size × color_count)` times
per step (where color_count is typically 4–16 from the sparse coloring). On a
1000-state model this can push the dictionary lookups into the milliseconds
range per step, which is when caching starts to pay off.

**Don't pre-optimise.** Write the readable form first, profile if it's slow,
and only then drop down.

---

## Typical patterns

### Single-entity system (group_size = 1)

```julia
function gravity_dynamics!(dx, x, p, t, spec, sys) where {T <: Real, S <: Real}
    for (eid,) in groups(sys)
        vel = get_state(spec, x, eid, "ball.velocity")
        add_deriv!(spec, dx, eid, "ball.position", vel)
        add_deriv!(spec, dx, eid, "ball.velocity", -9.81)
    end
end
```

Note the **trailing comma** in `(eid,)` — Julia tuple destructuring requires it
for single-element tuples.

### Multi-entity coupled system (group_size = 3)

```julia
function spring_force_dynamics!(dx, x, p, t, spec, sys) where {T <: Real, S <: Real}
    for (mass_a, spring, mass_b) in groups(sys)
        pos_a = get_state(spec, x, mass_a, "mass.position")
        pos_b = get_state(spec, x, mass_b, "mass.position")
        k     = get_param(spec, p, spring, "spring.k")
        rest  = get_param(spec, p, spring, "spring.rest_length")

        force = k * ((pos_b - pos_a) - rest)
        m_a   = get_param(spec, p, mass_a, "mass.mass")
        m_b   = get_param(spec, p, mass_b, "mass.mass")

        add_deriv!(spec, dx, mass_a, "mass.velocity",  force / m_a)
        add_deriv!(spec, dx, mass_b, "mass.velocity", -force / m_b)
    end
end
```

The destructured names (`mass_a`, `spring`, `mass_b`) match the `entity_slots`
declaration on the Python `System` class — `EntityGroup(MassComponent,
SpringComponent, MassComponent)`.

### Vector field (size > 1)

```julia
# Read all 8 frequencies at once
freqs = get_param_vec(spec, p, eid, "vibe.frequencies")
amps  = get_param_vec(spec, p, eid, "vibe.amplitudes")

# Compute aggregate force — sum of sinusoids
F = zero(T)
for k in 1:length(freqs)
    F += amps[k] * sin(2π * freqs[k] * t)
end
```

`get_state_vec` / `get_param_vec` return zero-allocation `SubArray` views.

---

## Function signature

Every dynamics function must use this exact signature:

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

The two type parameters `{T, S}` cover both `Float64` (normal solve steps) and
`ForwardDiff.Dual` (Jacobian evaluation under stiff solvers). The high-level
helpers preserve this — `get_state` returns `eltype(x)`, so the result is
correctly typed for whichever path the solver is on.

---

## String interpolation, not `*`

Julia's `*` operator is string concatenation, but it's confusing if you're not
a Julia native. **Always use `$` interpolation**:

```julia
# ✓ Correct
P = x[state_idx(spec, "$eid.control_volume.pressure")]

# ✗ Avoid — confusing to Python/C++/JS readers
P = x[state_idx(spec, eid * ".control_volume.pressure")]
```

The high-level helpers (`get_state`, `add_deriv!`, etc.) build the key
internally so you never type `$eid.` manually.

---

## Smooth contact

A bare `max(0, -pos)` stop has a C0 kink that wrecks adaptive ODE solvers
(thousands of rejected steps). Always use the C1-smooth ramp:

```julia
const STOP_DELTA = 1e-6   # 1 µm — matches Python _STOP_DELTA

function soft_pen(x::T) where T <: Real
    x <= 0.0        && return zero(T)
    x >= STOP_DELTA && return x - 0.5 * STOP_DELTA
    return 0.5 * x * x / STOP_DELTA
end
```

Return `zero(T)` (not `0.0`) from branches that don't touch `x` — keeps the
return type consistent across `Float64` / `Dual` paths.
