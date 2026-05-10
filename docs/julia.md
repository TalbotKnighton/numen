# Julia Reference

Numen uses Julia (via OrdinaryDiffEq.jl) as its fastest solver backend.
Each Python `System` has a corresponding Julia function in a `.jl` file.

## Dynamics function signature

```julia
# dynamics.jl
module MyDynamics
import Main: CompiledSpec, CompiledSystemSpec, state_idx, param_idx

function gravity_dynamics!(
    dx :: AbstractVector{T},
    x  :: AbstractVector{S},
    p  :: Vector{Float64},
    t  :: Real,
    spec :: CompiledSpec,
    sys  :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    for id_ball in sys.entity_ids
        i_pos = state_idx(spec, id_ball * ".ball.position")
        i_vel = state_idx(spec, id_ball * ".ball.velocity")
        dx[i_pos] += x[i_vel]
        dx[i_vel] += -9.81
    end
end

end  # module MyDynamics
```

The `{T <: Real, S <: Real}` signature is required for stiff solvers (Rodas5P):
during Jacobian evaluation, OrdinaryDiffEq calls dynamics with `x::Vector{Dual}`.
Both type parameters must be present so helper functions can correctly type their
return values.

Use `t :: Real` (not `Float64`) to cover both normal Float64 calls and any
potential ForwardDiff Dual paths.

## Index helpers

```julia
state_idx(spec, "entity_id.component_kind.field_name")  # Int (1-based)
param_idx(spec, "entity_id.component_kind.field_name")  # Int (1-based)
state_range(spec, "entity_id.component_kind.field_name")  # UnitRange{Int}
param_range(spec, "entity_id.component_kind.field_name")  # UnitRange{Int}
```

For multi-slot systems, entity IDs are stored flat in `sys.entity_ids` with
`sys.group_size` entities per group:

```julia
function spring_force!(dx, x, p, t, spec, sys)
    gs = sys.group_size   # = 3 for (MassComponent, SpringComponent, MassComponent)
    for i in 1:gs:length(sys.entity_ids)
        id_a = sys.entity_ids[i]
        id_s = sys.entity_ids[i+1]
        id_b = sys.entity_ids[i+2]
        # ...
    end
end
```

## Scalar helper functions

Helper functions called from dynamics must be generic over `T <: Real` so they work
with both `Float64` (normal calls) and `ForwardDiff.Dual` (Jacobian calls):

```julia
function soft_pen(x::T) where T <: Real
    x <= 0.0 && return zero(T)
    x >= STOP_DELTA && return x - T(0.5 * STOP_DELTA)
    return T(0.5) * x * x / STOP_DELTA
end
```

## Smooth hard stop (C1 ramp)

```julia
const STOP_DELTA = 1e-6   # 1 µm

function soft_pen(x::T)::T where T <: Real
    x <= 0.0 && return zero(T)
    x >= STOP_DELTA && return x - T(0.5 * STOP_DELTA)
    return T(0.5) * x * x / STOP_DELTA
end

# Velocity damping: blend in over the same 1 µm
alpha  = clamp(-pos / STOP_DELTA, 0.0, 1.0)
v_damp = max(0.0, -vel) * alpha
F_stop = k_stop * soft_pen(-pos) + c_stop * v_damp
```

## Calling from Python

```python
from numen.bridge.runtime import JuliaBackend

backend = JuliaBackend(
    julia_file="dynamics.jl",
    method="Tsit5",
    rtol=1e-6,
    atol=1e-8,
)
result = backend.solve(spec, tspan=(0.0, 5.0))
```

For repeated solves in the same session, use `JuliaServerBackend` (persistent process):

```python
from numen.bridge.server_backend import JuliaServerBackend

backend = JuliaServerBackend(
    julia_file="dynamics.jl",
    method="Tsit5",
    rtol=1e-6,
    atol=1e-8,
)
result = backend.solve(spec, tspan=(0.0, 5.0))
```

## Stiff solvers

For stiff problems (multiple timescales, pressure/force at different scales), use
Rosenbrock-type implicit solvers:

| Method | Best for |
|---|---|
| `Tsit5` | Non-stiff ODEs (default) |
| `Dopri5` | Non-stiff ODEs with tight tolerances |
| `Vern7` | Non-stiff, high-accuracy |
| `Rodas5P` | Stiff ODEs and index-1 DAEs (recommended) |
| `FBDF` | Very stiff DAE systems |

Numen handles both the sparse Jacobian (`jac_prototype` from ECS sparsity pattern)
and the time gradient (`tgrad!` via central FD) automatically — you do not need
to implement these in `dynamics.jl`.

```python
backend = JuliaServerBackend(
    julia_file="dynamics.jl",
    method="Rodas5P",
    rtol=1e-6,
    atol=1e-8,
)
```

## Controller callbacks

Julia callbacks fire inside the OrdinaryDiffEq integrator via `PeriodicCallback`:

```julia
function pid!(integrator, spec, params)
    i_force = state_idx(spec, "actuator.actuator.force")
    i_angle = state_idx(spec, "sensor.sensor.angle")
    err = integrator.u[i_angle] - params["setpoint"]
    integrator.u[i_force] = params["kp"] * err
end
```

Reference the callback in Python with `julia_fn = "MyDyn.pid!"` on the `Callback` class.

## Why the subprocess bridge?

Numen communicates with Julia via subprocess JSON — not `juliacall` / `PythonCall`.

- **No dependency on `juliacall`** — Julia is a black box that reads JSON and writes JSON.
- **Full isolation** — Julia subprocess crash doesn't crash the Python process.
- **Simpler debugging** — Julia stderr is captured and routed to `numen.backend.julia*` logger.

The `JuliaServerBackend` keeps the subprocess alive between solves to amortise the
~6 s startup cost. The `JuliaServerPool` runs N parallel servers for parameter sweeps.
