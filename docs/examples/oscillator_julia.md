# Oscillator — Julia dynamics

Source: [`src/numen/examples/oscillator/dynamics.jl`](https://github.com/TalbotKnighton/numen/blob/main/src/numen/examples/oscillator/dynamics.jl)

The oscillator is the simplest possible Julia dynamics file — one module, one
function, one entity group of size 1.

---

## Full source

```julia
module OscillatorDynamics

import Main: CompiledSpec, CompiledSystemSpec, state_idx, param_idx, groups

"""
    oscillator_dynamics!(dx, x, p, t, spec, sys)

Harmonic oscillator: ẋ = v,  v̇ = -ω²x - 2ζωv.
"""
function oscillator_dynamics!(
    dx  :: AbstractVector{T},
    x   :: AbstractVector{S},
    p   :: Vector{Float64},
    t   :: Real,
    spec:: CompiledSpec,
    sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    for (eid,) in groups(sys)
        pos_idx     = state_idx(spec, "$eid.oscillator.position")
        vel_idx     = state_idx(spec, "$eid.oscillator.velocity")
        omega_idx   = param_idx(spec, "$eid.oscillator.omega")
        damping_idx = param_idx(spec, "$eid.oscillator.damping")

        pos     = x[pos_idx]
        vel     = x[vel_idx]
        omega   = p[omega_idx]
        damping = p[damping_idx]

        dx[pos_idx] += vel
        dx[vel_idx] += -omega^2 * pos - 2 * damping * omega * vel
    end
end

end  # module OscillatorDynamics
```

---

## Key points

**`for (eid,) in groups(sys)`** — `groups(sys)` yields tuples of length
`sys.group_size` (here, 1). The trailing comma in `(eid,)` destructures the
single element. This mirrors the Python `for (entity_id,) in system.entity_groups:`
pattern.

**Reading state vs params** — `x[state_idx(...)]` for integrated fields
(position, velocity); `p[param_idx(...)]` for parameter fields (omega, damping).

**`+=` accumulation** — even though only one system writes to these slots in
this example, using `+=` is the correct pattern throughout. If a second system
also touched velocity, both contributions would add correctly.

---

## Connecting to Python

```python
# dynamics.py
class OscillatorSystem(System):
    component_types: ClassVar[tuple[type, ...]] = (OscillatorComponent,)
    python_fn:       ClassVar[DynamicsFn]       = staticmethod(oscillator_dynamics)
    kind:            Literal["oscillator"]       = "oscillator"
    dynamics_fn:     str = "OscillatorDynamics.oscillator_dynamics!"
```

```python
# run.py
from numen.bridge.runtime import JuliaBackend

result = JuliaBackend(
    julia_file = "dynamics.jl",
    method     = "Tsit5",
    rtol       = 1e-8,
    atol       = 1e-10,
).solve(spec, tspan=(0.0, 5.0))
```
