# Oscillator — Julia dynamics

Source: [`src/numen/examples/oscillator/dynamics.jl`](https://github.com/TalbotKnighton/numen/blob/main/src/numen/examples/oscillator/dynamics.jl)

The oscillator is the simplest possible Julia dynamics file — one module, one
function, one entity group of size 1.

---

## Full source

```julia
module OscillatorDynamics

import Main: CompiledSpec, CompiledSystemSpec, groups,
             get_state, get_param, add_deriv!

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
        pos     = get_state(spec, x, eid, "oscillator.position")
        vel     = get_state(spec, x, eid, "oscillator.velocity")
        omega   = get_param(spec, p, eid, "oscillator.omega")
        damping = get_param(spec, p, eid, "oscillator.damping")

        add_deriv!(spec, dx, eid, "oscillator.position", vel)
        add_deriv!(spec, dx, eid, "oscillator.velocity",
                   -omega^2 * pos - 2 * damping * omega * vel)
    end
end

end  # module OscillatorDynamics
```

---

## Key points

**`for (eid,) in groups(sys)`** — `groups(sys)` yields tuples of length
`sys.group_size` (here, 1). The trailing comma in `(eid,)` destructures the
single element. Mirrors Python's `for (entity_id,) in system.entity_groups:`.

**`get_state` / `get_param`** — read scalar values from `x` (state) or `p`
(parameters) using the `"kind.field"` path. The full key
`"$eid.oscillator.position"` is built internally.

**`add_deriv!`** — accumulates the value into the `dx` slot at
`"$eid.oscillator.position"`. Internally uses `+=`, so multiple systems can
write to the same slot and contributions add up correctly.

For details on the helper API, the lower-level `state_idx` / `param_idx`
form, and when to drop down for hot-loop performance, see the [Julia
Reference](../julia.md).

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
