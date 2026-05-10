# Coupled Spring — Julia dynamics

Source: [`src/numen/examples/coupled_spring/dynamics.jl`](https://github.com/TalbotKnighton/numen/blob/main/src/numen/examples/coupled_spring/dynamics.jl)

This example shows the **multi-entity topology pattern**: two systems with
different `group_size` values sharing the same entity slots.

Model: three masses (`m1`, `m2`, `m3`) connected by two springs (`s1`, `s2`).

---

## Full source

```julia
module SpringDynamics

import Main: CompiledSpec, CompiledSystemSpec, groups,
             get_state, get_param, add_deriv!

# -------------------------------------------------------------------
# MassKinematicsSystem  (group_size = 1)
# -------------------------------------------------------------------

"""
    mass_kinematics_dynamics!(dx, x, p, t, spec, sys)

Position kinematics: ẋ = v for every mass entity.
"""
function mass_kinematics_dynamics!(
    dx  :: AbstractVector{T},
    x   :: AbstractVector{S},
    p   :: Vector{Float64},
    t   :: Real,
    spec:: CompiledSpec,
    sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    for (eid,) in groups(sys)
        vel = get_state(spec, x, eid, "mass.velocity")
        add_deriv!(spec, dx, eid, "mass.position", vel)
    end
end

# -------------------------------------------------------------------
# SpringForceSystem  (group_size = 3)
# -------------------------------------------------------------------

"""
    spring_force_dynamics!(dx, x, p, t, spec, sys)

Hooke's law spring forces.
Group stride: [mass_a, spring, mass_b]  (group_size = 3).
Forces accumulate with += so multiple springs sharing a mass are correct.
"""
function spring_force_dynamics!(
    dx  :: AbstractVector{T},
    x   :: AbstractVector{S},
    p   :: Vector{Float64},
    t   :: Real,
    spec:: CompiledSpec,
    sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    for (id_a, id_s, id_b) in groups(sys)
        pos_a  = get_state(spec, x, id_a, "mass.position")
        pos_b  = get_state(spec, x, id_b, "mass.position")
        mass_a = get_param(spec, p, id_a, "mass.mass")
        mass_b = get_param(spec, p, id_b, "mass.mass")
        k      = get_param(spec, p, id_s, "spring.k")
        rest   = get_param(spec, p, id_s, "spring.rest_length")

        stretch = (pos_b - pos_a) - rest
        force   = k * stretch

        add_deriv!(spec, dx, id_a, "mass.velocity",  force / mass_a)
        add_deriv!(spec, dx, id_b, "mass.velocity", -force / mass_b)
    end
end

end  # module SpringDynamics
```

---

## Key points

**Tuple destructuring with `groups(sys)`** — the spring system declares
`entity_slots = EntityGroup(MassComponent, SpringComponent, MassComponent)` on
the Python side, so each group has three entries in slot order. Naming them
`(id_a, id_s, id_b)` makes the topology explicit at the loop header.

**Two systems with different `group_size`** — `mass_kinematics_dynamics!` uses
`group_size=1` (each mass independent); `spring_force_dynamics!` uses
`group_size=3`. The `for` headers show this at a glance.

**Force accumulation at shared masses** — mass `m2` appears in *both* spring
groups (`[m1, s1, m2]` and `[m2, s2, m3]`). Because `add_deriv!` uses `+=`,
both springs contribute to `m2`'s velocity correctly.

---

## Connecting to Python

```python
class MassKinematicsSystem(System):
    component_types: ClassVar = (MassComponent,)
    python_fn:       ClassVar = staticmethod(mass_kinematics_dynamics)
    kind:            Literal["mass_kinematics"] = "mass_kinematics"
    dynamics_fn:     str = "SpringDynamics.mass_kinematics_dynamics!"

class SpringForceSystem(System):
    component_types: ClassVar = ()             # no auto-population
    entity_slots:    ClassVar = EntityGroup(
        MassComponent, SpringComponent, MassComponent
    )
    python_fn:       ClassVar = staticmethod(spring_force_dynamics)
    kind:            Literal["spring_force"] = "spring_force"
    dynamics_fn:     str = "SpringDynamics.spring_force_dynamics!"
```
