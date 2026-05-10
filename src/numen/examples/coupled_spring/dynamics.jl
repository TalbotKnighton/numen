module SpringDynamics

import Main: CompiledSpec, CompiledSystemSpec, groups,
             get_state, get_param, add_deriv!

"""
    mass_kinematics_dynamics!(dx, x, p, t, spec, sys)

Position kinematics: ẋ = v for every mass entity.  ``group_size = 1``.
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

"""
    spring_force_dynamics!(dx, x, p, t, spec, sys)

Hooke's law spring forces.  ``group_size = 3``: each group is
``[mass_a, spring, mass_b]``.  Forces accumulate with ``+=`` so multiple
springs sharing a mass are correct.
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
