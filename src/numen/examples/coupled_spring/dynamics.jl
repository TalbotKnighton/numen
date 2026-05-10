module SpringDynamics

import Main: CompiledSpec, CompiledSystemSpec, state_idx, param_idx

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
    gs = sys.group_size   # = 1
    for i in 1:gs:length(sys.entity_ids)
        eid     = sys.entity_ids[i]
        pos_idx = state_idx(spec, eid * ".mass.position")
        vel_idx = state_idx(spec, eid * ".mass.velocity")
        dx[pos_idx] += x[vel_idx]
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
    gs = sys.group_size   # = 3: [mass_a, spring, mass_b]
    for i in 1:gs:length(sys.entity_ids)
        id_a = sys.entity_ids[i]
        id_s = sys.entity_ids[i + 1]
        id_b = sys.entity_ids[i + 2]

        pos_a    = state_idx(spec, id_a * ".mass.position")
        pos_b    = state_idx(spec, id_b * ".mass.position")
        vel_a    = state_idx(spec, id_a * ".mass.velocity")
        vel_b    = state_idx(spec, id_b * ".mass.velocity")
        mass_a   = param_idx(spec, id_a * ".mass.mass")
        mass_b   = param_idx(spec, id_b * ".mass.mass")
        k_idx    = param_idx(spec, id_s * ".spring.k")
        rest_idx = param_idx(spec, id_s * ".spring.rest_length")

        stretch = (x[pos_b] - x[pos_a]) - p[rest_idx]
        force   = p[k_idx] * stretch

        dx[vel_a] +=  force / p[mass_a]
        dx[vel_b] += -force / p[mass_b]
    end
end

end  # module SpringDynamics
