module NLOscillatorDynamics

import Main: CompiledSpec, CompiledSystemSpec, state_idx, param_idx

"""
    nl_oscillator_dynamics!(dx, x, p, t, spec, sys)

Nonlinear oscillator: ẋ = v,  v̇ = -(c0 + c1·x²)·v - ω²·x.

Mirrors the Python ``nl_oscillator_dynamics`` function.
c0 is the linear damping baseline; c1 scales damping with displacement squared.

Keys use the full path ``entity_id.nl_oscillator.field_name``.
"""
function nl_oscillator_dynamics!(
    dx  :: AbstractVector{T},
    x   :: AbstractVector{S},
    p   :: Vector{Float64},
    t   :: Real,
    spec:: CompiledSpec,
    sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    gs = sys.group_size   # = 1 for single-entity systems
    for i in 1:gs:length(sys.entity_ids)
        eid = sys.entity_ids[i]

        pos_idx   = state_idx(spec, eid * ".nl_oscillator.position")
        vel_idx   = state_idx(spec, eid * ".nl_oscillator.velocity")
        omega_idx = param_idx(spec, eid * ".nl_oscillator.omega")
        c0_idx    = param_idx(spec, eid * ".nl_oscillator.c0")
        c1_idx    = param_idx(spec, eid * ".nl_oscillator.c1")

        pos   = x[pos_idx]
        vel   = x[vel_idx]
        omega = p[omega_idx]
        c0    = p[c0_idx]
        c1    = p[c1_idx]

        effective_damping = c0 + c1 * pos^2

        dx[pos_idx] += vel
        dx[vel_idx] += -effective_damping * vel - omega^2 * pos
    end
end

end  # module NLOscillatorDynamics
