module NLOscillatorDynamics

import Main: CompiledSpec, CompiledSystemSpec, groups,
             get_state, get_param, add_deriv!

"""
    nl_oscillator_dynamics!(dx, x, p, t, spec, sys)

Nonlinear oscillator: ẋ = v,  v̇ = -(c0 + c1·x²)·v - ω²·x.

Mirrors the Python ``nl_oscillator_dynamics`` function.
c0 is the linear damping baseline; c1 scales damping with displacement squared.
"""
function nl_oscillator_dynamics!(
    dx  :: AbstractVector{T},
    x   :: AbstractVector{S},
    p   :: Vector{Float64},
    t   :: Real,
    spec:: CompiledSpec,
    sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    for (eid,) in groups(sys)
        pos   = get_state(spec, x, eid, "nl_oscillator.position")
        vel   = get_state(spec, x, eid, "nl_oscillator.velocity")
        omega = get_param(spec, p, eid, "nl_oscillator.omega")
        c0    = get_param(spec, p, eid, "nl_oscillator.c0")
        c1    = get_param(spec, p, eid, "nl_oscillator.c1")

        effective_damping = c0 + c1 * pos^2

        add_deriv!(spec, dx, eid, "nl_oscillator.position", vel)
        add_deriv!(spec, dx, eid, "nl_oscillator.velocity",
                   -effective_damping * vel - omega^2 * pos)
    end
end

end  # module NLOscillatorDynamics
