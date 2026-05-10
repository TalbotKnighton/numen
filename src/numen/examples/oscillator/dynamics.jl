module OscillatorDynamics

import Main: CompiledSpec, CompiledSystemSpec, groups,
             get_state, get_param, add_deriv!

"""
    oscillator_dynamics!(dx, x, p, t, spec, sys)

Harmonic oscillator: ẋ = v,  v̇ = -ω²x - 2ζωv.

Mirrors the Python ``oscillator_dynamics`` function.  Signature is identical
across Python and Julia: ``(dx, x, p, t, spec, sys)``.
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
