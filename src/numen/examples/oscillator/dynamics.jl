module OscillatorDynamics

# CompiledSpec, CompiledSystemSpec, state_idx, param_idx are available via Main.Numen
import Main: CompiledSpec, CompiledSystemSpec, state_idx, param_idx, groups

"""
    oscillator_dynamics!(dx, x, p, t, spec, sys)

Harmonic oscillator: ẋ = v,  v̇ = -ω²x - 2ζωv.

Mirrors the Python ``oscillator_dynamics`` function.  Signature is identical
across Python and Julia: ``(dx, x, p, t, spec, sys)``.

Keys use the full path ``entity_id.component_kind.field_name``.
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
        pos_idx     = state_idx(spec, eid * ".oscillator.position")
        vel_idx     = state_idx(spec, eid * ".oscillator.velocity")
        omega_idx   = param_idx(spec, eid * ".oscillator.omega")
        damping_idx = param_idx(spec, eid * ".oscillator.damping")

        pos     = x[pos_idx]
        vel     = x[vel_idx]
        omega   = p[omega_idx]
        damping = p[damping_idx]

        dx[pos_idx] += vel
        dx[vel_idx] += -omega^2 * pos - 2 * damping * omega * vel
    end
end

end  # module OscillatorDynamics
