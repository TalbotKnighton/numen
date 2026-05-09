module OscillatorDynamics

# CompiledSpec, CompiledSystemSpec, state_idx, param_idx are available via Main.Numen
import Main: CompiledSpec, CompiledSystemSpec, state_idx, param_idx

"""
    oscillator_dynamics!(dx, x, p, t, spec, sys)

Harmonic oscillator: ẋ = v,  v̇ = -ω²x - 2ζωv.

Mirrors the Python ``oscillator_dynamics`` function.  Signature is identical
across Python and Julia: ``(dx, x, p, t, spec, sys)``.

Dict lookups (``state_idx`` / ``param_idx``) happen once per entity per RHS
call; Julia's JIT compiles away the overhead after the first invocation.
"""
function oscillator_dynamics!(
    dx  :: AbstractVector{T},
    x   :: AbstractVector{S},
    p   :: Vector{Float64},
    t   :: Real,
    spec:: CompiledSpec,
    sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    gs = sys.group_size   # = 1; each entity is its own group
    for i in 1:gs:length(sys.entity_ids)
        eid = sys.entity_ids[i]

        pos_idx     = state_idx(spec, eid * ".position")
        vel_idx     = state_idx(spec, eid * ".velocity")
        omega_idx   = param_idx(spec, eid * ".omega")
        damping_idx = param_idx(spec, eid * ".damping")

        pos     = x[pos_idx]
        vel     = x[vel_idx]
        omega   = p[omega_idx]
        damping = p[damping_idx]

        dx[pos_idx] += vel
        dx[vel_idx] += -omega^2 * pos - 2 * damping * omega * vel
    end
end

end  # module OscillatorDynamics
