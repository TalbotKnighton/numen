"""
NumenCharacterization — built-in Julia dynamics for the characterization framework.

Loaded automatically by server.jl and runner.jl so that any spec produced by
inject_excitation(), inject_chirp_excitation(), or inject_table_excitation()
works without the user having to define these functions themselves.

Three functions are provided:
  NumenCharacterization.excitation_dynamics!       — sinusoidal forcing F(t) = amp·sin(2πft) + dc
  NumenCharacterization.chirp_dynamics!            — frequency-swept chirp forcing
  NumenCharacterization.table_excitation_dynamics! — pre-computed table lookup (random vibe, replay)
"""
module NumenCharacterization

import Main: CompiledSpec, CompiledSystemSpec, param_idx, param_range

"""
    excitation_dynamics!(dx, x, p, t, spec, sys)

Adds F(t) = amp·sin(2π·freq·t) + dc to the target state derivative.

Parameters stored under the synthetic entity id (e.g. `_exc_osc_force`):
  .amp        — sine amplitude
  .freq       — frequency [Hz]
  .dc         — DC offset
  .target_idx — 0-based index of the state slot to write (stored as Float64)
"""
function excitation_dynamics!(
    dx  :: AbstractVector{T},
    x   :: AbstractVector{S},
    p   :: Vector{Float64},
    t   :: Real,
    spec:: CompiledSpec,
    sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    eid   = sys.entity_ids[1]
    amp   = p[param_idx(spec, eid * ".amp")]
    freq  = p[param_idx(spec, eid * ".freq")]
    dc    = p[param_idx(spec, eid * ".dc")]
    tgt_i = Int(round(p[param_idx(spec, eid * ".target_idx")])) + 1  # 0-based → 1-based

    dx[tgt_i] += amp * sin(2π * freq * t) + dc
end


"""
    chirp_dynamics!(dx, x, p, t, spec, sys)

Adds a frequency-swept forcing F(t) = amp·sin(φ(t)) + dc to the target state derivative.

Parameters stored under the synthetic chirp entity id (e.g. `_chirp_osc_force`):
  .amp             — amplitude
  .f_start         — start frequency [Hz]
  .f_end           — end frequency [Hz]
  .duration        — sweep duration [s]
  .dc              — DC offset
  .target_idx      — 0-based index of the state slot to write (stored as Float64)
  .chirp_type_flag — 0.0 = log sweep, 1.0 = linear sweep

Log sweep:    φ(t) = 2π · f_s · T/k · (exp(k·t/T) − 1)  where k = ln(f_e/f_s)
Linear sweep: φ(t) = 2π · (f_s·t + (f_e−f_s)·t²/(2T))
"""
function chirp_dynamics!(
    dx  :: AbstractVector{T},
    x   :: AbstractVector{S},
    p   :: Vector{Float64},
    t   :: Real,
    spec:: CompiledSpec,
    sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    eid    = sys.entity_ids[1]
    amp    = p[param_idx(spec, eid * ".amp")]
    f_s    = p[param_idx(spec, eid * ".f_start")]
    f_e    = p[param_idx(spec, eid * ".f_end")]
    dur    = p[param_idx(spec, eid * ".duration")]
    dc     = p[param_idx(spec, eid * ".dc")]
    tgt_i  = Int(round(p[param_idx(spec, eid * ".target_idx")])) + 1
    is_lin = p[param_idx(spec, eid * ".chirp_type_flag")] > 0.5

    if is_lin
        phase = 2π * (f_s * t + (f_e - f_s) * t^2 / (2 * dur))
    else
        k     = log(f_e / f_s)
        phase = 2π * f_s * dur / k * (exp(k * t / dur) - 1.0)
    end

    dx[tgt_i] += amp * sin(phase) + dc
end

"""
    table_excitation_dynamics!(dx, x, p, t, spec, sys)

Adds a pre-computed forcing signal to the target state derivative via linear
table interpolation.  The signal was generated on the Python side (PSD → irfft
or direct file replay) and stored in the parameter vector before the solve.

Parameters stored under the synthetic table entity id (e.g. `_table_osc_force`):
  .n_samples   — number of signal samples (stored as Float64; cast to Int internally)
  .t_start     — time of first sample [s]
  .dt_sig      — uniform sample period [s]
  .dc          — DC offset added to the interpolated value
  .target_idx  — 0-based index of the state slot to write (stored as Float64)
  .signal      — contiguous block of N Float64 values  (param slice, not a scalar)

The signal is clamped at both ends (no extrapolation).
"""
function table_excitation_dynamics!(
    dx  :: AbstractVector{T},
    x   :: AbstractVector{S},
    p   :: Vector{Float64},
    t   :: Real,
    spec:: CompiledSpec,
    sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    eid   = sys.entity_ids[1]

    n     = Int(round(p[param_idx(spec, eid * ".n_samples")]))
    t0    = p[param_idx(spec, eid * ".t_start")]
    dt    = p[param_idx(spec, eid * ".dt_sig")]
    dc    = p[param_idx(spec, eid * ".dc")]
    tgt_i = Int(round(p[param_idx(spec, eid * ".target_idx")])) + 1  # 0-based → 1-based

    # Signal slice — param_range returns a Julia UnitRange{Int} (1-based)
    sig_r = param_range(spec, eid * ".signal")

    # Clamp t to table bounds and compute interpolation indices
    tc    = clamp(Float64(t) - t0, 0.0, (n - 1) * dt)
    frac  = tc / dt
    k     = clamp(Int(floor(frac)) + 1, 1, n)   # lower sample index (1-based within slice)
    k1    = clamp(k + 1, 1, n)                    # upper sample index
    alpha = frac - floor(frac)

    F = p[sig_r[k]] * (1.0 - alpha) + p[sig_r[k1]] * alpha + dc
    dx[tgt_i] += F
end

end  # module NumenCharacterization
