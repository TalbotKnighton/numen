"""
NumenCharacterization — built-in Julia dynamics for the characterization framework.

Loaded automatically by server.jl and runner.jl so that any spec produced by
inject_excitation() or inject_chirp_excitation() works without the user having
to define these functions themselves.

Two functions are provided:
  NumenCharacterization.excitation_dynamics!  — sinusoidal forcing F(t) = amp·sin(2πft) + dc
  NumenCharacterization.chirp_dynamics!       — frequency-swept chirp forcing
"""
module NumenCharacterization

import Main: CompiledSpec, CompiledSystemSpec, param_idx

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
    dx  :: Vector{Float64},
    x   :: Vector{Float64},
    p   :: Vector{Float64},
    t   :: Float64,
    spec:: CompiledSpec,
    sys :: CompiledSystemSpec,
)
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
    dx  :: Vector{Float64},
    x   :: Vector{Float64},
    p   :: Vector{Float64},
    t   :: Float64,
    spec:: CompiledSpec,
    sys :: CompiledSystemSpec,
)
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

end  # module NumenCharacterization
