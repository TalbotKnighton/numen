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
If a scale_by parameter index is set, F is **divided** by p[scale_by_idx]
before being added — used to convert force → acceleration via mass, etc.

Parameters stored under the synthetic entity id (e.g. `_exc_osc_force`):
  .amp          — sine amplitude
  .freq         — frequency [Hz]
  .dc           — DC offset
  .target_idx   — 0-based index of the state slot to write (stored as Float64)
  .scale_by_idx — 0-based index of a divisor in p, or -1.0 for no scaling (stored as Float64)
"""
function excitation_dynamics!(
    dx  :: AbstractVector{T},
    x   :: AbstractVector{S},
    p   :: Vector{Float64},
    t   :: Real,
    spec:: CompiledSpec,
    sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    eid       = sys.entity_ids[1]
    amp       = p[param_idx(spec, eid * ".amp")]
    freq      = p[param_idx(spec, eid * ".freq")]
    dc        = p[param_idx(spec, eid * ".dc")]
    tgt_i     = Int(round(p[param_idx(spec, eid * ".target_idx")])) + 1  # 0-based → 1-based
    scale_idx = Int(round(p[param_idx(spec, eid * ".scale_by_idx")]))    # -1 if none

    F = amp * sin(2π * freq * t) + dc
    if scale_idx >= 0
        F = F / p[scale_idx + 1]   # 0-based → 1-based
    end
    dx[tgt_i] += F
end


"""
    chirp_dynamics!(dx, x, p, t, spec, sys)

Adds a frequency-swept forcing F(t) = amp·sin(φ(t)) + dc to the target state derivative.
If a scale_by parameter index is set, F is **divided** by p[scale_by_idx]
before being added — used to convert force → acceleration via mass, etc.

Parameters stored under the synthetic chirp entity id (e.g. `_chirp_osc_force`):
  .amp             — amplitude
  .f_start         — start frequency [Hz]
  .f_end           — end frequency [Hz]
  .duration        — sweep duration [s]
  .dc              — DC offset
  .target_idx      — 0-based index of the state slot to write (stored as Float64)
  .chirp_type_flag — 0.0 = log sweep, 1.0 = linear sweep
  .scale_by_idx    — 0-based index of a divisor in p, or -1.0 for no scaling

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
    eid       = sys.entity_ids[1]
    amp       = p[param_idx(spec, eid * ".amp")]
    f_s       = p[param_idx(spec, eid * ".f_start")]
    f_e       = p[param_idx(spec, eid * ".f_end")]
    dur       = p[param_idx(spec, eid * ".duration")]
    dc        = p[param_idx(spec, eid * ".dc")]
    tgt_i     = Int(round(p[param_idx(spec, eid * ".target_idx")])) + 1
    is_lin    = p[param_idx(spec, eid * ".chirp_type_flag")] > 0.5
    scale_idx = Int(round(p[param_idx(spec, eid * ".scale_by_idx")]))

    if is_lin
        phase = 2π * (f_s * t + (f_e - f_s) * t^2 / (2 * dur))
    else
        k     = log(f_e / f_s)
        phase = 2π * f_s * dur / k * (exp(k * t / dur) - 1.0)
    end

    F = amp * sin(phase) + dc
    if scale_idx >= 0
        F = F / p[scale_idx + 1]
    end
    dx[tgt_i] += F
end

"""
    table_excitation_dynamics!(dx, x, p, t, spec, sys)

Adds a pre-computed forcing signal to the target state derivative via linear
table interpolation.  The signal was generated on the Python side (PSD → irfft
or direct file replay) and stored in the parameter vector before the solve.
If a scale_by parameter index is set, F is **divided** by p[scale_by_idx]
before being added.

Parameters stored under the synthetic table entity id (e.g. `_table_osc_force`):
  .n_samples    — number of signal samples (stored as Float64; cast to Int internally)
  .t_start      — time of first sample [s]
  .dt_sig       — uniform sample period [s]
  .dc           — DC offset added to the interpolated value
  .target_idx   — 0-based index of the state slot to write (stored as Float64)
  .scale_by_idx — 0-based index of a divisor in p, or -1.0 for no scaling
  .signal       — contiguous block of N Float64 values  (param slice, not a scalar)

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

    n         = Int(round(p[param_idx(spec, eid * ".n_samples")]))
    t0        = p[param_idx(spec, eid * ".t_start")]
    dt        = p[param_idx(spec, eid * ".dt_sig")]
    dc        = p[param_idx(spec, eid * ".dc")]
    tgt_i     = Int(round(p[param_idx(spec, eid * ".target_idx")])) + 1  # 0-based → 1-based
    scale_idx = Int(round(p[param_idx(spec, eid * ".scale_by_idx")]))

    # Signal slice — param_range returns a Julia UnitRange{Int} (1-based)
    sig_r = param_range(spec, eid * ".signal")

    # Clamp t to table bounds and compute interpolation indices.
    # ForwardDiff.Dual doesn't support Float64(t), so use real(t) to extract
    # the primal value for the discrete index computation.  The interpolation
    # weight alpha keeps the time dependence for accurate time-gradient AD.
    t_val = Float64(real(t))   # real(::Dual) returns the primal value; no-op for Float64
    tc    = clamp(t_val - t0, 0.0, (n - 1) * dt)
    frac  = tc / dt
    k     = clamp(Int(floor(frac)) + 1, 1, n)   # lower sample index (1-based within slice)
    k1    = clamp(k + 1, 1, n)                    # upper sample index
    alpha = frac - floor(frac)

    F = p[sig_r[k]] * (1.0 - alpha) + p[sig_r[k1]] * alpha + dc
    if scale_idx >= 0
        F = F / p[scale_idx + 1]
    end
    dx[tgt_i] += F
end

"""
    make_custom_excitation_dyn(user_fn, param_names) -> dynamics!

Returns a Numen-style dynamics function (signature
``(dx, x, p, t, spec, sys)``) that calls ``user_fn(t, params...)`` and
adds the result to ``dx[target_idx]``, applying an optional ``scale_by``
division.

Intended to be assigned to a module-level ``const`` in the user's
``dynamics.jl`` so that it can be referenced by a ``"Module.name"``
string from ``inject_custom_excitation``::

    function my_gate(t, amp, freq, t_on, t_off)
        (t < t_on || t >= t_off) && return 0.0
        return amp * sin(2π * freq * t)
    end

    const my_gate_dyn! = Main.NumenCharacterization.make_custom_excitation_dyn(
        my_gate, ("amp", "freq", "t_on", "t_off"),
    )

The closure captures both ``user_fn`` (parameterized type ``F``) and
``param_names`` (parameterized length ``N``), so Julia specializes the
generated function — there is no dynamic dispatch in the hot loop.

Synthetic-entity params read from ``p``:
  .<param_names[i]>  — one slot per user param, in given order
  .target_idx        — 0-based state slot index (stored as Float64)
  .scale_by_idx      — 0-based param index of divisor (-1.0 = no scaling)
"""
function make_custom_excitation_dyn(user_fn::F, param_names::NTuple{N, String}) where {F, N}
    function custom_dyn!(
        dx  :: AbstractVector{T},
        x   :: AbstractVector{S},
        p   :: Vector{Float64},
        t   :: Real,
        spec:: CompiledSpec,
        sys :: CompiledSystemSpec,
    ) where {T <: Real, S <: Real}
        eid       = sys.entity_ids[1]
        tgt_i     = Int(round(p[param_idx(spec, eid * ".target_idx")])) + 1
        scale_idx = Int(round(p[param_idx(spec, eid * ".scale_by_idx")]))
        # Read user params positionally from p in declared order.
        param_vals = ntuple(i -> p[param_idx(spec, eid * "." * param_names[i])], Val(N))
        F_val = user_fn(t, param_vals...)
        if scale_idx >= 0
            F_val = F_val / p[scale_idx + 1]
        end
        dx[tgt_i] += F_val
    end
    return custom_dyn!
end

end  # module NumenCharacterization
