module PneumaticDashpotDynamics

import Main: CompiledSpec, CompiledSystemSpec, groups,
             get_state, get_param, add_deriv!

# ---------------------------------------------------------------------------
# Smooth contact helper
# ---------------------------------------------------------------------------

const STOP_DELTA = 1e-6  # 1 µm C1 ramp (matches Python _STOP_DELTA)

function soft_pen(x::T) where T <: Real
    x <= 0.0 && return zero(T)
    x >= STOP_DELTA && return x - 0.5 * STOP_DELTA
    return 0.5 * x * x / STOP_DELTA
end

# ---------------------------------------------------------------------------
# Orifice mass-flow helpers
# ---------------------------------------------------------------------------

"""
    orifice_mdot(P_up, P_dn, T_up, R, Cd, A, gamma) -> T

Isentropic compressible mass flow (kg/s, always ≥ 0).
Switches between choked and unchoked branches at β_crit = (2/(γ+1))^(γ/(γ-1)).

P_up and P_dn may be Dual numbers when called from a stiff solver's Jacobian
evaluation; the remaining arguments are always Float64 (from the parameter
vector p).
"""
function orifice_mdot(
    P_up  :: T, P_dn  :: T, T_up  :: Float64,
    R     :: Float64, Cd    :: Float64, A     :: Float64, gamma :: Float64,
) :: T where T <: Real
    (P_up <= 0.0 || A <= 0.0) && return zero(T)

    beta      = max(zero(T), P_dn) / P_up
    beta_crit = (2.0 / (gamma + 1.0))^(gamma / (gamma - 1.0))

    if beta <= beta_crit
        choke_exp = (gamma + 1.0) / (2.0 * (gamma - 1.0))
        return Cd * A * P_up * sqrt(gamma / (R * T_up)) * (2.0 / (gamma + 1.0))^choke_exp
    else
        arg = beta^(2.0 / gamma) - beta^((gamma + 1.0) / gamma)
        return Cd * A * P_up * sqrt(max(zero(T), 2.0 * gamma / ((gamma - 1.0) * R * T_up) * arg))
    end
end

"""
    signed_orifice_flow(P_chamber, P_ambient, T_amb, R, gamma, Cd, A) -> T

Signed mass flow into the chamber (kg/s).
  Positive: atmosphere → chamber (P_ambient > P_chamber)
  Negative: chamber → atmosphere (P_chamber > P_ambient)
"""
function signed_orifice_flow(
    P_chamber :: T, P_ambient :: Float64,
    T_amb     :: Float64, R         :: Float64,
    gamma     :: Float64, Cd        :: Float64, A :: Float64,
) where T <: Real
    P_up  = max(T(P_ambient), P_chamber)
    P_dn  = min(T(P_ambient), P_chamber)
    mdot  = orifice_mdot(P_up, P_dn, T_amb, R, Cd, A, gamma)
    return P_ambient >= P_chamber ? mdot : -mdot
end

# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

"""
    pneumatic_dashpot_dynamics!(dx, x, p, t, spec, sys)

Isothermal gas-spring dashpot: two chambers vented to atmosphere through orifices.
Mirrors the Python `pneumatic_dashpot_dynamics` function.
"""
function pneumatic_dashpot_dynamics!(
    dx  :: AbstractVector{T},
    x   :: AbstractVector{S},
    p   :: Vector{Float64},
    t   :: Real,
    spec:: CompiledSpec,
    sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    for (eid,) in groups(sys)
        # ── State ─────────────────────────────────────────────────────────
        pos = get_state(spec, x, eid, "pneumatic_dashpot.position")
        vel = get_state(spec, x, eid, "pneumatic_dashpot.velocity")
        P_L = get_state(spec, x, eid, "pneumatic_dashpot.p_left")
        P_R = get_state(spec, x, eid, "pneumatic_dashpot.p_right")

        # ── Parameters ────────────────────────────────────────────────────
        bore  = get_param(spec, p, eid, "pneumatic_dashpot.bore_area")
        hs    = get_param(spec, p, eid, "pneumatic_dashpot.half_stroke")
        clr   = get_param(spec, p, eid, "pneumatic_dashpot.clearance")
        A_o   = get_param(spec, p, eid, "pneumatic_dashpot.orifice_area")
        Cd    = get_param(spec, p, eid, "pneumatic_dashpot.cd")
        mass  = get_param(spec, p, eid, "pneumatic_dashpot.mass")
        fric  = get_param(spec, p, eid, "pneumatic_dashpot.friction")
        kstop = get_param(spec, p, eid, "pneumatic_dashpot.k_stop")
        P_amb = get_param(spec, p, eid, "pneumatic_dashpot.p_ambient")
        T_gas = get_param(spec, p, eid, "pneumatic_dashpot.temp")
        R     = get_param(spec, p, eid, "pneumatic_dashpot.R_gas")
        gamma = get_param(spec, p, eid, "pneumatic_dashpot.gamma")

        # ── Volumes ───────────────────────────────────────────────────────
        V_L = max(bore * (hs + pos + clr), 1e-12)
        V_R = max(bore * (hs - pos + clr), 1e-12)
        dV_L = bore * vel     # expands when piston moves right
        dV_R = -bore * vel

        # ── Orifice flows ─────────────────────────────────────────────────
        mdot_L = signed_orifice_flow(P_L, P_amb, T_gas, R, gamma, Cd, A_o)
        mdot_R = signed_orifice_flow(P_R, P_amb, T_gas, R, gamma, Cd, A_o)

        # ── Pressure ODEs (isothermal) ────────────────────────────────────
        add_deriv!(spec, dx, eid, "pneumatic_dashpot.p_left",
                   (R * T_gas / V_L) * mdot_L - (P_L / V_L) * dV_L)
        add_deriv!(spec, dx, eid, "pneumatic_dashpot.p_right",
                   (R * T_gas / V_R) * mdot_R - (P_R / V_R) * dV_R)

        # ── Piston equation of motion ─────────────────────────────────────
        F_pneu    = (P_L - P_R) * bore
        F_fric    = -fric * vel
        pen_left  = -(pos + hs)   # > 0 when penetrating left stop
        pen_right = pos - hs      # > 0 when penetrating right stop
        F_stop    = kstop * (soft_pen(pen_left) - soft_pen(pen_right))

        add_deriv!(spec, dx, eid, "pneumatic_dashpot.position", vel)
        add_deriv!(spec, dx, eid, "pneumatic_dashpot.velocity",
                   (F_pneu + F_fric + F_stop) / mass)
    end
end

end  # module PneumaticDashpotDynamics
