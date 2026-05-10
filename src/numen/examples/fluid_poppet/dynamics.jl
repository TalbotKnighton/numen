module FluidPoppetDynamics

import Main: CompiledSpec, CompiledSystemSpec, groups,
             get_state, get_param, add_deriv!

# ---------------------------------------------------------------------------
# Smooth contact helper
# ---------------------------------------------------------------------------

const STOP_DELTA = 1e-6  # 1 µm smoothing distance (matches Python _STOP_DELTA)

"""
    soft_pen(x) -> T

C1-smooth approximation of max(0, x) with quadratic ramp over [0, STOP_DELTA].
Removes the slope kink at contact onset that causes ODE solvers to take many
tiny rejected steps each time the poppet grazes a stop.
"""
function soft_pen(x::T) where T <: Real
    x <= 0.0 && return zero(T)
    x >= STOP_DELTA && return x - 0.5 * STOP_DELTA
    return 0.5 * x * x / STOP_DELTA
end

# ---------------------------------------------------------------------------
# Shared physics helper
# ---------------------------------------------------------------------------

"""
    orifice_mdot(P_up, P_dn, T_up, R, Cd, A, gamma) -> T

Isentropic compressible mass flow through an orifice (kg/s, always ≥ 0).
Switches between choked and unchoked branches at the critical pressure ratio:
    β_crit = (2/(γ+1))^(γ/(γ-1))

P_up, P_dn, and A may be Dual numbers (state-derived); the remaining arguments
are always Float64 (from the parameter vector p).
"""
function orifice_mdot(
    P_up::T, P_dn::T, T_up::Float64,
    R::Float64, Cd::Float64, A::Real, gamma::Float64,
) where T <: Real
    (P_up <= 0.0 || A <= 0.0) && return zero(T)

    beta      = max(zero(T), P_dn) / P_up
    beta_crit = (2.0 / (gamma + 1.0))^(gamma / (gamma - 1.0))

    if beta <= beta_crit
        # choked
        choke_exp = (gamma + 1.0) / (2.0 * (gamma - 1.0))
        return Cd * A * P_up * sqrt(gamma / (R * T_up)) * (2.0 / (gamma + 1.0))^choke_exp
    else
        # unchoked
        arg = beta^(2.0 / gamma) - beta^((gamma + 1.0) / gamma)
        return Cd * A * P_up * sqrt(max(zero(T), 2.0 * gamma / ((gamma - 1.0) * R * T_up) * arg))
    end
end

# ---------------------------------------------------------------------------
# OrificeFlowSystem
# ---------------------------------------------------------------------------

"""
    orifice_flow_dynamics!(dx, x, p, t, spec, sys)

Fixed-area isentropic compressible orifice flow between two control volumes.
Entity group stride: [cv_a, orifice, cv_b]  (group_size = 3).
"""
function orifice_flow_dynamics!(
    dx :: AbstractVector{T}, x :: AbstractVector{S}, p :: Vector{Float64},
    t  :: Real, spec :: CompiledSpec, sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    for (cv_a, orifice, cv_b) in groups(sys)
        P_a = get_state(spec, x, cv_a, "control_volume.pressure")
        P_b = get_state(spec, x, cv_b, "control_volume.pressure")
        T_a = get_param(spec, p, cv_a, "control_volume.temperature")
        T_b = get_param(spec, p, cv_b, "control_volume.temperature")
        R_a = get_param(spec, p, cv_a, "control_volume.R_specific")
        R_b = get_param(spec, p, cv_b, "control_volume.R_specific")
        V_a = get_param(spec, p, cv_a, "control_volume.volume")
        V_b = get_param(spec, p, cv_b, "control_volume.volume")
        Cd  = get_param(spec, p, orifice, "orifice.Cd")
        A   = get_param(spec, p, orifice, "orifice.area")
        gam = get_param(spec, p, orifice, "orifice.gamma")

        if P_a >= P_b
            mdot = orifice_mdot(P_a, P_b, T_a, R_a, Cd, A, gam)
        else
            mdot = -orifice_mdot(P_b, P_a, T_b, R_b, Cd, A, gam)
        end

        add_deriv!(spec, dx, cv_a, "control_volume.pressure", -(R_a * T_a / V_a) * mdot)
        add_deriv!(spec, dx, cv_b, "control_volume.pressure",  (R_b * T_b / V_b) * mdot)
    end
end

# ---------------------------------------------------------------------------
# PoppetFlowSystem
# ---------------------------------------------------------------------------

"""
    poppet_flow_dynamics!(dx, x, p, t, spec, sys)

Variable-area orifice flow through the poppet valve.
Entity group stride: [cv_inlet, poppet, cv_outlet]  (group_size = 3).
Flow area = max_flow_area * clamp(position / max_travel, 0, 1).
"""
function poppet_flow_dynamics!(
    dx :: AbstractVector{T}, x :: AbstractVector{S}, p :: Vector{Float64},
    t  :: Real, spec :: CompiledSpec, sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    for (cv_a, poppet, cv_b) in groups(sys)
        pos           = get_state(spec, x, poppet, "poppet.position")
        max_travel    = get_param(spec, p, poppet, "poppet.max_travel")
        max_flow_area = get_param(spec, p, poppet, "poppet.max_flow_area")
        Cd            = get_param(spec, p, poppet, "poppet.Cd")
        gam           = get_param(spec, p, poppet, "poppet.gamma")

        opening = clamp(pos / max_travel, zero(S), one(S))
        A       = max_flow_area * opening
        A <= 0.0 && continue

        P_a = get_state(spec, x, cv_a, "control_volume.pressure")
        P_b = get_state(spec, x, cv_b, "control_volume.pressure")
        T_a = get_param(spec, p, cv_a, "control_volume.temperature")
        T_b = get_param(spec, p, cv_b, "control_volume.temperature")
        R_a = get_param(spec, p, cv_a, "control_volume.R_specific")
        R_b = get_param(spec, p, cv_b, "control_volume.R_specific")
        V_a = get_param(spec, p, cv_a, "control_volume.volume")
        V_b = get_param(spec, p, cv_b, "control_volume.volume")

        # A is S here (derived from poppet position); orifice_mdot accepts A::Real
        if P_a >= P_b
            mdot = orifice_mdot(P_a, P_b, T_a, R_a, Cd, A, gam)
        else
            mdot = -orifice_mdot(P_b, P_a, T_b, R_b, Cd, A, gam)
        end

        add_deriv!(spec, dx, cv_a, "control_volume.pressure", -(R_a * T_a / V_a) * mdot)
        add_deriv!(spec, dx, cv_b, "control_volume.pressure",  (R_b * T_b / V_b) * mdot)
    end
end

# ---------------------------------------------------------------------------
# PoppetKinematicsSystem
# ---------------------------------------------------------------------------

"""
    poppet_kinematics_dynamics!(dx, x, p, t, spec, sys)

Position kinematics: ẋ = v.  group_size = 1.
"""
function poppet_kinematics_dynamics!(
    dx :: AbstractVector{T}, x :: AbstractVector{S}, p :: Vector{Float64},
    t  :: Real, spec :: CompiledSpec, sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    for (poppet,) in groups(sys)
        vel = get_state(spec, x, poppet, "poppet.velocity")
        add_deriv!(spec, dx, poppet, "poppet.position", vel)
    end
end

# ---------------------------------------------------------------------------
# PoppetMechanicsSystem
# ---------------------------------------------------------------------------

"""
    poppet_mechanics_dynamics!(dx, x, p, t, spec, sys)

Newton's second law for the poppet.
Entity group stride: [cv_inlet, poppet, cv_outlet]  (group_size = 3).

Forces (positive = opening direction):
  F_pressure = (P_inlet − P_outlet) · seat_area
  F_spring   = −spring_k · position − spring_preload
  F_stop     = penalty springs + dampers at both hard stops
"""
function poppet_mechanics_dynamics!(
    dx :: AbstractVector{T}, x :: AbstractVector{S}, p :: Vector{Float64},
    t  :: Real, spec :: CompiledSpec, sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    for (cv_inlet, poppet, cv_outlet) in groups(sys)
        pos   = get_state(spec, x, poppet,    "poppet.position")
        vel   = get_state(spec, x, poppet,    "poppet.velocity")
        P_in  = get_state(spec, x, cv_inlet,  "control_volume.pressure")
        P_out = get_state(spec, x, cv_outlet, "control_volume.pressure")

        mass           = get_param(spec, p, poppet, "poppet.mass")
        spring_k       = get_param(spec, p, poppet, "poppet.spring_k")
        spring_preload = get_param(spec, p, poppet, "poppet.spring_preload")
        seat_area      = get_param(spec, p, poppet, "poppet.seat_area")
        max_travel     = get_param(spec, p, poppet, "poppet.max_travel")
        k_stop         = get_param(spec, p, poppet, "poppet.stop_stiffness")
        c_stop         = get_param(spec, p, poppet, "poppet.stop_damping")

        F_pressure = (P_in - P_out) * seat_area
        F_spring   = -(spring_k * pos + spring_preload)

        pen_close    = soft_pen(-pos)
        pen_open     = soft_pen(pos - max_travel)
        alpha_close  = clamp(-pos / STOP_DELTA, zero(S), one(S))
        alpha_open   = clamp((pos - max_travel) / STOP_DELTA, zero(S), one(S))
        v_damp_close = max(zero(S), -vel) * alpha_close
        v_damp_open  = max(zero(S),  vel) * alpha_open
        F_stop = (k_stop * pen_close + c_stop * v_damp_close
                 - k_stop * pen_open  - c_stop * v_damp_open)

        add_deriv!(spec, dx, poppet, "poppet.velocity",
                   (F_pressure + F_spring + F_stop) / mass)
    end
end

end  # module FluidPoppetDynamics
