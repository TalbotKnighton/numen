module FluidPoppetDynamics

import Main: CompiledSpec, CompiledSystemSpec, state_idx, param_idx

# ---------------------------------------------------------------------------
# Smooth contact helper
# ---------------------------------------------------------------------------

const STOP_DELTA = 1e-6  # 1 µm smoothing distance (matches Python _STOP_DELTA)

"""
    soft_pen(x) -> Float64

C1-smooth approximation of max(0, x) with quadratic ramp over [0, STOP_DELTA].
Removes the slope kink at contact onset that causes ODE solvers to take many
tiny rejected steps each time the poppet grazes a stop.
"""
function soft_pen(x::Float64)::Float64
    x <= 0.0 && return 0.0
    x >= STOP_DELTA && return x - 0.5 * STOP_DELTA
    return 0.5 * x * x / STOP_DELTA
end

# ---------------------------------------------------------------------------
# Shared physics helper
# ---------------------------------------------------------------------------

"""
    orifice_mdot(P_up, P_dn, T_up, R, Cd, A, gamma) -> Float64

Isentropic compressible mass flow through an orifice (kg/s, always ≥ 0).
Switches between choked and unchoked branches at the critical pressure ratio:
    β_crit = (2/(γ+1))^(γ/(γ-1))
"""
function orifice_mdot(
    P_up::Float64, P_dn::Float64, T_up::Float64,
    R::Float64, Cd::Float64, A::Float64, gamma::Float64,
)::Float64
    (P_up <= 0.0 || A <= 0.0) && return 0.0

    beta      = max(0.0, P_dn) / P_up
    beta_crit = (2.0 / (gamma + 1.0))^(gamma / (gamma - 1.0))

    if beta <= beta_crit
        # choked
        choke_exp = (gamma + 1.0) / (2.0 * (gamma - 1.0))
        return Cd * A * P_up * sqrt(gamma / (R * T_up)) * (2.0 / (gamma + 1.0))^choke_exp
    else
        # unchoked
        arg = beta^(2.0 / gamma) - beta^((gamma + 1.0) / gamma)
        return Cd * A * P_up * sqrt(max(0.0, 2.0 * gamma / ((gamma - 1.0) * R * T_up) * arg))
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
    dx::Vector{Float64}, x::Vector{Float64}, p::Vector{Float64},
    t::Float64, spec::CompiledSpec, sys::CompiledSystemSpec,
)
    gs = sys.group_size  # 3
    for i in 1:gs:length(sys.entity_ids)
        id_a = sys.entity_ids[i]
        id_o = sys.entity_ids[i + 1]
        id_b = sys.entity_ids[i + 2]

        P_a  = x[state_idx(spec, id_a * ".pressure")]
        P_b  = x[state_idx(spec, id_b * ".pressure")]
        T_a  = p[param_idx(spec, id_a * ".temperature")]
        T_b  = p[param_idx(spec, id_b * ".temperature")]
        R_a  = p[param_idx(spec, id_a * ".R_specific")]
        R_b  = p[param_idx(spec, id_b * ".R_specific")]
        V_a  = p[param_idx(spec, id_a * ".volume")]
        V_b  = p[param_idx(spec, id_b * ".volume")]
        Cd   = p[param_idx(spec, id_o * ".Cd")]
        A    = p[param_idx(spec, id_o * ".area")]
        gam  = p[param_idx(spec, id_o * ".gamma")]

        if P_a >= P_b
            mdot = orifice_mdot(P_a, P_b, T_a, R_a, Cd, A, gam)
        else
            mdot = -orifice_mdot(P_b, P_a, T_b, R_b, Cd, A, gam)
        end

        i_Pa = state_idx(spec, id_a * ".pressure")
        i_Pb = state_idx(spec, id_b * ".pressure")
        dx[i_Pa] += -(R_a * T_a / V_a) * mdot
        dx[i_Pb] +=  (R_b * T_b / V_b) * mdot
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
    dx::Vector{Float64}, x::Vector{Float64}, p::Vector{Float64},
    t::Float64, spec::CompiledSpec, sys::CompiledSystemSpec,
)
    gs = sys.group_size  # 3
    for i in 1:gs:length(sys.entity_ids)
        id_a = sys.entity_ids[i]
        id_p = sys.entity_ids[i + 1]
        id_b = sys.entity_ids[i + 2]

        pos          = x[state_idx(spec, id_p * ".position")]
        max_travel   = p[param_idx(spec, id_p * ".max_travel")]
        max_flow_area = p[param_idx(spec, id_p * ".max_flow_area")]
        Cd           = p[param_idx(spec, id_p * ".Cd")]
        gam          = p[param_idx(spec, id_p * ".gamma")]

        opening = clamp(pos / max_travel, 0.0, 1.0)
        A       = max_flow_area * opening
        A <= 0.0 && continue

        P_a = x[state_idx(spec, id_a * ".pressure")]
        P_b = x[state_idx(spec, id_b * ".pressure")]
        T_a = p[param_idx(spec, id_a * ".temperature")]
        T_b = p[param_idx(spec, id_b * ".temperature")]
        R_a = p[param_idx(spec, id_a * ".R_specific")]
        R_b = p[param_idx(spec, id_b * ".R_specific")]
        V_a = p[param_idx(spec, id_a * ".volume")]
        V_b = p[param_idx(spec, id_b * ".volume")]

        if P_a >= P_b
            mdot = orifice_mdot(P_a, P_b, T_a, R_a, Cd, A, gam)
        else
            mdot = -orifice_mdot(P_b, P_a, T_b, R_b, Cd, A, gam)
        end

        i_Pa = state_idx(spec, id_a * ".pressure")
        i_Pb = state_idx(spec, id_b * ".pressure")
        dx[i_Pa] += -(R_a * T_a / V_a) * mdot
        dx[i_Pb] +=  (R_b * T_b / V_b) * mdot
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
    dx::Vector{Float64}, x::Vector{Float64}, p::Vector{Float64},
    t::Float64, spec::CompiledSpec, sys::CompiledSystemSpec,
)
    for id_p in sys.entity_ids
        i_pos = state_idx(spec, id_p * ".position")
        i_vel = state_idx(spec, id_p * ".velocity")
        dx[i_pos] += x[i_vel]
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
    dx::Vector{Float64}, x::Vector{Float64}, p::Vector{Float64},
    t::Float64, spec::CompiledSpec, sys::CompiledSystemSpec,
)
    gs = sys.group_size  # 3
    for i in 1:gs:length(sys.entity_ids)
        id_inlet  = sys.entity_ids[i]
        id_p      = sys.entity_ids[i + 1]
        id_outlet = sys.entity_ids[i + 2]

        pos     = x[state_idx(spec, id_p * ".position")]
        vel     = x[state_idx(spec, id_p * ".velocity")]
        P_in    = x[state_idx(spec, id_inlet  * ".pressure")]
        P_out   = x[state_idx(spec, id_outlet * ".pressure")]

        mass           = p[param_idx(spec, id_p * ".mass")]
        spring_k       = p[param_idx(spec, id_p * ".spring_k")]
        spring_preload = p[param_idx(spec, id_p * ".spring_preload")]
        seat_area      = p[param_idx(spec, id_p * ".seat_area")]
        max_travel     = p[param_idx(spec, id_p * ".max_travel")]
        k_stop         = p[param_idx(spec, id_p * ".stop_stiffness")]
        c_stop         = p[param_idx(spec, id_p * ".stop_damping")]

        F_pressure = (P_in - P_out) * seat_area
        F_spring   = -(spring_k * pos + spring_preload)

        pen_close  = soft_pen(-pos)
        pen_open   = soft_pen(pos - max_travel)
        alpha_close = clamp(-pos / STOP_DELTA, 0.0, 1.0)
        alpha_open  = clamp((pos - max_travel) / STOP_DELTA, 0.0, 1.0)
        v_damp_close = max(0.0, -vel) * alpha_close
        v_damp_open  = max(0.0,  vel) * alpha_open
        F_stop = (k_stop * pen_close + c_stop * v_damp_close
                 - k_stop * pen_open  - c_stop * v_damp_open)

        i_vel = state_idx(spec, id_p * ".velocity")
        dx[i_vel] += (F_pressure + F_spring + F_stop) / mass
    end
end

end  # module FluidPoppetDynamics
