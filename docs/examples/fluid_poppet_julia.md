# Fluid Poppet — Julia dynamics

Source: [`src/numen/examples/fluid_poppet/dynamics.jl`](https://github.com/TalbotKnighton/numen/blob/main/src/numen/examples/fluid_poppet/dynamics.jl)

The most complete example: **4 control volumes + spring-mass poppet check valve**.
Demonstrates compressible orifice flow, smooth hard stops, and variable-area
flow coupling across multi-entity groups.

Network layout:
```
[inlet_tank] --orifice_in--> [inlet_pipe] --poppet--> [outlet_pipe] --orifice_out--> [outlet_tank]
```

The dynamics file imports the standard helpers at the top:

```julia
module FluidPoppetDynamics
import Main: CompiledSpec, CompiledSystemSpec, groups,
             get_state, get_param, add_deriv!
```

All four systems below use `groups(sys)` with tuple destructuring to name the
entities, plus `get_state` / `get_param` / `add_deriv!` for readable access.

---

## Smooth contact helper

```julia
const STOP_DELTA = 1e-6  # 1 µm — matches Python _STOP_DELTA

"""C1-smooth approximation of max(0, x)."""
function soft_pen(x::T) where T <: Real
    x <= 0.0        && return zero(T)
    x >= STOP_DELTA && return x - 0.5 * STOP_DELTA
    return 0.5 * x * x / STOP_DELTA
end
```

A bare `max(0, -pos)` stop force has a C0 kink that causes thousands of tiny
rejected steps at contact onset. The quadratic ramp over `[0, STOP_DELTA]`
makes the force C1-continuous, allowing the solver to step through contact
without step-size collapse.

---

## Isentropic orifice flow

```julia
"""
    orifice_mdot(P_up, P_dn, T_up, R, Cd, A, gamma) -> T

Compressible mass flow (kg/s, ≥ 0). Switches between choked and unchoked
branches at β_crit = (2/(γ+1))^(γ/(γ-1)).
"""
function orifice_mdot(
    P_up::T, P_dn::T, T_up::Float64,
    R::Float64, Cd::Float64, A::Real, gamma::Float64,
) where T <: Real
    (P_up <= 0.0 || A <= 0.0) && return zero(T)

    beta      = max(zero(T), P_dn) / P_up
    beta_crit = (2.0 / (gamma + 1.0))^(gamma / (gamma - 1.0))

    if beta <= beta_crit
        choke_exp = (gamma + 1.0) / (2.0 * (gamma - 1.0))
        return Cd * A * P_up * sqrt(gamma / (R * T_up)) *
               (2.0 / (gamma + 1.0))^choke_exp
    else
        arg = beta^(2.0 / gamma) - beta^((gamma + 1.0) / gamma)
        return Cd * A * P_up * sqrt(
            max(zero(T), 2.0 * gamma / ((gamma - 1.0) * R * T_up) * arg))
    end
end
```

`P_up`, `P_dn`, and `A` may be `Dual` numbers when called during Jacobian
evaluation; the remaining arguments are always `Float64` from the parameter vector.

---

## OrificeFlowSystem — fixed-area orifice

```julia
"""group_size = 3: [cv_a, orifice, cv_b]"""
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

        # Direction check — always call orifice_mdot with P_up ≥ P_dn
        if P_a >= P_b
            mdot = orifice_mdot(P_a, P_b, T_a, R_a, Cd, A, gam)
        else
            mdot = -orifice_mdot(P_b, P_a, T_b, R_b, Cd, A, gam)
        end

        # dP/dt = (R·T/V) · ṁ_net  (isothermal ideal gas)
        add_deriv!(spec, dx, cv_a, "control_volume.pressure", -(R_a * T_a / V_a) * mdot)
        add_deriv!(spec, dx, cv_b, "control_volume.pressure",  (R_b * T_b / V_b) * mdot)
    end
end
```

---

## PoppetFlowSystem — variable-area orifice

```julia
"""group_size = 3: [cv_inlet, poppet, cv_outlet]"""
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

        # Flow area scales linearly with position (0 → fully closed, max_travel → fully open)
        opening = clamp(pos / max_travel, zero(S), one(S))
        A       = max_flow_area * opening
        A <= 0.0 && continue    # poppet sealed — no flow

        P_a = get_state(spec, x, cv_a, "control_volume.pressure")
        P_b = get_state(spec, x, cv_b, "control_volume.pressure")
        T_a = get_param(spec, p, cv_a, "control_volume.temperature")
        R_a = get_param(spec, p, cv_a, "control_volume.R_specific")
        V_a = get_param(spec, p, cv_a, "control_volume.volume")
        T_b = get_param(spec, p, cv_b, "control_volume.temperature")
        R_b = get_param(spec, p, cv_b, "control_volume.R_specific")
        V_b = get_param(spec, p, cv_b, "control_volume.volume")

        if P_a >= P_b
            mdot = orifice_mdot(P_a, P_b, T_a, R_a, Cd, A, gam)
        else
            mdot = -orifice_mdot(P_b, P_a, T_b, R_b, Cd, A, gam)
        end

        add_deriv!(spec, dx, cv_a, "control_volume.pressure", -(R_a * T_a / V_a) * mdot)
        add_deriv!(spec, dx, cv_b, "control_volume.pressure",  (R_b * T_b / V_b) * mdot)
    end
end
```

Note `A = max_flow_area * opening` — `A` is derived from state `pos`, so it
may be a `Dual` number during Jacobian evaluation. `orifice_mdot` accepts
`A::Real` for exactly this reason.

---

## PoppetKinematicsSystem

```julia
"""group_size = 1 — ẋ = v for each poppet entity."""
function poppet_kinematics_dynamics!(
    dx :: AbstractVector{T}, x :: AbstractVector{S}, p :: Vector{Float64},
    t  :: Real, spec :: CompiledSpec, sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    for (poppet,) in groups(sys)
        vel = get_state(spec, x, poppet, "poppet.velocity")
        add_deriv!(spec, dx, poppet, "poppet.position", vel)
    end
end
```

---

## PoppetMechanicsSystem

```julia
"""group_size = 3: [cv_inlet, poppet, cv_outlet]

Forces (positive = opening):
  F_pressure = (P_inlet − P_outlet) · seat_area
  F_spring   = −k·pos − preload
  F_stop     = smooth penalty at both hard stops
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

        # Smooth stops at pos=0 (closed seat) and pos=max_travel (fully open)
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
```

---

## Performance

With `JuliaServerBackend(method="Tsit5", rtol=1e-8, atol=1e-10)`:

| Backend | Warm solve (150 ms sim) |
|---|---|
| ScipyBackend | ~9000 ms |
| JAXBackend (Dopri5) | ~6 ms |
| JuliaServerBackend (Tsit5) | ~14 ms |

Both JAX and Julia are ~600–1500× faster than scipy for this 6-state stiff system.
