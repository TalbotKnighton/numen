# Pneumatic Dashpot — Julia dynamics

Source: [`src/numen/examples/pneumatic_dashpot/dynamics.jl`](https://github.com/TalbotKnighton/numen/blob/main/src/numen/examples/pneumatic_dashpot/dynamics.jl)

A piston in a sealed cylinder with orifice vents on both ends.
The air spring / viscous damping effect is **strongly frequency-dependent**:
large orifices → low damping (gas escapes freely); small orifices → high
damping (gas is trapped, acts as a stiff spring).

This example uses **Rodas5P** (stiff solver) because the pressure ODEs are
stiff relative to the mechanical motion.

---

## Full source

```julia
module PneumaticDashpotDynamics

import Main: CompiledSpec, CompiledSystemSpec, groups,
             get_state, get_param, add_deriv!

const STOP_DELTA = 1e-6

function soft_pen(x::T) where T <: Real
    x <= 0.0        && return zero(T)
    x >= STOP_DELTA && return x - 0.5 * STOP_DELTA
    return 0.5 * x * x / STOP_DELTA
end

"""
    orifice_mdot(P_up, P_dn, T_up, R, Cd, A, gamma) -> T

Isentropic compressible mass flow (kg/s, ≥ 0).
Switches at β_crit = (2/(γ+1))^(γ/(γ-1)).
"""
function orifice_mdot(
    P_up::T, P_dn::T, T_up::Float64,
    R::Float64, Cd::Float64, A::Float64, gamma::Float64,
)::T where T <: Real
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

"""
    signed_orifice_flow(P_chamber, P_ambient, ...) -> T

Signed mass flow into the chamber (kg/s).
  Positive: atmosphere → chamber  (P_ambient > P_chamber)
  Negative: chamber → atmosphere  (P_chamber > P_ambient)
"""
function signed_orifice_flow(
    P_chamber::T, P_ambient::Float64,
    T_amb::Float64, R::Float64,
    gamma::Float64, Cd::Float64, A::Float64,
) where T <: Real
    P_up = max(T(P_ambient), P_chamber)
    P_dn = min(T(P_ambient), P_chamber)
    mdot = orifice_mdot(P_up, P_dn, T_amb, R, Cd, A, gamma)
    return P_ambient >= P_chamber ? mdot : -mdot
end

"""
    pneumatic_dashpot_dynamics!(dx, x, p, t, spec, sys)

Isothermal gas-spring dashpot: two chambers vented to atmosphere.
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
        # State
        pos = get_state(spec, x, eid, "pneumatic_dashpot.position")
        vel = get_state(spec, x, eid, "pneumatic_dashpot.velocity")
        P_L = get_state(spec, x, eid, "pneumatic_dashpot.p_left")
        P_R = get_state(spec, x, eid, "pneumatic_dashpot.p_right")

        # Parameters
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

        # Chamber volumes (clamped to avoid V=0 singularity)
        V_L  = max(bore * (hs + pos + clr), 1e-12)
        V_R  = max(bore * (hs - pos + clr), 1e-12)
        dV_L = bore * vel     # left chamber expands when piston moves right
        dV_R = -bore * vel

        # Orifice flows (positive = into chamber)
        mdot_L = signed_orifice_flow(P_L, P_amb, T_gas, R, gamma, Cd, A_o)
        mdot_R = signed_orifice_flow(P_R, P_amb, T_gas, R, gamma, Cd, A_o)

        # Pressure ODEs (isothermal: dP/dt = R·T/V · ṁ - P/V · dV/dt)
        add_deriv!(spec, dx, eid, "pneumatic_dashpot.p_left",
                   (R * T_gas / V_L) * mdot_L - (P_L / V_L) * dV_L)
        add_deriv!(spec, dx, eid, "pneumatic_dashpot.p_right",
                   (R * T_gas / V_R) * mdot_R - (P_R / V_R) * dV_R)

        # Piston equation of motion
        F_pneu    = (P_L - P_R) * bore
        F_fric    = -fric * vel
        pen_left  = -(pos + hs)   # > 0 inside left stop
        pen_right = pos - hs      # > 0 inside right stop
        F_stop    = kstop * (soft_pen(pen_left) - soft_pen(pen_right))

        add_deriv!(spec, dx, eid, "pneumatic_dashpot.position", vel)
        add_deriv!(spec, dx, eid, "pneumatic_dashpot.velocity",
                   (F_pneu + F_fric + F_stop) / mass)
    end
end

end  # module PneumaticDashpotDynamics
```

---

## Key points

**`signed_orifice_flow` helper** — wraps `orifice_mdot` so callers don't need
to check pressure direction. Returns positive flow into the chamber, negative
flow out. Using a helper keeps the main dynamics function readable.

**Isothermal pressure ODE** — `dP/dt = (R·T/V)·ṁ_net - (P/V)·(dV/dt)`. The
first term comes from mass flowing in; the second from the piston compressing
or expanding the volume. `add_deriv!` accumulates with `+=`, so any additional
system could contribute.

**Volume clamp** — `max(bore * (...), 1e-12)` prevents division by zero if
the piston reaches the end of stroke. The smooth stop force should prevent this
in practice; the clamp is a safety net.

**Stiff solver** — use `method="Rodas5P"` with this example. The pressure
dynamics are stiff relative to the mechanical motion (especially for small
orifice areas), and explicit solvers like Tsit5 will take very small steps.

**Performance note** — under `Rodas5P`, dynamics is called many times per step
during Jacobian evaluation. For very tight inner loops you can drop to the
low-level `state_idx` / `param_idx` form (cache the index, reuse for read+write).
See the [Julia Reference](../julia.md) for the trade-off.

---

## Running with the stiff solver

```python
from numen.bridge.runtime import JuliaBackend

result = JuliaBackend(
    julia_file = "dynamics.jl",
    method     = "Rodas5P",   # stiff Rosenbrock solver
    rtol       = 1e-6,
    atol       = 1e-8,
).solve(spec, tspan=(0.0, 1.0))
```

`Rodas5P` uses `ForwardDiff` to evaluate the state Jacobian ∂f/∂x via the
sparse `jac_prototype` pattern that `compile_spec()` builds. The
`{T <: Real, S <: Real}` signature in every dynamics function is what makes
this work without modification.
