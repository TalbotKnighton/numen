"""Pneumatic dashpot dynamics — JAX-compatible (scipy and JAX backends).

Physics
-------
Isothermal ideal-gas model:

    dP/dt = (R·T / V) · ṁ_in_net  −  (P / V) · dV/dt

Left  chamber: V_L = bore_area · (half_stroke + position + clearance)
                     dV_L/dt =  bore_area · velocity
Right chamber: V_R = bore_area · (half_stroke − position + clearance)
                     dV_R/dt = −bore_area · velocity

Each chamber vents to atmosphere through an orifice; the signed mass flow
(positive = into chamber) is computed with the isentropic compressible formula
from CLAUDE.md, switching on the larger (upstream) pressure.

Piston equation of motion:
    m · dv/dt = (P_L − P_R) · A  −  friction · v  +  F_stop  [+  F_excitation]

End stops use a C1-smooth 1 µm ramp to avoid catastrophic step rejection.
"""
from __future__ import annotations

from typing import ClassVar, Literal

import jax.numpy as jnp
import numpy as np

from numen.compiler.flatten import CompiledSpec, CompiledSystem  # noqa: F401
from numen.spec.system import System, DynamicsFn

from components import PneumaticDashpotComponent


# ---------------------------------------------------------------------------
# Smooth contact helper  (mirrors fluid_poppet)
# ---------------------------------------------------------------------------

_STOP_DELTA = 1e-6  # 1 µm C1 ramp width


def _soft_pen(pos_from_stop: float) -> float:
    """C1-smooth penetration depth: approximates max(0, pos_from_stop)."""
    x = pos_from_stop
    return jnp.where(
        x <= 0.0,
        0.0,
        jnp.where(x >= _STOP_DELTA, x - 0.5 * _STOP_DELTA, 0.5 * x * x / _STOP_DELTA),
    )


# ---------------------------------------------------------------------------
# Orifice mass-flow helpers
# ---------------------------------------------------------------------------

def _orifice_mdot(
    P_up: float,
    P_dn: float,
    T_up: float,
    R: float,
    Cd: float,
    A: float,
    gamma: float,
) -> float:
    """Unsigned isentropic compressible mass flow (kg/s), always ≥ 0.

    Both branches are always evaluated (required for JAX tracing); guarded
    against NaN with jnp.maximum before the sqrt.
    """
    safe_P_up = jnp.maximum(P_up, 1e-300)
    beta      = jnp.maximum(0.0, P_dn) / safe_P_up
    beta_crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))

    choke_exp    = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    mdot_choked  = Cd * A * P_up * jnp.sqrt(gamma / (R * T_up)) * (2.0 / (gamma + 1.0)) ** choke_exp

    arg           = beta ** (2.0 / gamma) - beta ** ((gamma + 1.0) / gamma)
    mdot_unchoked = Cd * A * P_up * jnp.sqrt(jnp.maximum(0.0, 2.0 * gamma / ((gamma - 1.0) * R * T_up) * arg))

    mdot = jnp.where(beta <= beta_crit, mdot_choked, mdot_unchoked)
    return jnp.where((P_up <= 0.0) | (A <= 0.0), 0.0, mdot)


def _signed_orifice_flow(
    P_chamber: float,
    P_ambient: float,
    T: float,
    R: float,
    gamma: float,
    Cd: float,
    A: float,
) -> float:
    """Signed orifice mass flow into the chamber (kg/s).

    Positive  → gas flows from atmosphere into the chamber (P_ambient > P_chamber).
    Negative  → gas flows out of the chamber to atmosphere (P_chamber > P_ambient).
    """
    P_up  = jnp.maximum(P_ambient, P_chamber)
    P_dn  = jnp.minimum(P_ambient, P_chamber)
    mdot  = _orifice_mdot(P_up, P_dn, T, R, Cd, A, gamma)
    return jnp.where(P_ambient >= P_chamber, mdot, -mdot)


# ---------------------------------------------------------------------------
# Dynamics function
# ---------------------------------------------------------------------------

def pneumatic_dashpot_dynamics(
    dx: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
    t: float,
    spec: CompiledSpec,
    system: CompiledSystem[tuple[str]],
) -> None:
    """Isothermal gas-spring dashpot with compressible orifice vents."""
    for (entity_id,) in system.entity_groups:
        c  = spec.view(entity_id, PneumaticDashpotComponent, x, p)
        dc = spec.dx_view(entity_id, PneumaticDashpotComponent, dx)

        pos = c.position
        vel = c.velocity
        P_L = c.p_left
        P_R = c.p_right

        # Chamber volumes (clearance keeps V > 0 at full stroke)
        V_L = jnp.maximum(c.bore_area * (c.half_stroke + pos + c.clearance), 1e-12)
        V_R = jnp.maximum(c.bore_area * (c.half_stroke - pos + c.clearance), 1e-12)

        # Volume rate of change: +A·v for left (expands when piston moves right)
        dV_L = c.bore_area * vel
        dV_R = -c.bore_area * vel

        # Signed orifice mass flows into each chamber
        mdot_L = _signed_orifice_flow(P_L, c.p_ambient, c.temp, c.R_gas, c.gamma, c.cd, c.orifice_area)
        mdot_R = _signed_orifice_flow(P_R, c.p_ambient, c.temp, c.R_gas, c.gamma, c.cd, c.orifice_area)

        # Pressure ODEs — isothermal ideal gas
        dc.p_left  += (c.R_gas * c.temp / V_L) * mdot_L - (P_L / V_L) * dV_L
        dc.p_right += (c.R_gas * c.temp / V_R) * mdot_R - (P_R / V_R) * dV_R

        # Net pneumatic force on piston (left chamber pushes right, right pushes left)
        F_pneumatic = (P_L - P_R) * c.bore_area

        # Viscous friction
        F_friction = -c.friction * vel

        # Soft end stops: push piston away from each wall
        pen_left  = -(pos + c.half_stroke)  # > 0 when penetrating left stop
        pen_right = pos - c.half_stroke     # > 0 when penetrating right stop
        F_stop = c.k_stop * (_soft_pen(pen_left) - _soft_pen(pen_right))

        # Kinematics
        dc.position += vel
        dc.velocity += (F_pneumatic + F_friction + F_stop) / c.mass


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

class PneumaticDashpotSystem(System):
    component_types: ClassVar[tuple[type, ...]] = (PneumaticDashpotComponent,)
    python_fn:       ClassVar[DynamicsFn]       = staticmethod(pneumatic_dashpot_dynamics)
    kind:            Literal["pneumatic_dashpot_system"] = "pneumatic_dashpot_system"
    dynamics_fn:     str = "PneumaticDashpotDynamics.pneumatic_dashpot_dynamics!"
