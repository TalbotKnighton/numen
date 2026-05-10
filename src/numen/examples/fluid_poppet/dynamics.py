"""
Fluid poppet check valve — Python dynamics functions (scipy backend).

Four systems:
  OrificeFlowSystem       — isentropic compressible flow through a fixed orifice
  PoppetFlowSystem        — same physics but area proportional to poppet opening
  PoppetKinematicsSystem  — ẋ = v
  PoppetMechanicsSystem   — m·v̇ = F_pressure + F_spring + F_stops
"""
from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
import jax.numpy as jnp

from numen.compiler.flatten import CompiledSpec, CompiledSystem
from numen.fields import EntityGroup
from numen.spec.system import System, DynamicsFn

from components import ControlVolumeComponent, OrificeComponent, PoppetComponent


# ---------------------------------------------------------------------------
# Smooth contact helper
# ---------------------------------------------------------------------------

_STOP_DELTA = 1e-6  # 1 µm smoothing distance for hard-stop forces

def _soft_pen(pos_from_stop: float, delta: float = _STOP_DELTA) -> float:
    """C1-smooth penetration depth.

    Approximates max(0, pos_from_stop) with a quadratic ramp over [0, delta]:
      0                              for pos_from_stop <= 0
      pos_from_stop²/(2·delta)       for 0 < pos_from_stop < delta   (C1 at 0)
      pos_from_stop − delta/2        for pos_from_stop >= delta       (C1 at delta)

    The transition smooths the slope discontinuity at contact onset, which
    prevents the ODE solver from taking many tiny rejected steps each time the
    poppet grazes the stop.
    """
    x = pos_from_stop
    return jnp.where(
        x <= 0.0,
        0.0,
        jnp.where(x >= delta, x - 0.5 * delta, 0.5 * x * x / delta),
    )


# ---------------------------------------------------------------------------
# Shared physics helper
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
    """Isentropic compressible mass flow through an orifice (kg/s, always ≥ 0).

    Switches between choked and unchoked branches at the critical pressure ratio:
        β_crit = (2 / (γ+1))^(γ/(γ-1))

    Choked   (β ≤ β_crit):  ṁ = Cd·A·P_up · √(γ/(R·T)) · (2/(γ+1))^((γ+1)/(2(γ-1)))
    Unchoked (β > β_crit):  ṁ = Cd·A·P_up · √(2γ / ((γ-1)·R·T) · (β^(2/γ) − β^((γ+1)/γ)))

    where β = P_dn / P_up.
    """
    # Both branches are always evaluated (required for JAX tracing).
    # np.where / np.maximum dispatch correctly on both numpy and JAX arrays.
    safe_P_up = jnp.maximum(P_up, 1e-300)           # avoid divide-by-zero in beta
    beta      = jnp.maximum(0.0, P_dn) / safe_P_up
    beta_crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))

    # Choked branch
    choke_exp   = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    mdot_choked = (
        Cd * A * P_up
        * jnp.sqrt(gamma / (R * T_up))
        * (2.0 / (gamma + 1.0)) ** choke_exp
    )

    # Unchoked branch — guard arg ≥ 0 so sqrt is always real
    arg           = beta ** (2.0 / gamma) - beta ** ((gamma + 1.0) / gamma)
    mdot_unchoked = (
        Cd * A * P_up
        * jnp.sqrt(jnp.maximum(0.0, 2.0 * gamma / ((gamma - 1.0) * R * T_up) * arg))
    )

    mdot = jnp.where(beta <= beta_crit, mdot_choked, mdot_unchoked)
    return jnp.where((P_up <= 0.0) | (A <= 0.0), 0.0, mdot)


def _apply_flow(
    mdot: float,
    id_a: str,
    id_b: str,
    cv_a: ControlVolumeComponent,
    cv_b: ControlVolumeComponent,
    dx: object,
    spec: CompiledSpec,
) -> None:
    """Accumulate dP/dt contributions.  Positive mdot = flow from a → b."""
    da = spec.dx_view(id_a, ControlVolumeComponent, dx)
    db = spec.dx_view(id_b, ControlVolumeComponent, dx)
    # dP/dt = R·T/V · dm/dt  (isothermal ideal gas)
    da.pressure += -(cv_a.R_specific * cv_a.temperature / cv_a.volume) * mdot
    db.pressure +=  (cv_b.R_specific * cv_b.temperature / cv_b.volume) * mdot


# ---------------------------------------------------------------------------
# OrificeFlowSystem
# ---------------------------------------------------------------------------

def orifice_flow_dynamics(
    dx, x, p, t,
    spec: CompiledSpec,
    system: CompiledSystem[tuple[str, str, str]],
) -> None:
    """Isentropic compressible flow through a fixed-area orifice.

    Entity group: [ControlVolumeComponent, OrificeComponent, ControlVolumeComponent].
    Flow direction is determined at runtime from the pressure difference.
    """
    for id_a, id_o, id_b in system.entity_groups:
        cv_a    = spec.view(id_a, ControlVolumeComponent, x, p)
        orifice = spec.view(id_o, OrificeComponent,       x, p)
        cv_b    = spec.view(id_b, ControlVolumeComponent, x, p)

        P_a, P_b = cv_a.pressure, cv_b.pressure
        a_is_up  = P_a >= P_b

        # Select upstream state without Python if (JAX-compatible)
        P_up = jnp.where(a_is_up, P_a, P_b)
        P_dn = jnp.where(a_is_up, P_b, P_a)
        T_up = jnp.where(a_is_up, cv_a.temperature, cv_b.temperature)
        R_up = jnp.where(a_is_up, cv_a.R_specific,  cv_b.R_specific)
        sign = jnp.where(a_is_up, 1.0, -1.0)

        mdot = sign * _orifice_mdot(P_up, P_dn, T_up, R_up,
                                    orifice.Cd, orifice.area, orifice.gamma)
        _apply_flow(mdot, id_a, id_b, cv_a, cv_b, dx, spec)


class OrificeFlowSystem(System):
    """Fixed-area orifice flow between pairs of control volumes.

    entity_groups: list of [cv_a, orifice, cv_b] triples.
    """
    component_types: ClassVar[tuple[type, ...]] = ()
    entity_slots:    ClassVar[EntityGroup]      = EntityGroup(
        ControlVolumeComponent, OrificeComponent, ControlVolumeComponent
    )
    python_fn: ClassVar[DynamicsFn] = staticmethod(orifice_flow_dynamics)

    kind:         Literal["orifice_flow"] = "orifice_flow"
    dynamics_fn:  str = "FluidPoppetDynamics.orifice_flow_dynamics!"


# ---------------------------------------------------------------------------
# PoppetFlowSystem
# ---------------------------------------------------------------------------

def poppet_flow_dynamics(
    dx, x, p, t,
    spec: CompiledSpec,
    system: CompiledSystem[tuple[str, str, str]],
) -> None:
    """Variable-area orifice flow through the poppet valve.

    Entity group: [ControlVolumeComponent, PoppetComponent, ControlVolumeComponent].
    Flow area interpolates linearly from 0 (closed) to max_flow_area (fully open).
    """
    for id_a, id_p, id_b in system.entity_groups:
        cv_a   = spec.view(id_a, ControlVolumeComponent, x, p)
        poppet = spec.view(id_p, PoppetComponent,        x, p)
        cv_b   = spec.view(id_b, ControlVolumeComponent, x, p)

        # Flow area: linear with position, clamped to [0, max_flow_area]
        # area=0 naturally produces mdot=0 through _orifice_mdot — no early exit needed
        opening = jnp.clip(poppet.position / poppet.max_travel, 0.0, 1.0)
        area    = poppet.max_flow_area * opening

        P_a, P_b = cv_a.pressure, cv_b.pressure
        a_is_up  = P_a >= P_b

        P_up = jnp.where(a_is_up, P_a, P_b)
        P_dn = jnp.where(a_is_up, P_b, P_a)
        T_up = jnp.where(a_is_up, cv_a.temperature, cv_b.temperature)
        R_up = jnp.where(a_is_up, cv_a.R_specific,  cv_b.R_specific)
        sign = jnp.where(a_is_up, 1.0, -1.0)

        mdot = sign * _orifice_mdot(P_up, P_dn, T_up, R_up, poppet.Cd, area, poppet.gamma)
        _apply_flow(mdot, id_a, id_b, cv_a, cv_b, dx, spec)


class PoppetFlowSystem(System):
    """Variable-area orifice flow through the poppet valve."""
    component_types: ClassVar[tuple[type, ...]] = ()
    entity_slots:    ClassVar[EntityGroup]      = EntityGroup(
        ControlVolumeComponent, PoppetComponent, ControlVolumeComponent
    )
    python_fn: ClassVar[DynamicsFn] = staticmethod(poppet_flow_dynamics)

    kind:        Literal["poppet_flow"] = "poppet_flow"
    dynamics_fn: str = "FluidPoppetDynamics.poppet_flow_dynamics!"


# ---------------------------------------------------------------------------
# PoppetKinematicsSystem
# ---------------------------------------------------------------------------

def poppet_kinematics_dynamics(
    dx, x, p, t,
    spec: CompiledSpec,
    system: CompiledSystem[tuple[str]],
) -> None:
    """Position kinematics: ẋ = v."""
    for (id_p,) in system.entity_groups:
        poppet = spec.view(id_p, PoppetComponent, x, p)
        dp     = spec.dx_view(id_p, PoppetComponent, dx)
        dp.position += poppet.velocity


class PoppetKinematicsSystem(System):
    """Integrates poppet position for all PoppetComponent entities."""
    component_types: ClassVar[tuple[type, ...]] = (PoppetComponent,)
    python_fn:       ClassVar[DynamicsFn]       = staticmethod(poppet_kinematics_dynamics)

    kind:        Literal["poppet_kinematics"] = "poppet_kinematics"
    dynamics_fn: str = "FluidPoppetDynamics.poppet_kinematics_dynamics!"


# ---------------------------------------------------------------------------
# PoppetMechanicsSystem
# ---------------------------------------------------------------------------

def poppet_mechanics_dynamics(
    dx, x, p, t,
    spec: CompiledSpec,
    system: CompiledSystem[tuple[str, str, str]],
) -> None:
    """Newton's second law for the poppet.

    Entity group: [ControlVolumeComponent (inlet), PoppetComponent, ControlVolumeComponent (outlet)].

    Forces (positive = opening direction, i.e. increasing position):
      F_pressure = (P_inlet − P_outlet) · seat_area
      F_spring   = −spring_k · position − spring_preload
      F_stop     = penalty springs + dampers at both hard stops
    """
    for id_inlet, id_p, id_outlet in system.entity_groups:
        inlet   = spec.view(id_inlet,  ControlVolumeComponent, x, p)
        poppet  = spec.view(id_p,      PoppetComponent,        x, p)
        outlet  = spec.view(id_outlet, ControlVolumeComponent, x, p)
        dp      = spec.dx_view(id_p,   PoppetComponent,        dx)

        pos = poppet.position
        vel = poppet.velocity

        # --- pressure force (net opening) ---
        F_pressure = (inlet.pressure - outlet.pressure) * poppet.seat_area

        # --- spring: progressive + preload (always closing) ---
        F_spring = -(poppet.spring_k * pos + poppet.spring_preload)

        # --- hard stop penalty springs with smooth 1-µm onset (C1 at contact) ---
        # _soft_pen removes the slope kink that causes ODE solvers to take many
        # tiny rejected steps each time the poppet grazes a stop.
        pen_close = _soft_pen(-pos)
        pen_open  = _soft_pen(pos - poppet.max_travel)

        # Damping blended by the same smooth contact factor (0→1 over _STOP_DELTA)
        alpha_close = jnp.clip(-pos / _STOP_DELTA, 0.0, 1.0)
        alpha_open  = jnp.clip((pos - poppet.max_travel) / _STOP_DELTA, 0.0, 1.0)
        v_damp_close = jnp.maximum(0.0, -vel) * alpha_close
        v_damp_open  = jnp.maximum(0.0,  vel) * alpha_open

        F_stop = (
            + poppet.stop_stiffness * pen_close + poppet.stop_damping * v_damp_close
            - poppet.stop_stiffness * pen_open  - poppet.stop_damping * v_damp_open
        )

        dp.velocity += (F_pressure + F_spring + F_stop) / poppet.mass


class PoppetMechanicsSystem(System):
    """Newton dynamics for the poppet: net force / mass → acceleration."""
    component_types: ClassVar[tuple[type, ...]] = ()
    entity_slots:    ClassVar[EntityGroup]      = EntityGroup(
        ControlVolumeComponent, PoppetComponent, ControlVolumeComponent
    )
    python_fn: ClassVar[DynamicsFn] = staticmethod(poppet_mechanics_dynamics)

    kind:        Literal["poppet_mechanics"] = "poppet_mechanics"
    dynamics_fn: str = "FluidPoppetDynamics.poppet_mechanics_dynamics!"
