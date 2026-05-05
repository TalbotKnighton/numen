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

from numen.compiler.flatten import CompiledSpec, CompiledSystem
from numen.fields import EntityGroup
from numen.spec.system import System, DynamicsFn

from components import ControlVolumeComponent, OrificeComponent, PoppetComponent


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
    if P_up <= 0.0 or A <= 0.0:
        return 0.0

    beta = max(0.0, P_dn) / P_up
    beta_crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))

    if beta <= beta_crit:
        # choked — flow is limited by sonic conditions at the throat
        choke_exp = (gamma + 1.0) / (2.0 * (gamma - 1.0))
        return (
            Cd * A * P_up
            * np.sqrt(gamma / (R * T_up))
            * (2.0 / (gamma + 1.0)) ** choke_exp
        )
    else:
        # unchoked — subsonic throughout
        arg = beta ** (2.0 / gamma) - beta ** ((gamma + 1.0) / gamma)
        return (
            Cd * A * P_up
            * np.sqrt(max(0.0, 2.0 * gamma / ((gamma - 1.0) * R * T_up) * arg))
        )


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

        if P_a >= P_b:
            mdot = _orifice_mdot(P_a, P_b, cv_a.temperature, cv_a.R_specific,
                                 orifice.Cd, orifice.area, orifice.gamma)
        else:
            mdot = -_orifice_mdot(P_b, P_a, cv_b.temperature, cv_b.R_specific,
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
        opening = np.clip(poppet.position / poppet.max_travel, 0.0, 1.0)
        area    = poppet.max_flow_area * opening

        if area <= 0.0:
            continue

        P_a, P_b = cv_a.pressure, cv_b.pressure

        if P_a >= P_b:
            mdot = _orifice_mdot(P_a, P_b, cv_a.temperature, cv_a.R_specific,
                                 poppet.Cd, area, poppet.gamma)
        else:
            mdot = -_orifice_mdot(P_b, P_a, cv_b.temperature, cv_b.R_specific,
                                  poppet.Cd, area, poppet.gamma)

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

        # --- hard stop penalty springs at x=0 (closed) and x=max_travel (open) ---
        pen_close = max(0.0, -pos)
        pen_open  = max(0.0,  pos - poppet.max_travel)

        # damping only when moving into the stop
        v_into_close = (-vel) if (pos <= 0.0 and vel < 0.0) else 0.0
        v_into_open  = ( vel) if (pos >= poppet.max_travel and vel > 0.0) else 0.0

        F_stop = (
            + poppet.stop_stiffness * pen_close + poppet.stop_damping * v_into_close
            - poppet.stop_stiffness * pen_open  - poppet.stop_damping * v_into_open
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
