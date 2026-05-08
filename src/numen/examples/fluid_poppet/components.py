from __future__ import annotations

from typing import Annotated, Literal

from numen.fields import IntegratedField, ParameterField
from numen.spec.component import Component


class ControlVolumeComponent(Component):
    """Lumped-parameter fluid control volume — isothermal ideal gas.

    Pressure is the only integrated state.  Volume, temperature, and specific
    gas constant are fixed parameters (no heat transfer, no moving walls).

    dP/dt = (R_specific * temperature / volume) * Σ ṁ_net
    where ṁ_net is the net mass flow rate into the volume (kg/s).
    """

    kind:        Literal["control_volume"] = "control_volume"
    pressure:    Annotated[float, IntegratedField()] = 101_325.0  # Pa
    volume:      Annotated[float, ParameterField()]  = 1e-3       # m³
    temperature: Annotated[float, ParameterField()]  = 293.15     # K
    R_specific:  Annotated[float, ParameterField()]  = 287.058    # J/(kg·K)  — air


class OrificeComponent(Component):
    """Fixed-geometry orifice connecting two control volumes.

    Carries only parameters — no state.  Flow is computed in OrificeFlowSystem
    using isentropic compressible equations with choked/unchoked branches.
    """

    kind:  Literal["orifice"] = "orifice"
    Cd:    Annotated[float, ParameterField()] = 0.7    # discharge coefficient
    area:  Annotated[float, ParameterField()] = 1e-5   # m²  (throat area)
    gamma: Annotated[float, ParameterField()] = 1.4    # cp/cv — air


class PoppetComponent(Component):
    """1-DOF poppet valve with spring preload, pressure-area forces, and hard stops.

    Position x = 0 is the closed/seated position (spring fully preloaded).
    Position x = max_travel is fully open (stop on the open side).

    Flow area varies linearly: A_flow = max_flow_area * clamp(x / max_travel, 0, 1).

    Force balance (positive = opening direction):
        F = (P_inlet - P_outlet) * seat_area   — net pressure force
          - spring_k * position                 — progressive spring
          - spring_preload                      — constant preload (closing)
          + F_stop_close + F_stop_open          — penalty hard stops
    """

    kind:     Literal["poppet"] = "poppet"

    # --- integrated state ---
    position: Annotated[float, IntegratedField()] = 0.0   # m  (0 = closed)
    velocity: Annotated[float, IntegratedField()] = 0.0   # m/s

    # --- mechanical parameters ---
    mass:           Annotated[float, ParameterField()] = 0.02     # kg
    spring_k:       Annotated[float, ParameterField()] = 5_000.0  # N/m
    spring_preload: Annotated[float, ParameterField()] = 10.0     # N  (closing)
    seat_area:      Annotated[float, ParameterField()] = 5e-5     # m²
    max_travel:     Annotated[float, ParameterField()] = 3e-3     # m  (3 mm)
    stop_stiffness: Annotated[float, ParameterField()] = 1e7      # N/m
    stop_damping:   Annotated[float, ParameterField()] = 100.0    # N·s/m

    # --- flow parameters ---
    max_flow_area:  Annotated[float, ParameterField()] = 5e-5     # m²
    Cd:             Annotated[float, ParameterField()] = 0.7
    gamma:          Annotated[float, ParameterField()] = 1.4
