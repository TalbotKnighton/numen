import math
from typing import Annotated, Literal

from numen.fields import ExcitationPort, IntegratedField, ParameterField
from numen.spec.component import Component

_BORE_AREA = math.pi * 0.02**2  # 40 mm bore → 1.2566e-3 m²


class PneumaticDashpotComponent(Component):
    """Piston in a sealed cylinder with orifice vents to atmosphere on both sides.

    Geometry
    --------
    The piston sits at `position` (metres, +right) inside a cylinder of
    `bore_area` (m²).  At equilibrium (position=0) each chamber has volume:

        V_eq = bore_area × (half_stroke + clearance)

    `clearance` is the small residual dead-volume length at each end of stroke;
    it prevents V → 0 if the piston reaches the mechanical stops.

    Pneumatic spring
    ----------------
    With no orifice flow the chambers act as pneumatic springs.  The
    linearised stiffness around equilibrium is:

        k = 2 × P_ambient × bore_area / (half_stroke + clearance)

    With the default geometry (40 mm bore, ±80 mm stroke, 5 mm clearance):
        k  ≈ 2 997 N/m,   f₀ = (1/2π)√(k/mass) ≈ 12.3 Hz

    Orifice damping
    ---------------
    Each chamber vents to atmosphere through an orifice of area `orifice_area`
    and discharge coefficient `cd`.  The orifice equalization time-constant:

        τ ≈ V_eq / (Cd × A_o × √(R × T))   ≈ 0.053 s  (A_o = 1e-5 m²)
        f_corner = 1 / (2π × τ)             ≈ 3 Hz

    When f_corner ≪ f₀ the chambers act as a pure spring (high Q resonance).
    When f_corner ≈ f₀ the orifice provides maximum damping at resonance.
    When f_corner ≫ f₀ pressures equalize instantly → spring vanishes.

    Excitation port
    ---------------
    The `force` ExcitationPort (targets="velocity") injects an acceleration
    signal directly into dx[velocity].  Amplitudes in the test plan are
    therefore in **m/s²** (= N/kg).  For a 0.5 kg piston, 2 m/s² ≡ 1 N.
    """

    kind: Literal["pneumatic_dashpot"] = "pneumatic_dashpot"

    # ── State (integrated by the ODE solver) ──────────────────────────────
    position: Annotated[float, IntegratedField()] = 0.0         # m  (+right from centre)
    velocity: Annotated[float, IntegratedField()] = 0.0         # m/s (+right)
    p_left:   Annotated[float, IntegratedField()] = 101_325.0   # Pa  left chamber
    p_right:  Annotated[float, IntegratedField()] = 101_325.0   # Pa  right chamber

    # ── Geometry ─────────────────────────────────────────────────────────
    bore_area:   Annotated[float, ParameterField()] = _BORE_AREA  # m²
    half_stroke: Annotated[float, ParameterField()] = 0.08        # m  (±80 mm total)
    clearance:   Annotated[float, ParameterField()] = 0.005       # m  dead-vol per side

    # ── Orifice ───────────────────────────────────────────────────────────
    orifice_area: Annotated[float, ParameterField()] = 1.0e-5  # m²
    cd:           Annotated[float, ParameterField()] = 0.7     # discharge coefficient

    # ── Piston mechanics ─────────────────────────────────────────────────
    mass:    Annotated[float, ParameterField()] = 0.5   # kg
    friction: Annotated[float, ParameterField()] = 2.0  # N·s/m  viscous friction
    k_stop:  Annotated[float, ParameterField()] = 2e5   # N/m   end-stop spring

    # ── Gas properties (dry air at 20 °C) ────────────────────────────────
    p_ambient: Annotated[float, ParameterField()] = 101_325.0  # Pa
    temp:      Annotated[float, ParameterField()] = 293.0      # K
    R_gas:     Annotated[float, ParameterField()] = 287.0      # J/(kg·K)
    gamma:     Annotated[float, ParameterField()] = 1.4

    # ── Excitation port ───────────────────────────────────────────────────
    # Amplitude units: m/s²  (force / mass applied to piston)
    force: Annotated[float, ExcitationPort(
        targets="velocity",
        port_type="effort",
        units="m/s²",
    )] = 0.0
