from numen.spec.world import GenericWorld

from components import PneumaticDashpotComponent
from dynamics import PneumaticDashpotSystem

World = GenericWorld[PneumaticDashpotComponent, PneumaticDashpotSystem, None]


def make_world(
    x0: float = 0.0,
    v0: float = 0.0,
    p_left: float = 101_325.0,
    p_right: float = 101_325.0,
    orifice_area: float = 1.0e-5,
    friction: float = 2.0,
    entity_id: str = "piston",
) -> World:
    """Factory for the pneumatic dashpot world.

    Args:
        x0:           Initial piston position (m, +right from centre).
        v0:           Initial piston velocity (m/s).
        p_left:       Initial left-chamber pressure (Pa).
        p_right:      Initial right-chamber pressure (Pa).
        orifice_area: Orifice area (m²) — controls the frequency-dependent
                      damping.  Default 1e-5 m² (f_corner ≈ 3 Hz < f₀ ≈ 12 Hz).
        friction:     Viscous friction coefficient (N·s/m).
        entity_id:    Name for the piston entity (used by SnapshotCollector).
    """
    return World(
        components={entity_id: PneumaticDashpotComponent(
            position=x0, velocity=v0,
            p_left=p_left, p_right=p_right,
            orifice_area=orifice_area,
            friction=friction,
        )},
        systems={"dashpot_sys": PneumaticDashpotSystem()},
    )
