from numen.spec.world import GenericWorld
from components import NLOscillatorComponent
from dynamics import NLOscillatorSystem

World = GenericWorld[NLOscillatorComponent, NLOscillatorSystem, None]


def make_world(
    x0: float = 1.0,
    v0: float = 0.0,
    omega: float = 1.0,
    c0: float = 0.1,
    c1: float = 1.0,
    entity_id: str = "osc",
) -> World:
    return World(
        components={entity_id: NLOscillatorComponent(
            position=x0, velocity=v0, omega=omega, c0=c0, c1=c1,
        )},
        systems={"osc_sys": NLOscillatorSystem()},
    )
