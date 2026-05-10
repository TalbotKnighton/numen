from numen.spec.world import GenericWorld
from components import OscillatorComponent
from dynamics import OscillatorSystem

OscillatorCatalogue = OscillatorComponent
OscillatorSystemCatalogue = OscillatorSystem
World = GenericWorld[OscillatorCatalogue, OscillatorSystemCatalogue, None]


def make_world(
    x0: float = 1.0,
    v0: float = 0.0,
    omega: float = 1.0,
    damping: float = 0.0,
    entity_id: str = "osc",
) -> World:
    return World(
        components={entity_id: {"oscillator": OscillatorComponent(position=x0, velocity=v0, omega=omega, damping=damping)}},
        systems={"osc_sys": OscillatorSystem()},
    )
