from typing import Annotated, Union

from pydantic import Field

from numen.spec.world import GenericWorld
from components import MassComponent, SpringComponent
from dynamics import MassKinematicsSystem, SpringForceSystem

ComponentCatalogue = Annotated[
    Union[MassComponent, SpringComponent],
    Field(discriminator="kind"),
]
SystemCatalogue = Annotated[
    Union[MassKinematicsSystem, SpringForceSystem],
    Field(discriminator="kind"),
]

World = GenericWorld[ComponentCatalogue, SystemCatalogue, None]


def make_world() -> World:
    """Three masses connected in a chain by two springs: m1 --- s1 --- m2 --- s2 --- m3.

    Components are pure data — no topology. Topology lives entirely in the system:
    SpringForceSystem.entity_groups declares which (mass_a, spring, mass_b) triplets
    exist. compile_spec validates each group's component types and flattens them to
    entity_ids = ["m1","s1","m2", "m2","s2","m3"].

    m2 receives force contributions from both springs; += accumulation sums them.
    """
    return World(
        components={
            "m1": MassComponent(position=0.0, velocity=0.0, mass=1.0),
            "s1": SpringComponent(k=10.0, rest_length=1.0),
            "m2": MassComponent(position=1.0, velocity=0.0, mass=1.0),
            "s2": SpringComponent(k=10.0, rest_length=1.0),
            "m3": MassComponent(position=3.0, velocity=0.0, mass=1.0),
        },
        systems={
            "kinematics": MassKinematicsSystem(),
            "spring_force": SpringForceSystem(entity_groups=[
                ["m1", "s1", "m2"],
                ["m2", "s2", "m3"],
            ]),
        },
    )
