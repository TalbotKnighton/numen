from typing import Generic, TypeVar
from pydantic import BaseModel

CC = TypeVar("CC")  # ComponentCatalogue union type
SC = TypeVar("SC")  # SystemCatalogue union type
BC = TypeVar("BC")  # CallbackCatalogue union type


class GenericWorld(BaseModel, Generic[CC, SC, BC]):
    """Typed ECS world. CC, SC, BC are discriminated union types for each catalogue.

    Each entity maps to a dict of components keyed by component kind, supporting
    multiple components per entity:

        components = {
            "robot": {
                "body":   BodyComponent(...),
                "sensor": SensorComponent(...),
            },
            "target": {
                "point_mass": PointMassComponent(...),
            },
        }

    State/param keys use the full path ``entity_id.component_kind.field_name``.

    Example:
        ComponentCatalogue = Annotated[TankComponent | PipeComponent, Field(discriminator="kind")]
        SystemCatalogue    = Annotated[TankSystem    | PipeSystem,    Field(discriminator="kind")]
        CallbackCatalogue  = Annotated[ControlCallback,               Field(discriminator="kind")]

        World = GenericWorld[ComponentCatalogue, SystemCatalogue, CallbackCatalogue]

        world = World(
            components={"tank_a": {"tank": TankComponent(...)}},
            systems={"tank_sys": TankSystem(...)},
            callbacks={},
        )
    """

    components: dict[str, dict[str, CC]] = {}
    systems:    dict[str, SC] = {}
    callbacks:  dict[str, BC] = {}
