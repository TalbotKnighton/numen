from pydantic import BaseModel


class Component(BaseModel):
    """Base class for all ECS components.

    Subclasses declare fields using Annotated with IntegratedField, DiscreteField, etc.

    Example:
        class TankComponent(Component):
            kind: Literal["tank"] = "tank"
            pressure: Annotated[float, IntegratedField()] = 0.0
            volume:   Annotated[float, ParameterField()]  = 1.0
    """

    model_config = {"frozen": True}
