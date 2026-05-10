"""Base class for all Numen ECS components.

Components are frozen Pydantic models that declare fields using Annotated
type hints with field annotations from ``numen.fields``.  They carry only
data — no topology, no solver knowledge.

The ``kind`` field (a ``Literal`` string) is used as the Pydantic
discriminator and to build index-map keys in ``compile_spec``.
"""
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
