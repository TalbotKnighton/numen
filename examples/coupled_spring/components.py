from typing import Annotated, Literal

from numen.spec.component import Component
from numen.fields import IntegratedField, ParameterField


class MassComponent(Component):
    """Point mass in 1D: position and velocity are integrated state."""

    kind:     Literal["mass"] = "mass"
    position: Annotated[float, IntegratedField()] = 0.0
    velocity: Annotated[float, IntegratedField()] = 0.0
    mass:     Annotated[float, ParameterField()]  = 1.0


class SpringComponent(Component):
    """Ideal spring — physical data only. Topology lives in SpringForceSystem.entity_groups."""

    kind:        Literal["spring"] = "spring"
    k:           Annotated[float, ParameterField()] = 1.0
    rest_length: Annotated[float, ParameterField()] = 0.0
