from typing import Annotated, Literal
from numen.spec.component import Component
from numen.fields import IntegratedField, ParameterField


class OscillatorComponent(Component):
    """Simple 1D harmonic oscillator: ẍ + 2ζω·ẋ + ω²x = 0."""

    kind: Literal["oscillator"] = "oscillator"
    position: Annotated[float, IntegratedField()] = 1.0
    velocity: Annotated[float, IntegratedField()] = 0.0
    omega:    Annotated[float, ParameterField()]  = 1.0   # natural frequency [rad/s]
    damping:  Annotated[float, ParameterField()]  = 0.0   # damping ratio ζ
