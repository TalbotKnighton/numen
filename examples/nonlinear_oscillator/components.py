from typing import Annotated, Literal
from numen.spec.component import Component
from numen.fields import IntegratedField, ParameterField


class NLOscillatorComponent(Component):
    """1D oscillator with position-dependent damping: ẍ + (c0 + c1·x²)ẋ + ω²x = 0.

    c0 is the baseline (linear) damping; c1 scales how much the damping
    grows with displacement squared.  Both must be >= 0 for the system to
    dissipate energy and settle to rest.
    """

    kind:     Literal["nl_oscillator"] = "nl_oscillator"
    position: Annotated[float, IntegratedField()] = 1.0
    velocity: Annotated[float, IntegratedField()] = 0.0
    omega:    Annotated[float, ParameterField()]  = 1.0   # natural frequency [rad/s]
    c0:       Annotated[float, ParameterField()]  = 0.1   # linear damping coefficient
    c1:       Annotated[float, ParameterField()]  = 1.0   # position-dependent damping coefficient
