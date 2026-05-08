from typing import Annotated, Literal
from numen.spec.component import Component
from numen.fields import IntegratedField, ParameterField, ExcitationPort


class NLOscillatorComponent(Component):
    """1D oscillator with position-dependent damping: ẍ + (c0 + c1·x²)ẋ + ω²x = F(t).

    c0 is the baseline (linear) damping; c1 scales how much the damping
    grows with displacement squared.  Both must be >= 0 for the system to
    dissipate energy and settle to rest.

    The ExcitationPort 'force' is an effort-source input port.  When used with
    the characterization framework, inject_excitation() adds a time-varying
    F(t) = amp·sin(2π·freq·t) + dc directly to d(velocity)/dt.  The
    NLOscillatorSystem dynamics function does not need to reference it — the
    framework handles injection transparently on top of the existing dynamics.
    """

    kind:     Literal["nl_oscillator"] = "nl_oscillator"
    position: Annotated[float, IntegratedField()] = 1.0
    velocity: Annotated[float, IntegratedField()] = 0.0
    omega:    Annotated[float, ParameterField()]  = 1.0   # natural frequency [rad/s]
    c0:       Annotated[float, ParameterField()]  = 0.1   # linear damping coefficient
    c1:       Annotated[float, ParameterField()]  = 1.0   # position-dependent damping coefficient
    force:    Annotated[float, ExcitationPort(
                  targets   = "velocity",
                  port_type = "effort",
                  units     = "N",
              )] = 0.0
