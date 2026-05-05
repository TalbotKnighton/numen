from typing import ClassVar, Literal

import numpy as np

from numen.compiler.flatten import CompiledSpec, CompiledSystem  # noqa: F401
from numen.spec.system import System, DynamicsFn
from components import OscillatorComponent


def oscillator_dynamics(
    dx: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
    t: float,
    spec: CompiledSpec,
    system: CompiledSystem[tuple[str]],
) -> None:
    """Harmonic oscillator dynamics: ẋ = v, v̇ = -ω²x - 2ζω·v"""
    for (entity_id,) in system.entity_groups:
        c  = spec.view(entity_id, OscillatorComponent, x, p)
        dc = spec.dx_view(entity_id, OscillatorComponent, dx)
        dc.position += c.velocity
        dc.velocity += -c.omega**2 * c.position - 2 * c.damping * c.omega * c.velocity


class OscillatorSystem(System):
    component_types: ClassVar[tuple[type, ...]] = (OscillatorComponent,)
    python_fn:       ClassVar[DynamicsFn]       = staticmethod(oscillator_dynamics)
    kind:            Literal["oscillator_system"] = "oscillator_system"
    dynamics_fn:     str = "OscillatorDynamics.oscillator_dynamics!"
