from typing import ClassVar, Literal

import jax.numpy as jnp
import numpy as np

from numen.compiler.flatten import CompiledSpec, CompiledSystem  # noqa: F401
from numen.spec.system import System, DynamicsFn
from components import NLOscillatorComponent


def nl_oscillator_dynamics(
    dx: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
    t: float,
    spec: CompiledSpec,
    system: CompiledSystem[tuple[str]],
) -> None:
    """ẋ = v,  v̇ = -(c0 + c1·x²)·v - ω²·x"""
    for (entity_id,) in system.entity_groups:
        c  = spec.view(entity_id, NLOscillatorComponent, x, p)
        dc = spec.dx_view(entity_id, NLOscillatorComponent, dx)
        effective_damping = c.c0 + c.c1 * c.position ** 2
        dc.position += c.velocity
        dc.velocity += -effective_damping * c.velocity - c.omega ** 2 * c.position


class NLOscillatorSystem(System):
    component_types: ClassVar[tuple[type, ...]] = (NLOscillatorComponent,)
    python_fn:       ClassVar[DynamicsFn]       = staticmethod(nl_oscillator_dynamics)
    kind:            Literal["nl_oscillator_system"] = "nl_oscillator_system"
    dynamics_fn:     str = "NLOscillatorDynamics.nl_oscillator_dynamics!"
