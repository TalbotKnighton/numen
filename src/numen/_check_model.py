"""Minimal oscillator model used by `numen check`.  Not for external use."""
from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from numen.spec.component import Component
from numen.fields import IntegratedField, ParameterField
from numen.spec.system import System, DynamicsFn
from numen.spec.world import GenericWorld
import jax.numpy as jnp


class _CheckOsc(Component):
    kind:     Literal["_check_osc"] = "_check_osc"
    position: Annotated[float, IntegratedField()] = 1.0
    velocity: Annotated[float, IntegratedField()] = 0.0
    omega:    Annotated[float, ParameterField()]  = 6.2832   # 1 Hz


def _check_dyn(dx, x, p, t, spec, system):
    for (eid,) in system.entity_groups:
        c  = spec.view(eid, _CheckOsc, x, p)
        dc = spec.dx_view(eid, _CheckOsc, dx)
        dc.position += c.velocity
        dc.velocity += -(c.omega ** 2) * c.position


def _check_dyn_jax(dx, x, p, t, spec, system):
    for (eid,) in system.entity_groups:
        c  = spec.view(eid, _CheckOsc, x, p)
        dc = spec.dx_view(eid, _CheckOsc, dx)
        dc.position += c.velocity
        dc.velocity += -(c.omega ** 2) * c.position


class _CheckOscSys(System):
    component_types: ClassVar[tuple[type, ...]] = (_CheckOsc,)
    python_fn:       ClassVar[DynamicsFn]       = staticmethod(_check_dyn)
    kind:            Literal["_check_osc_sys"]  = "_check_osc_sys"
    dynamics_fn:     str = ""


class _CheckOscSysJax(System):
    component_types: ClassVar[tuple[type, ...]] = (_CheckOsc,)
    python_fn:       ClassVar[DynamicsFn]       = staticmethod(_check_dyn_jax)
    kind:            Literal["_check_osc_jax"]  = "_check_osc_jax"
    dynamics_fn:     str = ""


CheckWorld      = GenericWorld[_CheckOsc, _CheckOscSys,    None]
CheckWorldJax   = GenericWorld[_CheckOsc, _CheckOscSysJax, None]
