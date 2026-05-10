from typing import ClassVar, Literal

import numpy as np

from numen.compiler.flatten import CompiledSpec, CompiledSystem
from numen.fields import EntityGroup
from numen.spec.system import System, DynamicsFn
from components import MassComponent, SpringComponent


def mass_kinematics_dynamics(
    dx: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
    t: float,
    spec: CompiledSpec,
    system: CompiledSystem[tuple[str]],
) -> None:
    """ẋ = v for every mass entity. group_size=1 so each group is a single entity."""
    for (entity_id,) in system.entity_groups:
        c  = spec.view(entity_id, MassComponent, x, p)
        dc = spec.dx_view(entity_id, MassComponent, dx)
        dc.position += c.velocity


def spring_force_dynamics(
    dx: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
    t: float,
    spec: CompiledSpec,
    system: CompiledSystem[tuple[str, str, str]],
) -> None:
    """Hooke's law spring force. group_size=3: each group is [mass_a, spring, mass_b].

    Forces accumulate additively (+=) so multiple springs sharing a mass are correct.
    """
    for id_a, id_s, id_b in system.entity_groups:
        a = spec.view(id_a, MassComponent,   x, p)
        b = spec.view(id_b, MassComponent,   x, p)
        s = spec.view(id_s, SpringComponent, x, p)

        stretch = (b.position - a.position) - s.rest_length
        force   = s.k * stretch

        da = spec.dx_view(id_a, MassComponent, dx)
        db = spec.dx_view(id_b, MassComponent, dx)
        da.velocity +=  force / a.mass
        db.velocity += -force / b.mass


class MassKinematicsSystem(System):
    """Integrates position for all MassComponent entities (auto-populated, group_size=1)."""

    component_types: ClassVar[tuple[type, ...]] = (MassComponent,)
    python_fn:       ClassVar[DynamicsFn]       = staticmethod(mass_kinematics_dynamics)
    kind:            Literal["mass_kinematics"] = "mass_kinematics"
    dynamics_fn:     str = "SpringDynamics.mass_kinematics_dynamics!"


class SpringForceSystem(System):
    """Applies spring forces for an explicit set of (mass_a, spring, mass_b) connections.

    Topology is declared here, not in components. Add connections via entity_groups.
    entity_slots drives both compile-time type validation and group_size=3 in CompiledSystem.

    Example:
        SpringForceSystem(entity_groups=[
            ["m1", "s1", "m2"],
            ["m2", "s2", "m3"],
        ])
    """

    component_types: ClassVar[tuple[type, ...]] = ()
    entity_slots:    ClassVar[EntityGroup]      = EntityGroup(MassComponent, SpringComponent, MassComponent)
    python_fn:       ClassVar[DynamicsFn]       = staticmethod(spring_force_dynamics)
    kind:            Literal["spring_force"]    = "spring_force"
    dynamics_fn:     str = "SpringDynamics.spring_force_dynamics!"
