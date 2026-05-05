from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Generic, Protocol, TypeVar
from pydantic import BaseModel

import numpy as np

from numen.fields import EntityGroup

if TYPE_CHECKING:
    from numen.compiler.flatten import CompiledSpec, CompiledSystem

GroupT = TypeVar("GroupT", bound=tuple)


class DynamicsFn(Protocol[GroupT]):
    """Calling convention for all Python dynamics functions.

    Generic over GroupT — the tuple type of one entity group — so the type checker
    knows the group arity at each call site:
        CompiledSystem[tuple[str]]           → each group is one entity
        CompiledSystem[tuple[str, str, str]] → each group is three entities
    """

    def __call__(
        self,
        dx: np.ndarray,
        x: np.ndarray,
        p: np.ndarray,
        t: float,
        spec: CompiledSpec,
        system: CompiledSystem[GroupT],
    ) -> None: ...


class System(BaseModel):
    """Base class for all ECS systems.

    Class variables (statically declared, not serialized):
        component_types: Component subclasses this system auto-queries.
                         compile_spec finds all matching entities and populates
                         entity_ids. Leave empty when using entity_groups instead.
        entity_slots:    Declares slot types and group size for multi-slot systems.
                         compile_spec validates entity_groups against slot_types and
                         stores size in CompiledSystem.group_size.
        python_fn:       Python dynamics callable for the scipy backend.

    Pydantic fields (serialized):
        dynamics_fn:   Julia function reference, e.g. "MyModule.my_fn!".
        entity_ids:    Flat entity list for single-type systems (auto-populated from
                       component_types, or set manually to restrict the set).
        entity_groups: Explicit connection topology for multi-slot systems. Each inner
                       list is one group; types validated against entity_slots.slot_types.
                       compile_spec flattens and caches groups on CompiledSystem.

    Single-type system (auto-populated):
        class MassKinematicsSystem(System):
            component_types: ClassVar[tuple[type, ...]] = (MassComponent,)
            python_fn:       ClassVar[DynamicsFn]       = staticmethod(mass_kinematics_dynamics)
            kind:            Literal["mass_kinematics"] = "mass_kinematics"
            dynamics_fn:     str = "Dynamics.mass_kinematics!"

    Multi-slot coupled system (explicit topology):
        class SpringForceSystem(System):
            entity_slots: ClassVar[EntityGroup] = EntityGroup(MassComponent, SpringComponent, MassComponent)
            python_fn:    ClassVar[DynamicsFn]  = staticmethod(spring_force_dynamics)
            kind:         Literal["spring_force"] = "spring_force"
            dynamics_fn:  str = "Dynamics.spring_force!"

        SpringForceSystem(entity_groups=[["m1","s1","m2"], ["m2","s2","m3"]])
    """

    component_types: ClassVar[tuple[type, ...]] = ()
    entity_slots:    ClassVar[EntityGroup | None] = None
    python_fn:       ClassVar[DynamicsFn | None]  = None

    dynamics_fn:   str             = ""
    entity_ids:    list[str]       = []
    entity_groups: list[list[str]] = []
    model_config = {"frozen": True}
