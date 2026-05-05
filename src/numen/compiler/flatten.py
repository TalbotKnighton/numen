from __future__ import annotations

import typing
from dataclasses import dataclass, field, asdict
from typing import Any, Generic, TypeVar, get_type_hints, get_args, get_origin

GroupT = TypeVar("GroupT", bound=tuple)
CT     = TypeVar("CT")

import numpy as np

from numen.fields import IntegratedField, ContinuousField, DiscreteField, ParameterField, EntityGroup


class DxBuffer:
    """Derivative accumulator compatible with both NumPy (in-place) and JAX (functional).

    NumPy: ``dx[s] = value`` mutates in place.
    JAX:   ``dx[s] = value`` calls ``arr.at[s].set(value)`` and stores the new array —
           preserving JAX's functional semantics while keeping the same user-facing API.

    Dynamics functions never touch ``dx`` directly; they always go through
    ``spec.dx_view()`` which wraps ``dx`` in a ``DerivativeView``.  The buffer is
    created by the backend before each RHS call, so it is always fresh.
    """

    __slots__ = ("_arr",)

    def __init__(self, arr: Any) -> None:
        object.__setattr__(self, "_arr", arr)

    def __getitem__(self, key: Any) -> Any:
        return object.__getattribute__(self, "_arr")[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        arr = object.__getattribute__(self, "_arr")
        if hasattr(arr, "at"):
            object.__setattr__(self, "_arr", arr.at[key].set(value))
        else:
            arr[key] = value

    @property
    def array(self) -> Any:
        return object.__getattribute__(self, "_arr")


class ComponentView:
    """Read-only accessor for a single entity's component fields during a dynamics call.

    Attribute reads resolve to values in the flat state/param arrays.
    Raises AttributeError with a clear message if an unknown field is accessed.
    """

    __slots__ = ("_entity_id", "_component_type", "_x", "_p", "_spec")

    def __init__(
        self,
        entity_id: str,
        component_type: type,
        x: np.ndarray,
        p: np.ndarray,
        spec: CompiledSpec,
    ) -> None:
        object.__setattr__(self, "_entity_id", entity_id)
        object.__setattr__(self, "_component_type", component_type)
        object.__setattr__(self, "_x", x)
        object.__setattr__(self, "_p", p)
        object.__setattr__(self, "_spec", spec)

    def __getattr__(self, name: str) -> Any:
        entity_id = object.__getattribute__(self, "_entity_id")
        comp_type = object.__getattribute__(self, "_component_type")
        x         = object.__getattribute__(self, "_x")
        p         = object.__getattribute__(self, "_p")
        spec      = object.__getattribute__(self, "_spec")
        key = f"{entity_id}.{name}"
        if key in spec.state_index_map:
            s, e = spec.state_index_map[key]
            return x[s] if e - s == 1 else x[s:e]
        if key in spec.param_index_map:
            s, e = spec.param_index_map[key]
            return p[s] if e - s == 1 else p[s:e]
        raise AttributeError(
            f"{comp_type.__name__} entity '{entity_id}' has no field '{name}'. "
            f"Available state fields: {[k.split('.', 1)[1] for k in spec.state_index_map if k.startswith(entity_id + '.')]}, "
            f"param fields: {[k.split('.', 1)[1] for k in spec.param_index_map if k.startswith(entity_id + '.')]}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("ComponentView is read-only. Use DerivativeView to write derivatives.")


class DerivativeView:
    """Write accessor for a single entity's derivative slots during a dynamics call.

    Attribute assignment writes into the flat dx array at the correct index.
    Only state fields (IntegratedField, DiscreteField, ContinuousField) are writable —
    attempting to write a ParameterField raises AttributeError.
    """

    __slots__ = ("_entity_id", "_component_type", "_dx", "_spec")

    def __init__(
        self,
        entity_id: str,
        component_type: type,
        dx: np.ndarray,
        spec: CompiledSpec,
    ) -> None:
        object.__setattr__(self, "_entity_id", entity_id)
        object.__setattr__(self, "_component_type", component_type)
        object.__setattr__(self, "_dx", dx)
        object.__setattr__(self, "_spec", spec)

    def __setattr__(self, name: str, value: Any) -> None:
        entity_id  = object.__getattribute__(self, "_entity_id")
        comp_type  = object.__getattribute__(self, "_component_type")
        dx         = object.__getattribute__(self, "_dx")
        spec       = object.__getattribute__(self, "_spec")
        key = f"{entity_id}.{name}"
        if key in spec.state_index_map:
            s, e = spec.state_index_map[key]
            if e - s == 1:
                dx[s] = value
            else:
                dx[s:e] = value
            return
        if key in spec.param_index_map:
            raise AttributeError(
                f"{comp_type.__name__} field '{name}' is a ParameterField (constant) — "
                f"derivatives cannot be assigned to parameters."
            )
        raise AttributeError(
            f"{comp_type.__name__} entity '{entity_id}' has no state field '{name}'. "
            f"Available: {[k.split('.', 1)[1] for k in spec.state_index_map if k.startswith(entity_id + '.')]}"
        )

    def __getattr__(self, name: str) -> Any:
        entity_id = object.__getattribute__(self, "_entity_id")
        dx        = object.__getattribute__(self, "_dx")
        spec      = object.__getattribute__(self, "_spec")
        key = f"{entity_id}.{name}"
        if key in spec.state_index_map:
            s, e = spec.state_index_map[key]
            return dx[s] if e - s == 1 else dx[s:e]
        raise AttributeError(
            f"DerivativeView has no slot '{name}'. "
            f"Available: {[k.split('.', 1)[1] for k in spec.state_index_map if k.startswith(entity_id + '.')]}"
        )


@dataclass
class CompiledSystem(Generic[GroupT]):
    dynamics_fn:   str
    entity_ids:    list[str]
    group_size:    int                    = 1   # entities per dynamics invocation
    entity_groups: tuple[GroupT, ...]     = ()  # pre-grouped, immutable; not serialized
    python_fn:     Any                    = field(default=None, repr=False)  # not serialized


@dataclass
class CompiledSpec:
    state_size:       int
    param_size:       int
    state_index_map:  dict[str, tuple[int, int]]   # "entity.field" → (start, end) into x
    param_index_map:  dict[str, tuple[int, int]]   # "entity.field" → (start, end) into p
    discrete_dts:     list[float]
    x0:               list[float]
    p:                list[float]
    systems:          list[CompiledSystem] = field(default_factory=list)

    def view(
        self,
        entity_id: str,
        component_type: type[CT],
        x: np.ndarray,
        p: np.ndarray,
    ) -> CT:
        """Read-only accessor for an entity's fields. Use inside dynamics functions."""
        return ComponentView(entity_id, component_type, x, p, self)  # type: ignore[return-value]

    def dx_view(
        self,
        entity_id: str,
        component_type: type[CT],
        dx: "DxBuffer | np.ndarray",
    ) -> CT:
        """Write accessor for an entity's derivative slots. Use inside dynamics functions."""
        return DerivativeView(entity_id, component_type, dx, self)  # type: ignore[return-value]

    # --- Low-level index accessors (for advanced use / Julia interop) ---

    def state_idx(self, key: str) -> int:
        return self.state_index_map[key][0]

    def param_idx(self, key: str) -> int:
        return self.param_index_map[key][0]

    def state_slice(self, key: str) -> slice:
        s, e = self.state_index_map[key]
        return slice(s, e)

    def param_slice(self, key: str) -> slice:
        s, e = self.param_index_map[key]
        return slice(s, e)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_size":      self.state_size,
            "param_size":      self.param_size,
            "state_index_map": {k: list(v) for k, v in self.state_index_map.items()},
            "param_index_map": {k: list(v) for k, v in self.param_index_map.items()},
            "discrete_dts":    self.discrete_dts,
            "x0":              self.x0,
            "p":               self.p,
            "systems": [
                {"dynamics_fn": s.dynamics_fn, "entity_ids": s.entity_ids, "group_size": s.group_size}
                for s in self.systems
            ],
        }


def _get_numen_fields(component: Any) -> list[tuple[str, Any, Any]]:
    hints = get_type_hints(type(component), include_extras=True)
    results = []
    for name, hint in hints.items():
        if get_origin(hint) is not typing.Annotated:
            continue
        args = get_args(hint)
        for meta in args[1:]:
            if isinstance(meta, (IntegratedField, ContinuousField, DiscreteField, ParameterField)):
                results.append((name, meta, getattr(component, name, 0.0)))
    return results


def _validate_group(
    world: Any,
    dynamics_fn: str,
    group: list[str],
    slot_types: tuple[type, ...],
) -> None:
    """Validate one entity group against its declared slot types."""
    if len(group) != len(slot_types):
        raise ValueError(
            f"System '{dynamics_fn}': group {group!r} has {len(group)} entries "
            f"but entity_slot_types declares {len(slot_types)} slots"
        )
    for eid, expected in zip(group, slot_types):
        comp = world.components.get(eid)
        if comp is None:
            raise ValueError(
                f"System '{dynamics_fn}': entity '{eid}' not found in world"
            )
        if not isinstance(comp, expected):
            raise TypeError(
                f"System '{dynamics_fn}': slot for {expected.__name__!r} "
                f"received entity '{eid}' of type {type(comp).__name__!r}"
            )


def compile_spec(world: Any) -> CompiledSpec:
    """Walk a GenericWorld and produce a flat CompiledSpec for the solver."""
    state_cursor = 0
    param_cursor = 0
    state_index_map: dict[str, tuple[int, int]] = {}
    param_index_map: dict[str, tuple[int, int]] = {}
    x0: list[float] = []
    p:  list[float] = []
    discrete_dts: set[float] = set()

    for entity_id, component in world.components.items():
        for field_name, meta, value in _get_numen_fields(component):
            key = f"{entity_id}.{field_name}"
            size = meta.size
            values = [value] if size == 1 else list(value)

            if isinstance(meta, ParameterField):
                param_index_map[key] = (param_cursor, param_cursor + size)
                p.extend(values)
                param_cursor += size
            else:
                state_index_map[key] = (state_cursor, state_cursor + size)
                x0.extend(values)
                state_cursor += size
                if isinstance(meta, DiscreteField) and meta.dt > 0:
                    discrete_dts.add(meta.dt)

    systems = []
    for sys_model in (world.systems or {}).values():
        if sys_model is None or not sys_model.dynamics_fn:
            continue

        comp_types   = type(sys_model).component_types
        entity_slots = type(sys_model).entity_slots   # EntityGroup | None
        group_size   = entity_slots.size if entity_slots is not None else 1

        if sys_model.entity_groups:
            if entity_slots is None:
                raise ValueError(
                    f"System '{sys_model.dynamics_fn}': entity_groups requires "
                    f"entity_slots to be declared on the System class"
                )
            for group in sys_model.entity_groups:
                _validate_group(world, sys_model.dynamics_fn, group, entity_slots.slot_types)
            entity_ids = [eid for group in sys_model.entity_groups for eid in group]

        elif sys_model.entity_ids:
            if comp_types:
                for eid in sys_model.entity_ids:
                    comp = world.components.get(eid)
                    if comp is None:
                        raise ValueError(
                            f"System '{sys_model.dynamics_fn}': entity '{eid}' not found in world"
                        )
                    if not isinstance(comp, comp_types):
                        names = ", ".join(t.__name__ for t in comp_types)
                        raise TypeError(
                            f"System '{sys_model.dynamics_fn}': entity '{eid}' has type "
                            f"{type(comp).__name__!r}, expected one of ({names})"
                        )
            entity_ids = list(sys_model.entity_ids)

        elif comp_types:
            entity_ids = [
                eid for eid, comp in world.components.items()
                if isinstance(comp, comp_types)
            ]
            if not entity_ids:
                names = ", ".join(t.__name__ for t in comp_types)
                raise ValueError(
                    f"System '{sys_model.dynamics_fn}': no entities matching ({names}) found in world"
                )
        else:
            raise ValueError(
                f"System '{sys_model.dynamics_fn}': must declare 'component_types' "
                f"or provide 'entity_ids' / 'entity_groups'"
            )

        gs = group_size
        groups = tuple(
            tuple(entity_ids[i:i + gs]) for i in range(0, len(entity_ids), gs)
        )
        python_fn = type(sys_model).python_fn
        systems.append(CompiledSystem(
            dynamics_fn=sys_model.dynamics_fn,
            entity_ids=entity_ids,
            group_size=gs,
            entity_groups=groups,
            python_fn=python_fn,
        ))

    return CompiledSpec(
        state_size=state_cursor,
        param_size=param_cursor,
        state_index_map=state_index_map,
        param_index_map=param_index_map,
        discrete_dts=sorted(discrete_dts),
        x0=x0,
        p=p,
        systems=systems,
    )
