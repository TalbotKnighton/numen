"""Compiler that flattens a GenericWorld into a flat CompiledSpec for solvers.

The central function ``compile_spec(world)`` performs a single pass over the
world's entities and systems to produce a ``CompiledSpec`` — the canonical
representation passed to every backend:

- Builds flat ``state_index_map`` and ``param_index_map`` dictionaries mapping
  ``"entity_id.component_kind.field_name"`` keys to ``(start, end)`` index pairs.
- Allocates initial state vector ``x0`` and parameter vector ``p``.
- Computes the ``differential_mask`` (1.0 for integrated slots, 0.0 for
  algebraic ``ContinuousField`` slots).
- Detects required features (``vector_fields``, ``discrete_fields``,
  ``continuous_fields``, ``dae_constraints``, ``control_callbacks``).
- Validates system entity groups against declared slot types.
- Computes a conservative Jacobian sparsity pattern (COO format) from the
  ECS entity-group graph for sparse ForwardDiff in Julia.

Key classes:
    ``CompiledSpec``    — the compiled flat representation
    ``CompiledSystem``  — per-system compiled data (entity ids, group size, python_fn)
    ``DxBuffer``        — derivative accumulator for both NumPy and JAX
    ``ComponentView``   — read-only accessor for entity fields inside dynamics
    ``DerivativeView``  — write accessor for derivative slots inside dynamics
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field, asdict
from typing import Any, Generic, TypeVar, get_type_hints, get_args, get_origin

GroupT = TypeVar("GroupT", bound=tuple)
CT     = TypeVar("CT")

import numpy as np

from numen.fields import IntegratedField, ContinuousField, DiscreteField, ParameterField, ExcitationPort, EntityGroup


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


def _kind_of(component_type: type) -> str:
    """Extract the kind literal value from a Component subclass.

    Reads ``ComponentType.model_fields['kind'].default`` (Pydantic v2).
    """
    if hasattr(component_type, "model_fields") and "kind" in component_type.model_fields:
        default = component_type.model_fields["kind"].default
        if default is not None:
            return str(default)
    raise ValueError(
        f"Cannot determine 'kind' for component type {component_type.__name__}. "
        f"Ensure it has a 'kind: Literal[\"...\"]' field with a default value."
    )


class ComponentView:
    """Read-only accessor for a single entity's component fields during a dynamics call.

    Attribute reads resolve to values in the flat state/param arrays.
    Keys in the spec index maps use the full path ``entity_id.component_kind.field_name``.
    Raises AttributeError with a clear message if an unknown field is accessed.
    """

    __slots__ = ("_entity_id", "_component_kind", "_component_type", "_x", "_p", "_spec")

    def __init__(
        self,
        entity_id: str,
        component_kind: str,
        component_type: type,
        x: np.ndarray,
        p: np.ndarray,
        spec: CompiledSpec,
    ) -> None:
        object.__setattr__(self, "_entity_id",      entity_id)
        object.__setattr__(self, "_component_kind", component_kind)
        object.__setattr__(self, "_component_type", component_type)
        object.__setattr__(self, "_x",              x)
        object.__setattr__(self, "_p",              p)
        object.__setattr__(self, "_spec",           spec)

    def __getattr__(self, name: str) -> Any:
        entity_id      = object.__getattribute__(self, "_entity_id")
        component_kind = object.__getattribute__(self, "_component_kind")
        comp_type      = object.__getattribute__(self, "_component_type")
        x              = object.__getattribute__(self, "_x")
        p              = object.__getattribute__(self, "_p")
        spec           = object.__getattribute__(self, "_spec")
        key = f"{entity_id}.{component_kind}.{name}"
        if key in spec.state_index_map:
            s, e = spec.state_index_map[key]
            return x[s] if e - s == 1 else x[s:e]
        if key in spec.param_index_map:
            s, e = spec.param_index_map[key]
            return p[s] if e - s == 1 else p[s:e]
        prefix = f"{entity_id}.{component_kind}."
        raise AttributeError(
            f"{comp_type.__name__} entity '{entity_id}' has no field '{name}'. "
            f"Available state fields: {[k[len(prefix):] for k in spec.state_index_map if k.startswith(prefix)]}, "
            f"param fields: {[k[len(prefix):] for k in spec.param_index_map if k.startswith(prefix)]}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("ComponentView is read-only. Use DerivativeView to write derivatives.")


class DerivativeView:
    """Write accessor for a single entity's derivative slots during a dynamics call.

    Attribute assignment writes into the flat dx array at the correct index.
    Only state fields (IntegratedField, DiscreteField, ContinuousField) are writable —
    attempting to write a ParameterField raises AttributeError.
    """

    __slots__ = ("_entity_id", "_component_kind", "_component_type", "_dx", "_spec")

    def __init__(
        self,
        entity_id: str,
        component_kind: str,
        component_type: type,
        dx: np.ndarray,
        spec: CompiledSpec,
    ) -> None:
        object.__setattr__(self, "_entity_id",      entity_id)
        object.__setattr__(self, "_component_kind", component_kind)
        object.__setattr__(self, "_component_type", component_type)
        object.__setattr__(self, "_dx",             dx)
        object.__setattr__(self, "_spec",           spec)

    def __setattr__(self, name: str, value: Any) -> None:
        entity_id      = object.__getattribute__(self, "_entity_id")
        component_kind = object.__getattribute__(self, "_component_kind")
        comp_type      = object.__getattribute__(self, "_component_type")
        dx             = object.__getattribute__(self, "_dx")
        spec           = object.__getattribute__(self, "_spec")
        key = f"{entity_id}.{component_kind}.{name}"
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
        prefix = f"{entity_id}.{component_kind}."
        raise AttributeError(
            f"{comp_type.__name__} entity '{entity_id}' has no state field '{name}'. "
            f"Available: {[k[len(prefix):] for k in spec.state_index_map if k.startswith(prefix)]}"
        )

    def __getattr__(self, name: str) -> Any:
        entity_id      = object.__getattribute__(self, "_entity_id")
        component_kind = object.__getattribute__(self, "_component_kind")
        dx             = object.__getattribute__(self, "_dx")
        spec           = object.__getattribute__(self, "_spec")
        key = f"{entity_id}.{component_kind}.{name}"
        if key in spec.state_index_map:
            s, e = spec.state_index_map[key]
            return dx[s] if e - s == 1 else dx[s:e]
        prefix = f"{entity_id}.{component_kind}."
        raise AttributeError(
            f"DerivativeView has no slot '{name}'. "
            f"Available: {[k[len(prefix):] for k in spec.state_index_map if k.startswith(prefix)]}"
        )


@dataclass
class CompiledSystem(Generic[GroupT]):
    """Compiled representation of one ECS system, ready for solver dispatch.

    Attributes:
        dynamics_fn:   Julia function reference string, e.g. ``"Module.fn!"``.
        entity_ids:    Flat list of entity IDs operated on by this system.
                       For multi-slot systems: [a0, s0, b0, a1, s1, b1, …]
                       where ``group_size`` = 3.
        group_size:    Number of entity IDs per dynamics invocation. 1 for
                       single-entity systems; > 1 for coupled systems declared
                       with ``entity_slots``.
        entity_groups: Pre-grouped tuple of entity ID tuples, derived from
                       ``entity_ids`` and ``group_size``. Not serialized to JSON.
        python_fn:     Python callable for scipy/JAX backends. Not serialized.
    """
    dynamics_fn:   str
    entity_ids:    list[str]
    group_size:    int                    = 1
    entity_groups: tuple[GroupT, ...]     = ()
    python_fn:     Any                    = field(default=None, repr=False)


@dataclass
class CompiledCallback:
    """A compiled controller callback — one entry per world.callbacks item."""
    name:      str                  # world key
    dt:        float                # controller period (s)
    julia_fn:  str                  # "Module.fn!" resolved in Julia scope
    params:    dict[str, float]     # passed to both python_fn and Julia callback
    python_fn: Any = field(default=None, repr=False)  # not serialized


@dataclass
class CompiledSpec:
    """Flat representation of a compiled world, consumed by all solver backends.

    Produced by ``compile_spec(world)``. Never construct manually.

    Attributes:
        state_size:        Total number of state slots in ``x``.
        param_size:        Total number of parameter slots in ``p``.
        state_index_map:   Maps ``"entity.kind.field"`` to ``(start, end)`` in ``x``.
        param_index_map:   Maps ``"entity.kind.field"`` to ``(start, end)`` in ``p``.
        discrete_dts:      Sorted list of unique ``DiscreteField.dt`` values.
        x0:                Initial state vector (length ``state_size``).
        p:                 Parameter vector (length ``param_size``).
        differential_mask: 1.0 for ``IntegratedField``/``DiscreteField`` slots;
                           0.0 for ``ContinuousField(algebraic=True)`` slots.
                           Length equals ``state_size``. All-ones means pure ODE.
                           Julia passes ``Diagonal(differential_mask)`` as the
                           mass matrix to ``ODEFunction`` for DAE problems.
        systems:           Compiled systems in world-definition order.
        compiled_callbacks: Compiled controller callbacks.
        required_features: Feature strings set by ``compile_spec`` based on the
                           field types present. Checked against each backend's
                           ``supported_features`` before solving.
        jac_sparsity_rows: Row indices of the Jacobian sparsity pattern (0-based,
                           COO format). Julia converts to 1-based SparseMatrixCSC
                           for ``ODEFunction`` sparse coloring.
        jac_sparsity_cols: Column indices of the Jacobian sparsity pattern.
    """
    state_size:         int
    param_size:         int
    state_index_map:    dict[str, tuple[int, int]]
    param_index_map:    dict[str, tuple[int, int]]
    discrete_dts:       list[float]
    x0:                 list[float]
    p:                  list[float]
    differential_mask:  list[float]      = field(default_factory=list)
    systems:            list[CompiledSystem]   = field(default_factory=list)
    compiled_callbacks: list[CompiledCallback] = field(default_factory=list)
    required_features:  frozenset[str]         = field(default_factory=frozenset)
    jac_sparsity_rows:  list[int]        = field(default_factory=list)
    jac_sparsity_cols:  list[int]        = field(default_factory=list)

    def view(
        self,
        entity_id: str,
        component_type: type[CT],
        x: np.ndarray,
        p: np.ndarray,
    ) -> CT:
        """Read-only accessor for an entity's component fields. Use inside dynamics functions.

        The component_type's ``kind`` literal is used to construct the key prefix
        ``entity_id.kind.*`` for all field lookups.
        """
        kind = _kind_of(component_type)
        return ComponentView(entity_id, kind, component_type, x, p, self)  # type: ignore[return-value]

    def dx_view(
        self,
        entity_id: str,
        component_type: type[CT],
        dx: "DxBuffer | np.ndarray",
    ) -> CT:
        """Write accessor for an entity's derivative slots. Use inside dynamics functions."""
        kind = _kind_of(component_type)
        return DerivativeView(entity_id, kind, component_type, dx, self)  # type: ignore[return-value]

    # --- Low-level index accessors (for advanced use / Julia interop) ---

    def state_idx(self, key: str) -> int:
        """Return the start index for a state field.

        Args:
            key: Full path ``"entity_id.component_kind.field_name"``.

        Returns:
            Integer index into the flat state vector ``x``.
        """
        return self.state_index_map[key][0]

    def param_idx(self, key: str) -> int:
        """Return the start index for a parameter field.

        Args:
            key: Full path ``"entity_id.component_kind.field_name"``.

        Returns:
            Integer index into the flat parameter vector ``p``.
        """
        return self.param_index_map[key][0]

    def state_slice(self, key: str) -> slice:
        """Return a slice for a (possibly vector) state field.

        Args:
            key: Full path ``"entity_id.component_kind.field_name"``.

        Returns:
            ``slice(start, end)`` for use with ``x[spec.state_slice(key)]``.
        """
        s, e = self.state_index_map[key]
        return slice(s, e)

    def param_slice(self, key: str) -> slice:
        """Return a slice for a (possibly vector) parameter field.

        Args:
            key: Full path ``"entity_id.component_kind.field_name"``.

        Returns:
            ``slice(start, end)`` for use with ``p[spec.param_slice(key)]``.
        """
        s, e = self.param_index_map[key]
        return slice(s, e)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_size":        self.state_size,
            "param_size":        self.param_size,
            "state_index_map":   {k: list(v) for k, v in self.state_index_map.items()},
            "param_index_map":   {k: list(v) for k, v in self.param_index_map.items()},
            "discrete_dts":      self.discrete_dts,
            "x0":                self.x0,
            "p":                 self.p,
            "differential_mask": self.differential_mask,
            "systems": [
                {"dynamics_fn": s.dynamics_fn, "entity_ids": s.entity_ids, "group_size": s.group_size}
                for s in self.systems
            ],
            "callbacks": [
                {"name": c.name, "dt": c.dt, "julia_fn": c.julia_fn, "params": c.params}
                for c in self.compiled_callbacks
            ],
            "jac_sparsity_rows": self.jac_sparsity_rows,
            "jac_sparsity_cols": self.jac_sparsity_cols,
        }


def _get_numen_fields(component: Any) -> list[tuple[str, Any, Any]]:
    hints = get_type_hints(type(component), include_extras=True)
    results = []
    for name, hint in hints.items():
        if get_origin(hint) is not typing.Annotated:
            continue
        args = get_args(hint)
        for meta in args[1:]:
            if isinstance(meta, (IntegratedField, ContinuousField, DiscreteField, ParameterField, ExcitationPort)):
                results.append((name, meta, getattr(component, name, 0.0)))
    return results


def _has_component_of_type(world: Any, entity_id: str, comp_types: tuple[type, ...]) -> bool:
    comps = world.components.get(entity_id, {})
    return any(isinstance(c, comp_types) for c in comps.values())


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
        comps = world.components.get(eid)
        if comps is None:
            raise ValueError(
                f"System '{dynamics_fn}': entity '{eid}' not found in world"
            )
        if not any(isinstance(c, expected) for c in comps.values()):
            found = [type(c).__name__ for c in comps.values()]
            raise TypeError(
                f"System '{dynamics_fn}': slot for {expected.__name__!r} "
                f"received entity '{eid}' with components {found!r}"
            )


def _compute_jac_sparsity(
    state_index_map: dict[str, tuple[int, int]],
    systems: list[CompiledSystem],
) -> tuple[list[int], list[int]]:
    """Compute Jacobian sparsity pattern from the ECS entity-group graph.

    Strategy: for each system, collect all state indices for every entity in
    each group and mark every (row, col) pair within that set as potentially
    nonzero.  This is conservative — it over-includes when a system only reads
    a subset of an entity's fields — but it is always safe (never misses a
    real nonzero entry for well-formed dynamics).

    Entities with no state variables (e.g. synthetic excitation entities) contribute
    nothing, so the excitation systems correctly add no Jacobian entries.

    Returns 0-based (rows, cols) lists for direct inclusion in the JSON payload
    (Julia converts to 1-based before constructing the SparseMatrixCSC).
    """
    nonzero: set[tuple[int, int]] = set()
    for sys in systems:
        gs = sys.group_size
        eids = sys.entity_ids
        for g_start in range(0, len(eids), gs):
            group = eids[g_start : g_start + gs]
            idxs: list[int] = []
            for eid in group:
                prefix = eid + "."
                for key, (s, e) in state_index_map.items():
                    if key.startswith(prefix):
                        idxs.extend(range(s, e))
            for row in idxs:
                for col in idxs:
                    nonzero.add((row, col))
    if not nonzero:
        return [], []
    pairs = sorted(nonzero)
    return [r for r, _ in pairs], [c for _, c in pairs]


def compile_spec(world: Any) -> CompiledSpec:
    """Walk a GenericWorld and produce a flat CompiledSpec for the solver.

    State/param keys use the full path ``entity_id.component_kind.field_name``.
    ExcitationPort-annotated fields are skipped (pure metadata for the
    characterization framework; not compiled into p).
    """
    state_cursor = 0
    param_cursor = 0
    state_index_map: dict[str, tuple[int, int]] = {}
    param_index_map: dict[str, tuple[int, int]] = {}
    x0:               list[float] = []
    p:                list[float] = []
    differential_mask: list[float] = []
    discrete_dts: set[float] = set()
    features: set[str] = set()

    for entity_id, comps_by_kind in world.components.items():
        for kind, component in comps_by_kind.items():
            for field_name, meta, value in _get_numen_fields(component):
                key  = f"{entity_id}.{kind}.{field_name}"
                size = meta.size
                values = [value] if size == 1 else list(value)

                if size > 1:
                    features.add("vector_fields")

                if isinstance(meta, ExcitationPort):
                    pass  # pure metadata — not compiled into p
                elif isinstance(meta, ParameterField):
                    param_index_map[key] = (param_cursor, param_cursor + size)
                    p.extend(values)
                    param_cursor += size
                else:
                    state_index_map[key] = (state_cursor, state_cursor + size)
                    x0.extend(values)
                    state_cursor += size
                    if isinstance(meta, DiscreteField):
                        features.add("discrete_fields")
                        if meta.dt > 0:
                            discrete_dts.add(meta.dt)
                        differential_mask.extend([1.0] * size)
                    elif isinstance(meta, ContinuousField):
                        if getattr(meta, "algebraic", False):
                            features.add("dae_constraints")
                            differential_mask.extend([0.0] * size)
                        else:
                            features.add("continuous_fields")
                            differential_mask.extend([1.0] * size)
                    else:
                        # IntegratedField
                        differential_mask.extend([1.0] * size)

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
                    if eid not in world.components:
                        raise ValueError(
                            f"System '{sys_model.dynamics_fn}': entity '{eid}' not found in world"
                        )
                    if not _has_component_of_type(world, eid, comp_types):
                        names = ", ".join(t.__name__ for t in comp_types)
                        found = [type(c).__name__ for c in world.components[eid].values()]
                        raise TypeError(
                            f"System '{sys_model.dynamics_fn}': entity '{eid}' has components "
                            f"{found!r}, expected one of ({names})"
                        )
            entity_ids = list(sys_model.entity_ids)

        elif comp_types:
            entity_ids = [
                eid for eid in world.components
                if _has_component_of_type(world, eid, comp_types)
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

    # --- Callbacks ---
    compiled_callbacks: list[CompiledCallback] = []
    for cb_name, cb_model in (world.callbacks or {}).items():
        if cb_model is None:
            continue
        if not cb_model.dt or cb_model.dt <= 0:
            raise ValueError(
                f"Callback '{cb_name}': dt must be > 0, got {cb_model.dt!r}"
            )
        python_fn = type(cb_model).python_fn
        compiled_callbacks.append(CompiledCallback(
            name=cb_name,
            dt=cb_model.dt,
            julia_fn=cb_model.julia_fn,
            params=dict(cb_model.params),
            python_fn=python_fn,
        ))
    if compiled_callbacks:
        features.add("control_callbacks")

    jac_rows, jac_cols = _compute_jac_sparsity(state_index_map, systems)

    return CompiledSpec(
        state_size=state_cursor,
        param_size=param_cursor,
        state_index_map=state_index_map,
        param_index_map=param_index_map,
        discrete_dts=sorted(discrete_dts),
        x0=x0,
        p=p,
        differential_mask=differential_mask,
        systems=systems,
        compiled_callbacks=compiled_callbacks,
        required_features=frozenset(features),
        jac_sparsity_rows=jac_rows,
        jac_sparsity_cols=jac_cols,
    )
