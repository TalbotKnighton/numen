"""Field annotation types for the Numen ECS framework.

Each annotation declares how a component field maps into the flat solver arrays:

- ``IntegratedField`` — state variable; solver integrates ``dx/dt = f(...)``
- ``ParameterField`` — constant parameter in ``p``; never changes during a solve
- ``ContinuousField`` — algebraic or output variable written each RHS call
- ``DiscreteField`` — zero-order-hold variable; updated at a fixed rate
- ``ExcitationPort`` — injectable forcing input for the characterization framework
- ``EntityGroup`` — slot-type declaration for multi-entity coupled systems

Usage example::

    from typing import Annotated, Literal
    from numen.spec.component import Component
    from numen.fields import IntegratedField, ParameterField

    class MassComponent(Component):
        kind:     Literal["mass"] = "mass"
        position: Annotated[float, IntegratedField()] = 0.0
        velocity: Annotated[float, IntegratedField()] = 0.0
        mass:     Annotated[float, ParameterField()]  = 1.0
"""

from dataclasses import dataclass


class EntityGroup:
    """Declares the slot types for one entity group in a multi-slot system.

    Analogous to IntegratedField(size=N): bundles type declarations with an
    implicit size (= number of slots). Used as a ClassVar on System subclasses
    to declare coupling topology and drive compile-time validation.

    Example:
        entity_slots: ClassVar[EntityGroup] = EntityGroup(MassComponent, SpringComponent, MassComponent)

    compile_spec reads entity_slots.slot_types for validation and entity_slots.size
    to populate CompiledSystem.group_size, which the backend uses to pre-group
    entity_ids before dispatching to the dynamics function.
    """

    __slots__ = ("slot_types",)

    def __init__(self, *types: type) -> None:
        self.slot_types: tuple[type, ...] = types

    @property
    def size(self) -> int:
        return len(self.slot_types)

    def __repr__(self) -> str:
        names = ", ".join(t.__name__ for t in self.slot_types)
        return f"EntityGroup({names})"


@dataclass(frozen=True)
class IntegratedField:
    """Continuous state variable; solver integrates ``dx/dt = f(...)``.

    Placed in the state vector ``x``. The ODE solver updates it every step.

    Args:
        size: Number of contiguous scalar slots. Default 1.
              Use ``size=N`` for vector fields (e.g. frequency bins).

    Example::

        position: Annotated[float, IntegratedField()] = 0.0
        frequencies: Annotated[list[float], IntegratedField(size=8)] = [0.0]*8
    """
    size: int = 1


@dataclass(frozen=True)
class ContinuousField:
    """Algebraic or output variable; computed from state each RHS call.

    When ``algebraic=False`` (default): output variable — the dynamics function
    writes a derived quantity each RHS call (e.g. force, power, flow rate).
    The ``differential_mask`` is 1 for these slots; all backends support it.

    When ``algebraic=True``: algebraic constraint — the dynamics function writes
    a residual ``g(x)=0``. The ``differential_mask`` is 0 for these slots.
    Julia-only (requires an implicit solver such as Rodas5P or FBDF).

    Args:
        size:      Number of contiguous scalar slots.
        algebraic: If True, marks this field as a DAE algebraic constraint
                   (residual form). Julia-only; raises ``NumenFeatureError``
                   on scipy and JAX backends.
    """
    size:      int  = 1
    algebraic: bool = False


@dataclass(frozen=True)
class DiscreteField:
    """Zero-order-hold variable updated at a fixed rate.

    Occupies a slot in the state vector ``x``. Solver is given required stop
    times at multiples of ``dt`` so the controller callback fires at exact times.

    Args:
        dt:   Update period in seconds. Must be > 0.
        size: Number of contiguous scalar slots.
    """
    dt: float = 0.0
    size: int = 1


@dataclass(frozen=True)
class ParameterField:
    """Constant parameter that enters the parameter vector ``p``.

    Never updated during a solve. Appears in ``spec.param_index_map`` and
    is accessible via ``spec.view(entity_id, ComponentType, x, p).field_name``.

    Args:
        size: Number of contiguous scalar slots. Use ``size=N`` for vector params.
    """
    size: int = 1


@dataclass(frozen=True)
class ExcitationPort:
    """Marks a component field as an injectable excitation input port.

    Compiles like a ParameterField (goes into parameter vector p).  The
    characterization framework reads the annotation metadata to discover
    available ports and uses inject_excitation() to add a time-varying
    forcing system post-compilation.

    Args:
        targets:   Name of the IntegratedField whose derivative receives F(t).
                   E.g. "velocity" means F(t) is added to d(velocity)/dt.
        port_type: Bond graph port type — "effort" (force, pressure, voltage)
                   or "flow" (velocity, flow rate, current).  Metadata only;
                   used for axis labels and FRF naming conventions.
        units:     SI units string for axis labels, e.g. "N", "Pa", "V".
        size:      Number of scalar values (consistent with other field types).

    Example::

        class MassComponent(Component):
            velocity: Annotated[float, IntegratedField()] = 0.0
            force:    Annotated[float, ExcitationPort(
                          targets   = "velocity",
                          port_type = "effort",
                          units     = "N",
                      )] = 0.0
    """
    targets:   str = ""
    port_type: str = "effort"   # "effort" | "flow"
    units:     str = ""
    size:      int = 1

