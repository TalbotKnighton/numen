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
    """Continuous state variable; solver integrates dx/dt = f(...)."""
    size: int = 1


@dataclass(frozen=True)
class ContinuousField:
    """Algebraic or output variable; computed from state, not integrated."""
    size: int = 1


@dataclass(frozen=True)
class DiscreteField:
    """Zero-order-hold variable; updated at a fixed rate. Injects required solver times."""
    dt: float = 0.0
    size: int = 1


@dataclass(frozen=True)
class ParameterField:
    """Constant parameter; enters parameter vector p, not state vector x."""
    size: int = 1

