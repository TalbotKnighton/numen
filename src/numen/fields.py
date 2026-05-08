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
    """Algebraic or output variable; computed from state each RHS call.

    algebraic=False (default): output variable — dynamics fn writes a derived
        quantity (force, power, flow rate).  differential_mask = 1.  All backends.

    algebraic=True: algebraic constraint — dynamics fn writes a residual g(x)=0.
        differential_mask = 0 for these slots.  Julia-only (Rodas5P / FBDF / IDA).
        An implicit solver is required; numen raises an error if an explicit
        Julia solver is selected with algebraic constraints present.

    See docs/architecture.md — "The differential_mask convention".
    """
    size:      int  = 1
    algebraic: bool = False


@dataclass(frozen=True)
class DiscreteField:
    """Zero-order-hold variable; updated at a fixed rate. Injects required solver times."""
    dt: float = 0.0
    size: int = 1


@dataclass(frozen=True)
class ParameterField:
    """Constant parameter; enters parameter vector p, not state vector x."""
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

