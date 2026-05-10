"""Tests for numen/fields.py — field metadata, annotations, vector fields, ExcitationPort."""
from typing import Annotated, Literal, get_type_hints, get_args, get_origin
import typing
import pytest

from numen.fields import (
    IntegratedField,
    ParameterField,
    DiscreteField,
    ContinuousField,
    ExcitationPort,
    EntityGroup,
)
from numen.spec.component import Component


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_numen_field_meta(component_type, field_name):
    hints = get_type_hints(component_type, include_extras=True)
    hint = hints[field_name]
    assert get_origin(hint) is typing.Annotated
    args = get_args(hint)
    return args[1]  # first metadata item


# ---------------------------------------------------------------------------
# IntegratedField
# ---------------------------------------------------------------------------

class TestIntegratedField:
    def test_default_size(self):
        f = IntegratedField()
        assert f.size == 1

    def test_custom_size(self):
        f = IntegratedField(size=4)
        assert f.size == 4

    def test_frozen_dataclass(self):
        f = IntegratedField(size=3)
        with pytest.raises((AttributeError, TypeError)):
            f.size = 99  # type: ignore[misc]

    def test_equality(self):
        assert IntegratedField() == IntegratedField()
        assert IntegratedField(size=3) == IntegratedField(size=3)
        assert IntegratedField(size=1) != IntegratedField(size=2)

    def test_annotation_on_component(self):
        class C(Component):
            kind: Literal["c"] = "c"
            x: Annotated[float, IntegratedField()] = 0.0

        meta = _get_numen_field_meta(C, "x")
        assert isinstance(meta, IntegratedField)
        assert meta.size == 1


# ---------------------------------------------------------------------------
# ParameterField
# ---------------------------------------------------------------------------

class TestParameterField:
    def test_default_size(self):
        assert ParameterField().size == 1

    def test_custom_size(self):
        assert ParameterField(size=8).size == 8

    def test_frozen(self):
        f = ParameterField()
        with pytest.raises((AttributeError, TypeError)):
            f.size = 5  # type: ignore[misc]

    def test_annotation_on_component(self):
        class C(Component):
            kind: Literal["c"] = "c"
            mass: Annotated[float, ParameterField()] = 1.0

        meta = _get_numen_field_meta(C, "mass")
        assert isinstance(meta, ParameterField)


# ---------------------------------------------------------------------------
# DiscreteField
# ---------------------------------------------------------------------------

class TestDiscreteField:
    def test_defaults(self):
        f = DiscreteField()
        assert f.dt == 0.0
        assert f.size == 1

    def test_custom_dt_and_size(self):
        f = DiscreteField(dt=0.01, size=2)
        assert f.dt == pytest.approx(0.01)
        assert f.size == 2

    def test_frozen(self):
        f = DiscreteField(dt=0.01)
        with pytest.raises((AttributeError, TypeError)):
            f.dt = 0.1  # type: ignore[misc]

    def test_annotation_on_component(self):
        class C(Component):
            kind: Literal["c"] = "c"
            cmd: Annotated[float, DiscreteField(dt=0.05)] = 0.0

        meta = _get_numen_field_meta(C, "cmd")
        assert isinstance(meta, DiscreteField)
        assert meta.dt == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# ContinuousField
# ---------------------------------------------------------------------------

class TestContinuousField:
    def test_defaults(self):
        f = ContinuousField()
        assert f.size == 1
        assert f.algebraic is False

    def test_algebraic_flag(self):
        f = ContinuousField(algebraic=True)
        assert f.algebraic is True

    def test_size(self):
        f = ContinuousField(size=3)
        assert f.size == 3

    def test_frozen(self):
        f = ContinuousField()
        with pytest.raises((AttributeError, TypeError)):
            f.algebraic = True  # type: ignore[misc]

    def test_annotation_on_component(self):
        class C(Component):
            kind: Literal["c"] = "c"
            force: Annotated[float, ContinuousField()] = 0.0

        meta = _get_numen_field_meta(C, "force")
        assert isinstance(meta, ContinuousField)
        assert meta.algebraic is False

    def test_algebraic_annotation(self):
        class C(Component):
            kind: Literal["c"] = "c"
            residual: Annotated[float, ContinuousField(algebraic=True)] = 0.0

        meta = _get_numen_field_meta(C, "residual")
        assert meta.algebraic is True


# ---------------------------------------------------------------------------
# ExcitationPort
# ---------------------------------------------------------------------------

class TestExcitationPort:
    def test_defaults(self):
        ep = ExcitationPort()
        assert ep.targets == ""
        assert ep.port_type == "effort"
        assert ep.units == ""
        assert ep.size == 1

    def test_custom_attrs(self):
        ep = ExcitationPort(targets="velocity", port_type="effort", units="N", size=1)
        assert ep.targets == "velocity"
        assert ep.port_type == "effort"
        assert ep.units == "N"

    def test_frozen(self):
        ep = ExcitationPort()
        with pytest.raises((AttributeError, TypeError)):
            ep.targets = "x"  # type: ignore[misc]

    def test_excitation_port_not_in_state_or_param(self):
        """ExcitationPort fields must NOT appear in compiled state or param vectors."""
        from numen.spec.world import GenericWorld
        from numen.compiler.flatten import compile_spec

        class MassComp(Component):
            kind: Literal["mass"] = "mass"
            velocity: Annotated[float, IntegratedField()] = 0.0
            force: Annotated[float, ExcitationPort(targets="velocity", port_type="effort", units="N")] = 0.0

        World = GenericWorld[MassComp, None, None]
        world = World(components={"m": {"mass": MassComp()}})
        spec = compile_spec(world)

        # velocity is integrated — it must appear in the state map
        assert "m.mass.velocity" in spec.state_index_map
        # force has ExcitationPort — must NOT appear anywhere
        assert "m.mass.force" not in spec.state_index_map
        assert "m.mass.force" not in spec.param_index_map
        # Only 1 state slot (velocity)
        assert spec.state_size == 1
        # No param slots
        assert spec.param_size == 0


# ---------------------------------------------------------------------------
# EntityGroup
# ---------------------------------------------------------------------------

class TestEntityGroup:
    def test_size_matches_types(self):
        class A(Component):
            kind: Literal["a"] = "a"

        class B(Component):
            kind: Literal["b"] = "b"

        eg = EntityGroup(A, B, A)
        assert eg.size == 3
        assert eg.slot_types == (A, B, A)

    def test_repr(self):
        class Foo(Component):
            kind: Literal["foo"] = "foo"

        eg = EntityGroup(Foo, Foo)
        r = repr(eg)
        assert "Foo" in r
        assert "EntityGroup" in r

    def test_empty_group(self):
        eg = EntityGroup()
        assert eg.size == 0
        assert eg.slot_types == ()


# ---------------------------------------------------------------------------
# Vector fields (size=N)
# ---------------------------------------------------------------------------

class TestVectorFields:
    def test_integrated_vector_size_annotation(self):
        class VibComp(Component):
            kind: Literal["vib"] = "vib"
            modes: Annotated[list[float], IntegratedField(size=4)] = [0.0, 0.0, 0.0, 0.0]

        meta = _get_numen_field_meta(VibComp, "modes")
        assert meta.size == 4

    def test_parameter_vector_size_annotation(self):
        class TuneComp(Component):
            kind: Literal["tune"] = "tune"
            gains: Annotated[list[float], ParameterField(size=3)] = [1.0, 2.0, 3.0]

        meta = _get_numen_field_meta(TuneComp, "gains")
        assert meta.size == 3

    def test_vector_field_compiles_to_correct_state_size(self):
        from numen.spec.world import GenericWorld
        from numen.compiler.flatten import compile_spec

        class VibComp(Component):
            kind: Literal["vib"] = "vib"
            pos: Annotated[list[float], IntegratedField(size=3)] = [1.0, 2.0, 3.0]
            vel: Annotated[list[float], IntegratedField(size=3)] = [0.0, 0.0, 0.0]

        World = GenericWorld[VibComp, None, None]
        world = World(components={"e": {"vib": VibComp()}})
        spec = compile_spec(world)
        assert spec.state_size == 6

    def test_vector_field_state_indices_are_contiguous(self):
        from numen.spec.world import GenericWorld
        from numen.compiler.flatten import compile_spec

        class VibComp(Component):
            kind: Literal["vib"] = "vib"
            pos: Annotated[list[float], IntegratedField(size=3)] = [1.0, 2.0, 3.0]

        World = GenericWorld[VibComp, None, None]
        world = World(components={"e": {"vib": VibComp()}})
        spec = compile_spec(world)
        start, end = spec.state_index_map["e.vib.pos"]
        assert end - start == 3
        assert list(spec.x0[start:end]) == pytest.approx([1.0, 2.0, 3.0])

    def test_vector_field_features_flag(self):
        from numen.spec.world import GenericWorld
        from numen.compiler.flatten import compile_spec

        class VibComp(Component):
            kind: Literal["vib"] = "vib"
            modes: Annotated[list[float], IntegratedField(size=4)] = [0.0] * 4

        World = GenericWorld[VibComp, None, None]
        world = World(components={"e": {"vib": VibComp()}})
        spec = compile_spec(world)
        assert "vector_fields" in spec.required_features

    def test_scalar_field_does_not_set_vector_flag(self):
        from numen.spec.world import GenericWorld
        from numen.compiler.flatten import compile_spec

        class SimpleComp(Component):
            kind: Literal["simple"] = "simple"
            x: Annotated[float, IntegratedField()] = 0.0

        World = GenericWorld[SimpleComp, None, None]
        world = World(components={"e": {"simple": SimpleComp()}})
        spec = compile_spec(world)
        assert "vector_fields" not in spec.required_features
