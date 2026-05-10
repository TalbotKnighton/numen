"""Advanced compiler tests: vector fields, multi-system, entity groups, dx accumulation."""
from typing import Annotated, ClassVar, Literal
import numpy as np
import pytest

from numen.fields import IntegratedField, ParameterField, DiscreteField, ContinuousField, EntityGroup
from numen.spec.component import Component
from numen.spec.system import System, DynamicsFn
from numen.spec.world import GenericWorld
from numen.compiler.flatten import compile_spec, DxBuffer, CompiledSpec


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

class MassComp(Component):
    kind: Literal["mass"] = "mass"
    position: Annotated[float, IntegratedField()] = 0.0
    velocity: Annotated[float, IntegratedField()] = 0.0
    mass:     Annotated[float, ParameterField()]  = 1.0


class SpringComp(Component):
    kind: Literal["spring"] = "spring"
    stiffness: Annotated[float, ParameterField()] = 10.0


def gravity_dynamics(dx, x, p, t, spec, system):
    for (eid,) in system.entity_groups:
        m = spec.view(eid, MassComp, x, p)
        dm = spec.dx_view(eid, MassComp, dx)
        dm.position += m.velocity
        dm.velocity += -9.81


class GravitySystem(System):
    component_types: ClassVar[tuple[type, ...]] = (MassComp,)
    python_fn:       ClassVar[DynamicsFn]       = staticmethod(gravity_dynamics)
    kind:            Literal["gravity"]         = "gravity"
    dynamics_fn:     str                        = "Dynamics.gravity!"


# ---------------------------------------------------------------------------
# DxBuffer
# ---------------------------------------------------------------------------

class TestDxBuffer:
    def test_numpy_setitem(self):
        arr = np.zeros(5)
        buf = DxBuffer(arr)
        buf[2] = 99.0
        assert buf.array[2] == pytest.approx(99.0)

    def test_numpy_getitem(self):
        arr = np.array([1.0, 2.0, 3.0])
        buf = DxBuffer(arr)
        assert buf[1] == pytest.approx(2.0)

    def test_array_property(self):
        arr = np.zeros(4)
        buf = DxBuffer(arr)
        assert buf.array is arr

    def test_slice_assignment(self):
        arr = np.zeros(6)
        buf = DxBuffer(arr)
        buf[2:5] = [1.0, 2.0, 3.0]
        assert list(buf.array[2:5]) == pytest.approx([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# CompiledSpec: view / dx_view
# ---------------------------------------------------------------------------

def _make_mass_spec():
    World = GenericWorld[MassComp, GravitySystem, None]
    w = World(
        components={"m": {"mass": MassComp(position=5.0, velocity=2.0, mass=3.0)}},
        systems={"grav": GravitySystem()},
    )
    return compile_spec(w)


class TestCompiledSpecView:
    def test_view_reads_position(self):
        spec = _make_mass_spec()
        x = np.array(spec.x0)
        p = np.array(spec.p)
        view = spec.view("m", MassComp, x, p)
        assert view.position == pytest.approx(5.0)

    def test_view_reads_velocity(self):
        spec = _make_mass_spec()
        x = np.array(spec.x0)
        p = np.array(spec.p)
        view = spec.view("m", MassComp, x, p)
        assert view.velocity == pytest.approx(2.0)

    def test_view_reads_parameter(self):
        spec = _make_mass_spec()
        x = np.array(spec.x0)
        p = np.array(spec.p)
        view = spec.view("m", MassComp, x, p)
        assert view.mass == pytest.approx(3.0)

    def test_view_is_readonly(self):
        spec = _make_mass_spec()
        x = np.array(spec.x0)
        p = np.array(spec.p)
        view = spec.view("m", MassComp, x, p)
        with pytest.raises(AttributeError):
            view.position = 99.0  # type: ignore[misc]

    def test_view_unknown_field_raises_attribute_error(self):
        spec = _make_mass_spec()
        x = np.array(spec.x0)
        p = np.array(spec.p)
        view = spec.view("m", MassComp, x, p)
        with pytest.raises(AttributeError):
            _ = view.nonexistent_field

    def test_dx_view_write(self):
        spec = _make_mass_spec()
        dx = DxBuffer(np.zeros(spec.state_size))
        dv = spec.dx_view("m", MassComp, dx)
        dv.velocity += 5.0
        vel_idx = spec.state_idx("m.mass.velocity")
        assert dx.array[vel_idx] == pytest.approx(5.0)

    def test_dx_view_accumulates(self):
        spec = _make_mass_spec()
        dx = DxBuffer(np.zeros(spec.state_size))
        dv = spec.dx_view("m", MassComp, dx)
        dv.position += 1.0
        dv.position += 2.0
        pos_idx = spec.state_idx("m.mass.position")
        assert dx.array[pos_idx] == pytest.approx(3.0)

    def test_dx_view_write_to_param_raises(self):
        spec = _make_mass_spec()
        dx = DxBuffer(np.zeros(spec.state_size))
        dv = spec.dx_view("m", MassComp, dx)
        with pytest.raises(AttributeError):
            dv.mass = 5.0  # type: ignore[misc]

    def test_dx_view_read_current_value(self):
        spec = _make_mass_spec()
        dx = DxBuffer(np.zeros(spec.state_size))
        dv = spec.dx_view("m", MassComp, dx)
        dv.position += 7.0
        assert dv.position == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# state_slice / param_slice
# ---------------------------------------------------------------------------

class TestSliceHelpers:
    def test_state_slice_scalar(self):
        spec = _make_mass_spec()
        s = spec.state_slice("m.mass.position")
        assert isinstance(s, slice)
        assert s.stop - s.start == 1

    def test_param_slice_scalar(self):
        spec = _make_mass_spec()
        s = spec.param_slice("m.mass.mass")
        assert isinstance(s, slice)

    def test_state_slice_vector(self):
        class VibComp(Component):
            kind: Literal["vib"] = "vib"
            modes: Annotated[list[float], IntegratedField(size=4)] = [1.0, 2.0, 3.0, 4.0]

        World = GenericWorld[VibComp, None, None]
        w = World(components={"v": {"vib": VibComp()}})
        spec = compile_spec(w)
        sl = spec.state_slice("v.vib.modes")
        assert sl.stop - sl.start == 4


# ---------------------------------------------------------------------------
# Multi-system compilation
# ---------------------------------------------------------------------------

def spring_dynamics(dx, x, p, t, spec, system):
    for (eid_m1, eid_s, eid_m2) in system.entity_groups:
        m1 = spec.view(eid_m1, MassComp,   x, p)
        sp = spec.view(eid_s,  SpringComp, x, p)
        m2 = spec.view(eid_m2, MassComp,   x, p)
        dm1 = spec.dx_view(eid_m1, MassComp, dx)
        dm2 = spec.dx_view(eid_m2, MassComp, dx)
        F = sp.stiffness * (m2.position - m1.position)
        dm1.velocity += F / m1.mass
        dm2.velocity -= F / m2.mass


class SpringSystem(System):
    component_types: ClassVar[tuple[type, ...]] = ()
    entity_slots:    ClassVar[EntityGroup]      = EntityGroup(MassComp, SpringComp, MassComp)
    python_fn:       ClassVar[DynamicsFn]       = staticmethod(spring_dynamics)
    kind:            Literal["spring"]          = "spring"
    dynamics_fn:     str                        = "Dynamics.spring!"


class TestEntityGroups:
    def _make_coupled_world(self):
        from typing import Union
        from pydantic import Field as PydField

        AnyComp = Annotated[Union[MassComp, SpringComp], PydField(discriminator="kind")]
        AnySystem = Annotated[Union[GravitySystem, SpringSystem], PydField(discriminator="kind")]
        World = GenericWorld[AnyComp, AnySystem, None]
        return World(
            components={
                "m1": {"mass": MassComp(position=0.0, velocity=0.0, mass=1.0)},
                "s1": {"spring": SpringComp(stiffness=100.0)},
                "m2": {"mass": MassComp(position=1.0, velocity=0.0, mass=1.0)},
            },
            systems={
                "grav": GravitySystem(),
                "spring": SpringSystem(entity_groups=[["m1", "s1", "m2"]]),
            },
        )

    def test_entity_group_compiled(self):
        w = self._make_coupled_world()
        spec = compile_spec(w)
        spring_sys = next(s for s in spec.systems if "spring" in s.dynamics_fn)
        assert spring_sys.group_size == 3
        assert spring_sys.entity_ids == ["m1", "s1", "m2"]

    def test_entity_groups_tuple_structure(self):
        w = self._make_coupled_world()
        spec = compile_spec(w)
        spring_sys = next(s for s in spec.systems if "spring" in s.dynamics_fn)
        # entity_groups is a tuple of tuples
        assert len(spring_sys.entity_groups) == 1
        assert spring_sys.entity_groups[0] == ("m1", "s1", "m2")

    def test_wrong_group_size_raises(self):
        from typing import Union
        from pydantic import Field as PydField

        AnyComp = Annotated[Union[MassComp, SpringComp], PydField(discriminator="kind")]
        AnySystem = Annotated[Union[SpringSystem], PydField(discriminator="kind")]
        World = GenericWorld[AnyComp, AnySystem, None]
        w = World(
            components={
                "m1": {"mass": MassComp()},
                "s1": {"spring": SpringComp()},
                "m2": {"mass": MassComp()},
            },
            systems={
                "spring": SpringSystem(entity_groups=[["m1", "s1"]]),  # only 2 instead of 3
            },
        )
        with pytest.raises(ValueError, match="group"):
            compile_spec(w)

    def test_entity_not_found_raises(self):
        from typing import Union
        from pydantic import Field as PydField

        AnyComp = Annotated[Union[MassComp, SpringComp], PydField(discriminator="kind")]
        AnySystem = Annotated[Union[SpringSystem], PydField(discriminator="kind")]
        World = GenericWorld[AnyComp, AnySystem, None]
        w = World(
            components={
                "m1": {"mass": MassComp()},
                "s1": {"spring": SpringComp()},
            },
            systems={
                "spring": SpringSystem(entity_groups=[["m1", "s1", "ghost"]]),  # ghost not in world
            },
        )
        with pytest.raises(ValueError, match="ghost"):
            compile_spec(w)


# ---------------------------------------------------------------------------
# Dynamics via ScipyBackend — verifying dx accumulation
# ---------------------------------------------------------------------------

class TestDynamicsIntegration:
    def test_gravity_dynamics_sets_derivatives(self):
        """gravity_dynamics correctly writes d(position)/dt=velocity and d(velocity)/dt=-9.81."""
        World = GenericWorld[MassComp, GravitySystem, None]
        w = World(
            components={"m": {"mass": MassComp(position=0.0, velocity=5.0, mass=1.0)}},
            systems={"grav": GravitySystem()},
        )
        spec = compile_spec(w)
        x = np.array(spec.x0)
        p = np.array(spec.p)
        dx = DxBuffer(np.zeros(spec.state_size))
        sys = spec.systems[0]
        sys.python_fn(dx, x, p, 0.0, spec, sys)

        pos_idx = spec.state_idx("m.mass.position")
        vel_idx = spec.state_idx("m.mass.velocity")
        assert dx.array[pos_idx] == pytest.approx(5.0)   # d(pos)/dt = velocity = 5.0
        assert dx.array[vel_idx] == pytest.approx(-9.81) # d(vel)/dt = -g

    def test_two_systems_accumulate_in_dx(self):
        """Two systems writing to the same entity's dx slots both contribute."""

        def extra_force_dynamics(dx, x, p, t, spec, system):
            for (eid,) in system.entity_groups:
                dm = spec.dx_view(eid, MassComp, dx)
                dm.velocity += 1.0  # constant extra acceleration

        class ExtraForceSystem(System):
            component_types: ClassVar[tuple[type, ...]] = (MassComp,)
            python_fn:       ClassVar[DynamicsFn]       = staticmethod(extra_force_dynamics)
            kind:            Literal["extra_force"]     = "extra_force"
            dynamics_fn:     str                        = "Dynamics.extra!"

        from typing import Union
        from pydantic import Field as PydField

        AnySystem = Annotated[Union[GravitySystem, ExtraForceSystem], PydField(discriminator="kind")]
        World = GenericWorld[MassComp, AnySystem, None]
        w = World(
            components={"m": {"mass": MassComp(velocity=3.0)}},
            systems={
                "grav": GravitySystem(),
                "extra": ExtraForceSystem(),
            },
        )
        spec = compile_spec(w)
        x = np.array(spec.x0)
        p = np.array(spec.p)
        dx = DxBuffer(np.zeros(spec.state_size))
        for sys in spec.systems:
            sys.python_fn(dx, x, p, 0.0, spec, sys)

        vel_idx = spec.state_idx("m.mass.velocity")
        # -9.81 from gravity + 1.0 from extra = -8.81
        assert dx.array[vel_idx] == pytest.approx(-9.81 + 1.0)


# ---------------------------------------------------------------------------
# Jacobian sparsity
# ---------------------------------------------------------------------------

class TestJacobianSparsity:
    def test_single_system_sparsity_is_non_empty(self):
        spec = _make_mass_spec()
        # Two state slots for one mass: position and velocity → 2×2 block
        assert len(spec.jac_sparsity_rows) > 0
        assert len(spec.jac_sparsity_cols) > 0
        assert len(spec.jac_sparsity_rows) == len(spec.jac_sparsity_cols)

    def test_no_system_yields_empty_sparsity(self):
        World = GenericWorld[MassComp, None, None]
        w = World(components={"m": {"mass": MassComp()}})
        spec = compile_spec(w)
        assert spec.jac_sparsity_rows == []
        assert spec.jac_sparsity_cols == []

    def test_sparsity_indices_are_in_bounds(self):
        spec = _make_mass_spec()
        for r in spec.jac_sparsity_rows:
            assert 0 <= r < spec.state_size
        for c in spec.jac_sparsity_cols:
            assert 0 <= c < spec.state_size
