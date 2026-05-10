"""Tests for numen/spec/world.py — GenericWorld construction and component access."""
from typing import Annotated, Literal, Union
import pytest
from pydantic import Field

from numen.fields import IntegratedField, ParameterField, DiscreteField, ContinuousField
from numen.spec.component import Component
from numen.spec.world import GenericWorld
from numen.compiler.flatten import compile_spec


# ---------------------------------------------------------------------------
# Minimal single-entity world
# ---------------------------------------------------------------------------

class BallComp(Component):
    kind: Literal["ball"] = "ball"
    position: Annotated[float, IntegratedField()] = 10.0
    velocity: Annotated[float, IntegratedField()] = 0.0
    mass:     Annotated[float, ParameterField()]  = 2.0


def make_single_ball_world(pos=10.0, vel=0.0, mass=2.0):
    World = GenericWorld[BallComp, None, None]
    return World(components={"b": {"ball": BallComp(position=pos, velocity=vel, mass=mass)}})


class TestWorldConstruction:
    def test_empty_world(self):
        World = GenericWorld[BallComp, None, None]
        w = World(components={}, systems={})
        assert w.components == {}

    def test_single_entity_single_component(self):
        w = make_single_ball_world()
        assert "b" in w.components
        assert "ball" in w.components["b"]
        assert isinstance(w.components["b"]["ball"], BallComp)

    def test_component_values_preserved(self):
        w = make_single_ball_world(pos=5.5, vel=-1.0, mass=3.0)
        comp = w.components["b"]["ball"]
        assert comp.position == pytest.approx(5.5)
        assert comp.velocity == pytest.approx(-1.0)
        assert comp.mass == pytest.approx(3.0)

    def test_component_is_frozen(self):
        """Component subclasses declare frozen=True — direct field mutation must raise."""
        comp = BallComp(position=1.0)
        with pytest.raises(Exception):
            comp.position = 99.0  # type: ignore[misc]

    def test_default_callbacks_empty(self):
        w = make_single_ball_world()
        assert w.callbacks == {}


# ---------------------------------------------------------------------------
# Multi-entity world
# ---------------------------------------------------------------------------

class TestMultiEntityWorld:
    def test_two_entities_same_component_type(self):
        World = GenericWorld[BallComp, None, None]
        w = World(components={
            "b1": {"ball": BallComp(position=0.0)},
            "b2": {"ball": BallComp(position=5.0)},
        })
        spec = compile_spec(w)
        # 2 balls × 2 IntegratedFields = 4 state slots; 2 param slots
        assert spec.state_size == 4
        assert spec.param_size == 2
        assert "b1.ball.position" in spec.state_index_map
        assert "b2.ball.position" in spec.state_index_map

    def test_initial_conditions_are_distinct(self):
        World = GenericWorld[BallComp, None, None]
        w = World(components={
            "b1": {"ball": BallComp(position=0.0, velocity=1.0)},
            "b2": {"ball": BallComp(position=10.0, velocity=-1.0)},
        })
        spec = compile_spec(w)
        pos1_idx = spec.state_idx("b1.ball.position")
        pos2_idx = spec.state_idx("b2.ball.position")
        assert spec.x0[pos1_idx] == pytest.approx(0.0)
        assert spec.x0[pos2_idx] == pytest.approx(10.0)
        vel1_idx = spec.state_idx("b1.ball.velocity")
        vel2_idx = spec.state_idx("b2.ball.velocity")
        assert spec.x0[vel1_idx] == pytest.approx(1.0)
        assert spec.x0[vel2_idx] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Multi-component per entity
# ---------------------------------------------------------------------------

class BodyComp(Component):
    kind: Literal["body"] = "body"
    x: Annotated[float, IntegratedField()] = 0.0
    y: Annotated[float, IntegratedField()] = 0.0


class SensorComp(Component):
    kind: Literal["sensor"] = "sensor"
    reading: Annotated[float, ContinuousField()] = 0.0
    gain:    Annotated[float, ParameterField()]  = 1.0


class TestMultiComponentPerEntity:
    def test_two_components_on_one_entity(self):
        AnyComp = Annotated[Union[BodyComp, SensorComp], Field(discriminator="kind")]
        World = GenericWorld[AnyComp, None, None]
        w = World(components={
            "robot": {
                "body":   BodyComp(x=1.0, y=2.0),
                "sensor": SensorComp(gain=0.5),
            }
        })
        spec = compile_spec(w)
        # 2 IntegratedField (body.x, body.y) + 1 ContinuousField (sensor.reading) = 3 state
        assert spec.state_size == 3
        # 1 ParameterField (sensor.gain) = 1 param
        assert spec.param_size == 1
        assert "robot.body.x" in spec.state_index_map
        assert "robot.body.y" in spec.state_index_map
        assert "robot.sensor.reading" in spec.state_index_map
        assert "robot.sensor.gain" in spec.param_index_map

    def test_multi_component_initial_values(self):
        AnyComp = Annotated[Union[BodyComp, SensorComp], Field(discriminator="kind")]
        World = GenericWorld[AnyComp, None, None]
        w = World(components={
            "robot": {
                "body":   BodyComp(x=3.0, y=-1.0),
                "sensor": SensorComp(gain=2.0),
            }
        })
        spec = compile_spec(w)
        x_idx = spec.state_idx("robot.body.x")
        y_idx = spec.state_idx("robot.body.y")
        g_idx = spec.param_idx("robot.sensor.gain")
        assert spec.x0[x_idx] == pytest.approx(3.0)
        assert spec.x0[y_idx] == pytest.approx(-1.0)
        assert spec.p[g_idx] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Feature flags from world
# ---------------------------------------------------------------------------

class TestFeatureFlags:
    def test_discrete_field_sets_feature(self):
        class CtrlComp(Component):
            kind: Literal["ctrl"] = "ctrl"
            cmd: Annotated[float, DiscreteField(dt=0.01)] = 0.0

        World = GenericWorld[CtrlComp, None, None]
        w = World(components={"ctrl": {"ctrl": CtrlComp()}})
        spec = compile_spec(w)
        assert "discrete_fields" in spec.required_features

    def test_continuous_field_sets_feature(self):
        class OutComp(Component):
            kind: Literal["out"] = "out"
            power: Annotated[float, ContinuousField()] = 0.0

        World = GenericWorld[OutComp, None, None]
        w = World(components={"o": {"out": OutComp()}})
        spec = compile_spec(w)
        assert "continuous_fields" in spec.required_features

    def test_algebraic_continuous_field_sets_dae_feature(self):
        class DaeComp(Component):
            kind: Literal["dae"] = "dae"
            p_state:   Annotated[float, IntegratedField()] = 1e5
            p_balance: Annotated[float, ContinuousField(algebraic=True)] = 0.0

        World = GenericWorld[DaeComp, None, None]
        w = World(components={"cv": {"dae": DaeComp()}})
        spec = compile_spec(w)
        assert "dae_constraints" in spec.required_features

    def test_algebraic_slot_differential_mask_zero(self):
        class DaeComp(Component):
            kind: Literal["dae"] = "dae"
            p_state:   Annotated[float, IntegratedField()] = 1e5
            p_balance: Annotated[float, ContinuousField(algebraic=True)] = 0.0

        World = GenericWorld[DaeComp, None, None]
        w = World(components={"cv": {"dae": DaeComp()}})
        spec = compile_spec(w)
        # p_state → mask=1; p_balance → mask=0
        alg_idx = spec.state_idx("cv.dae.p_balance")
        int_idx = spec.state_idx("cv.dae.p_state")
        assert spec.differential_mask[int_idx] == pytest.approx(1.0)
        assert spec.differential_mask[alg_idx] == pytest.approx(0.0)

    def test_differential_mask_all_ones_for_pure_ode(self):
        w = make_single_ball_world()
        spec = compile_spec(w)
        assert all(v == pytest.approx(1.0) for v in spec.differential_mask)
        assert len(spec.differential_mask) == spec.state_size
