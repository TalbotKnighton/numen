from typing import Annotated, Literal
import pytest
from numen.fields import IntegratedField, ParameterField, DiscreteField
from numen.spec.component import Component
from numen.spec.world import GenericWorld
from numen.compiler.flatten import compile_spec


class TankComponent(Component):
    kind: Literal["tank"] = "tank"
    pressure:    Annotated[float, IntegratedField()] = 0.0
    temperature: Annotated[float, IntegratedField()] = 300.0
    volume:      Annotated[float, ParameterField()]  = 1.0
    valve_cmd:   Annotated[float, DiscreteField(dt=0.01)] = 0.0


def make_world():
    World = GenericWorld[TankComponent, None, None]
    return World(components={"tank_a": TankComponent()})


def test_compile_state_size():
    spec = compile_spec(make_world())
    # pressure + temperature + valve_cmd = 3
    assert spec.state_size == 3


def test_compile_param_size():
    spec = compile_spec(make_world())
    assert spec.param_size == 1


def test_state_index_map_keys():
    spec = compile_spec(make_world())
    assert "tank_a.pressure"    in spec.state_index_map
    assert "tank_a.temperature" in spec.state_index_map
    assert "tank_a.valve_cmd"   in spec.state_index_map


def test_param_index_map_keys():
    spec = compile_spec(make_world())
    assert "tank_a.volume" in spec.param_index_map
    assert "tank_a.volume" not in spec.state_index_map


def test_discrete_dts():
    spec = compile_spec(make_world())
    assert 0.01 in spec.discrete_dts


def test_x0_length():
    spec = compile_spec(make_world())
    assert len(spec.x0) == spec.state_size


def test_p_length():
    spec = compile_spec(make_world())
    assert len(spec.p) == spec.param_size


def test_scalar_idx_helpers():
    spec = compile_spec(make_world())
    idx = spec.state_idx("tank_a.pressure")
    assert isinstance(idx, int)
    assert spec.x0[idx] == 0.0   # default pressure

    pidx = spec.param_idx("tank_a.volume")
    assert spec.p[pidx] == 1.0   # default volume
