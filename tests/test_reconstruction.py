"""Tests for SnapshotCollector.at(), .field_series(), and reconstruct_snapshot()."""
from typing import Annotated, ClassVar, Literal
import numpy as np
import pytest

from numen.fields import IntegratedField, ParameterField
from numen.spec.component import Component
from numen.spec.system import System, DynamicsFn
from numen.spec.world import GenericWorld
from numen.compiler.flatten import compile_spec
from numen.bridge.scipy_backend import ScipyBackend
from numen.bridge.runtime import SolveResult
from numen.reconstruction.collector import SnapshotCollector
from numen.reconstruction.snapshot import reconstruct_snapshot


# ---------------------------------------------------------------------------
# Shared fixtures: simple harmonic oscillator
# ---------------------------------------------------------------------------

OMEGA = 2.0 * np.pi
X0    = 1.5


class OscComp(Component):
    kind:     Literal["osc"] = "osc"
    position: Annotated[float, IntegratedField()] = X0
    velocity: Annotated[float, IntegratedField()] = 0.0
    omega_sq: Annotated[float, ParameterField()]  = OMEGA ** 2


def osc_dynamics(dx, x, p, t, spec, system):
    for (eid,) in system.entity_groups:
        s  = spec.view(eid, OscComp, x, p)
        ds = spec.dx_view(eid, OscComp, dx)
        ds.position += s.velocity
        ds.velocity += -s.omega_sq * s.position


class OscSystem(System):
    component_types: ClassVar[tuple[type, ...]] = (OscComp,)
    python_fn:       ClassVar[DynamicsFn]       = staticmethod(osc_dynamics)
    kind:            Literal["osc_sys"]         = "osc_sys"
    dynamics_fn:     str                        = "Dynamics.osc!"


def _make_world_and_solve(x0=X0, tspan=(0.0, 1.0), n_save_points=200):
    World = GenericWorld[OscComp, OscSystem, None]
    w = World(
        components={"osc": {"osc": OscComp(position=x0, velocity=0.0, omega_sq=OMEGA**2)}},
        systems={"sys": OscSystem()},
    )
    spec = compile_spec(w)
    result = ScipyBackend(rtol=1e-10, atol=1e-12, n_save_points=n_save_points).solve(
        spec, tspan=tspan,
    )
    return w, spec, result


# ---------------------------------------------------------------------------
# SnapshotCollector.field_series
# ---------------------------------------------------------------------------

class TestFieldSeries:
    def test_returns_tuple_of_arrays(self):
        w, spec, result = _make_world_and_solve()
        collector = SnapshotCollector(w, spec, result)
        t, vals = collector.field_series("osc", "osc", "position")
        assert isinstance(t, np.ndarray)
        assert isinstance(vals, np.ndarray)

    def test_t_matches_result_t(self):
        w, spec, result = _make_world_and_solve()
        collector = SnapshotCollector(w, spec, result)
        t, _ = collector.field_series("osc", "osc", "position")
        np.testing.assert_array_equal(t, result.t)

    def test_position_values_match_analytic(self):
        """field_series for position should match X0 * cos(ω t)."""
        w, spec, result = _make_world_and_solve(n_save_points=100)
        collector = SnapshotCollector(w, spec, result)
        t, pos = collector.field_series("osc", "osc", "position")
        exact = X0 * np.cos(OMEGA * t)
        np.testing.assert_allclose(pos, exact, rtol=1e-4, atol=1e-4)

    def test_velocity_values_match_analytic(self):
        """v(t) = -X0 ω sin(ω t)."""
        w, spec, result = _make_world_and_solve(n_save_points=100)
        collector = SnapshotCollector(w, spec, result)
        t, vel = collector.field_series("osc", "osc", "velocity")
        exact = -X0 * OMEGA * np.sin(OMEGA * t)
        np.testing.assert_allclose(vel, exact, rtol=1e-4, atol=1e-3)

    def test_parameter_field_series_is_constant(self):
        """A ParameterField should yield a constant time series equal to its value."""
        w, spec, result = _make_world_and_solve()
        collector = SnapshotCollector(w, spec, result)
        t, omega_sq_vals = collector.field_series("osc", "osc", "omega_sq")
        np.testing.assert_allclose(omega_sq_vals, OMEGA**2, rtol=1e-12)
        assert len(omega_sq_vals) == len(result.t)

    def test_field_series_unknown_key_raises(self):
        w, spec, result = _make_world_and_solve()
        collector = SnapshotCollector(w, spec, result)
        with pytest.raises(KeyError):
            collector.field_series("osc", "osc", "nonexistent_field")


# ---------------------------------------------------------------------------
# SnapshotCollector.at
# ---------------------------------------------------------------------------

class TestSnapshotAt:
    def test_at_returns_world_type(self):
        w, spec, result = _make_world_and_solve()
        collector = SnapshotCollector(w, spec, result)
        snap = collector.at(0.0)
        assert hasattr(snap, "components")

    def test_at_t0_position_is_x0(self):
        w, spec, result = _make_world_and_solve()
        collector = SnapshotCollector(w, spec, result)
        snap = collector.at(0.0)
        comp = snap.components["osc"]["osc"]
        assert comp.position == pytest.approx(X0, rel=1e-4)

    def test_at_half_period_position_is_minus_x0(self):
        """At T/2 = 0.5 s, cos(ωT/2) = cos(π) = -1."""
        w, spec, result = _make_world_and_solve(tspan=(0.0, 0.5), n_save_points=500)
        collector = SnapshotCollector(w, spec, result)
        snap = collector.at(0.5)
        comp = snap.components["osc"]["osc"]
        assert comp.position == pytest.approx(-X0, rel=1e-3)

    def test_at_preserves_parameter_values(self):
        """Snapshot should preserve the parameter (omega_sq) unchanged."""
        w, spec, result = _make_world_and_solve()
        collector = SnapshotCollector(w, spec, result)
        snap = collector.at(0.5)
        comp = snap.components["osc"]["osc"]
        assert comp.omega_sq == pytest.approx(OMEGA**2)

    def test_at_returns_independent_copies(self):
        """Two calls to at() should return independent snapshot objects."""
        w, spec, result = _make_world_and_solve()
        collector = SnapshotCollector(w, spec, result)
        snap1 = collector.at(0.0)
        snap2 = collector.at(0.5)
        assert snap1 is not snap2
        assert snap1.components["osc"]["osc"] is not snap2.components["osc"]["osc"]

    def test_at_clamps_to_last_time(self):
        """Requesting a time beyond result.t[-1] should return the last state."""
        w, spec, result = _make_world_and_solve(tspan=(0.0, 1.0), n_save_points=100)
        collector = SnapshotCollector(w, spec, result)
        snap_last = collector.at(result.t[-1])
        snap_beyond = collector.at(1e9)
        pos_last   = snap_last.components["osc"]["osc"].position
        pos_beyond = snap_beyond.components["osc"]["osc"].position
        assert pos_last == pytest.approx(pos_beyond)


# ---------------------------------------------------------------------------
# reconstruct_snapshot (lower-level)
# ---------------------------------------------------------------------------

class TestReconstructSnapshot:
    def test_reconstruct_at_t0(self):
        w, spec, result = _make_world_and_solve()
        snap = reconstruct_snapshot(w, spec, result, t=0.0)
        comp = snap.components["osc"]["osc"]
        assert comp.position == pytest.approx(X0, rel=1e-4)

    def test_reconstruct_does_not_mutate_original(self):
        w, spec, result = _make_world_and_solve()
        original_pos = w.components["osc"]["osc"].position
        _snap = reconstruct_snapshot(w, spec, result, t=0.5)
        # Original world must be unchanged
        assert w.components["osc"]["osc"].position == pytest.approx(original_pos)


# ---------------------------------------------------------------------------
# SnapshotCollector.uniform / at_times
# ---------------------------------------------------------------------------

class TestUniformAndAtTimes:
    def test_uniform_length(self):
        w, spec, result = _make_world_and_solve()
        collector = SnapshotCollector(w, spec, result)
        snapshots = collector.uniform(n=10)
        assert len(snapshots) == 10

    def test_uniform_returns_time_snapshot_tuples(self):
        w, spec, result = _make_world_and_solve()
        collector = SnapshotCollector(w, spec, result)
        snapshots = collector.uniform(n=5)
        for t_val, snap in snapshots:
            assert isinstance(t_val, float)
            assert hasattr(snap, "components")

    def test_at_times_order_preserved(self):
        w, spec, result = _make_world_and_solve()
        collector = SnapshotCollector(w, spec, result)
        query_times = [0.1, 0.3, 0.7]
        results = collector.at_times(query_times)
        assert [r[0] for r in results] == query_times
