"""Tests for numen/bridge/scipy_backend.py — solve a harmonic oscillator analytically."""
from typing import Annotated, ClassVar, Literal
import numpy as np
import pytest

from numen.fields import IntegratedField, ParameterField, DiscreteField
from numen.spec.component import Component
from numen.spec.system import System, DynamicsFn
from numen.spec.world import GenericWorld
from numen.compiler.flatten import compile_spec, DxBuffer
from numen.bridge.scipy_backend import ScipyBackend
from numen.bridge.runtime import SolveResult
from numen.errors import NumenFeatureError, NumenMissingFnError


# ---------------------------------------------------------------------------
# Simple harmonic oscillator: x'' = -ω²x
# Exact solution: x(t) = A cos(ω t), v(t) = -A ω sin(ω t)
# ---------------------------------------------------------------------------

OMEGA = 2.0 * np.pi   # rad/s → 1 Hz natural frequency
X0    = 1.0           # amplitude


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


def make_osc_world(x0=X0, v0=0.0, omega=OMEGA):
    World = GenericWorld[OscComp, OscSystem, None]
    return World(
        components={"osc": {"osc": OscComp(position=x0, velocity=v0, omega_sq=omega**2)}},
        systems={"sys": OscSystem()},
    )


# ---------------------------------------------------------------------------
# Basic solve shape / correctness
# ---------------------------------------------------------------------------

class TestScipySolveSHO:
    def test_solve_result_type(self):
        spec = compile_spec(make_osc_world())
        result = ScipyBackend(rtol=1e-10, atol=1e-12).solve(spec, tspan=(0.0, 1.0))
        assert isinstance(result, SolveResult)

    def test_result_t_is_1d(self):
        spec = compile_spec(make_osc_world())
        result = ScipyBackend(rtol=1e-10, atol=1e-12).solve(spec, tspan=(0.0, 1.0))
        assert result.t.ndim == 1
        assert len(result.t) > 0

    def test_result_x_shape(self):
        spec = compile_spec(make_osc_world())
        result = ScipyBackend(rtol=1e-10, atol=1e-12).solve(spec, tspan=(0.0, 1.0))
        assert result.x.shape == (spec.state_size, len(result.t))

    def test_initial_condition_at_t0(self):
        spec = compile_spec(make_osc_world())
        result = ScipyBackend(rtol=1e-10, atol=1e-12).solve(spec, tspan=(0.0, 1.0))
        pos_idx = spec.state_idx("osc.osc.position")
        # First time point should be at or very near t=0
        assert result.t[0] == pytest.approx(0.0, abs=1e-10)
        assert result.x[pos_idx, 0] == pytest.approx(X0, rel=1e-5)

    def test_cosine_solution_at_half_period(self):
        """At t = T/2 = 0.5 s the cosine solution gives x = -X0."""
        spec = compile_spec(make_osc_world())
        T_half = 0.5  # 0.5 s for 1 Hz oscillator
        result = ScipyBackend(rtol=1e-10, atol=1e-12).solve(
            spec, tspan=(0.0, T_half),
            t_eval=np.array([T_half]),
        )
        pos_idx = spec.state_idx("osc.osc.position")
        # x(T/2) = cos(π) = -1
        assert result.x[pos_idx, -1] == pytest.approx(-X0, rel=1e-5)

    def test_cosine_solution_at_full_period(self):
        """At t = T = 1.0 s the cosine solution returns to x = +X0."""
        spec = compile_spec(make_osc_world())
        T = 1.0
        result = ScipyBackend(rtol=1e-10, atol=1e-12).solve(
            spec, tspan=(0.0, T),
            t_eval=np.array([T]),
        )
        pos_idx = spec.state_idx("osc.osc.position")
        assert result.x[pos_idx, -1] == pytest.approx(X0, rel=1e-4)

    def test_energy_conservation(self):
        """Total energy E = ½ v² + ½ ω² x² should be constant ≈ ½ ω² X0²."""
        spec = compile_spec(make_osc_world())
        result = ScipyBackend(rtol=1e-10, atol=1e-12).solve(spec, tspan=(0.0, 5.0))
        pos_idx = spec.state_idx("osc.osc.position")
        vel_idx = spec.state_idx("osc.osc.velocity")
        x_t = result.x[pos_idx]
        v_t = result.x[vel_idx]
        E0 = 0.5 * OMEGA**2 * X0**2
        E_t = 0.5 * v_t**2 + 0.5 * OMEGA**2 * x_t**2
        np.testing.assert_allclose(E_t, E0, rtol=1e-5)

    def test_solution_matches_analytic_curve(self):
        """Verify x(t) ≈ cos(ω t) at 20 equally-spaced time points."""
        spec = compile_spec(make_osc_world())
        t_check = np.linspace(0.0, 2.0, 20)
        result = ScipyBackend(rtol=1e-10, atol=1e-12).solve(
            spec, tspan=(0.0, 2.0), t_eval=t_check,
        )
        pos_idx = spec.state_idx("osc.osc.position")
        x_numeric = result.x[pos_idx]
        x_exact   = X0 * np.cos(OMEGA * t_check)
        np.testing.assert_allclose(x_numeric, x_exact, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# n_save_points
# ---------------------------------------------------------------------------

class TestNSavePoints:
    def test_n_save_points_output_length(self):
        spec = compile_spec(make_osc_world())
        N = 50
        result = ScipyBackend(rtol=1e-10, atol=1e-12, n_save_points=N).solve(
            spec, tspan=(0.0, 1.0),
        )
        assert len(result.t) == N

    def test_n_save_points_span(self):
        spec = compile_spec(make_osc_world())
        N = 100
        result = ScipyBackend(rtol=1e-10, atol=1e-12, n_save_points=N).solve(
            spec, tspan=(0.0, 2.0),
        )
        assert result.t[0] == pytest.approx(0.0)
        assert result.t[-1] == pytest.approx(2.0)

    def test_n_save_points_and_dtsave_mutually_exclusive(self):
        with pytest.raises(ValueError):
            ScipyBackend(n_save_points=50, dtsave=0.01)


# ---------------------------------------------------------------------------
# dtsave
# ---------------------------------------------------------------------------

class TestDtSave:
    def test_dtsave_output_length(self):
        spec = compile_spec(make_osc_world())
        dtsave = 0.1
        tspan = (0.0, 1.0)
        result = ScipyBackend(rtol=1e-10, atol=1e-12, dtsave=dtsave).solve(
            spec, tspan=tspan,
        )
        # arange(0, 1.0 + dtsave*0.5, dtsave) → 11 points [0.0, 0.1, ..., 1.0]
        assert len(result.t) >= 10

    def test_dtsave_time_spacing(self):
        spec = compile_spec(make_osc_world())
        dtsave = 0.05
        result = ScipyBackend(rtol=1e-10, atol=1e-12, dtsave=dtsave).solve(
            spec, tspan=(0.0, 0.5),
        )
        diffs = np.diff(result.t)
        assert np.all(diffs >= dtsave * 0.9)


# ---------------------------------------------------------------------------
# Backend feature checking
# ---------------------------------------------------------------------------

class TestBackendFeatureCheck:
    def test_dae_constraint_raises_feature_error(self):
        from numen.fields import ContinuousField

        class DaeComp(Component):
            kind:      Literal["dae"] = "dae"
            p_state:   Annotated[float, IntegratedField()] = 1e5
            p_balance: Annotated[float, ContinuousField(algebraic=True)] = 0.0

        World = GenericWorld[DaeComp, None, None]
        w = World(components={"cv": {"dae": DaeComp()}})
        spec = compile_spec(w)
        with pytest.raises(NumenFeatureError, match="dae_constraints"):
            ScipyBackend().solve(spec, tspan=(0.0, 1.0))

    def test_missing_python_fn_raises(self):
        """A system with dynamics_fn but no python_fn raises NumenMissingFnError."""

        class NoFnSystem(System):
            component_types: ClassVar[tuple[type, ...]] = (OscComp,)
            python_fn:       ClassVar = None
            kind:            Literal["nofn"] = "nofn"
            dynamics_fn:     str             = "Dynamics.nofn!"

        World = GenericWorld[OscComp, NoFnSystem, None]
        w = World(
            components={"osc": {"osc": OscComp()}},
            systems={"sys": NoFnSystem()},
        )
        spec = compile_spec(w)
        with pytest.raises(NumenMissingFnError):
            ScipyBackend().solve(spec, tspan=(0.0, 1.0))


# ---------------------------------------------------------------------------
# Multiple initial conditions
# ---------------------------------------------------------------------------

class TestMultipleICs:
    def test_different_x0_gives_different_trajectory(self):
        spec_a = compile_spec(make_osc_world(x0=1.0))
        spec_b = compile_spec(make_osc_world(x0=2.0))
        backend = ScipyBackend(rtol=1e-10, atol=1e-12, n_save_points=100)
        res_a = backend.solve(spec_a, tspan=(0.0, 1.0))
        res_b = backend.solve(spec_b, tspan=(0.0, 1.0))
        pos_idx_a = spec_a.state_idx("osc.osc.position")
        pos_idx_b = spec_b.state_idx("osc.osc.position")
        # Amplitudes should be in 2:1 ratio
        np.testing.assert_allclose(
            res_b.x[pos_idx_b], 2.0 * res_a.x[pos_idx_a], rtol=1e-4
        )
