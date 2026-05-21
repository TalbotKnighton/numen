"""CharacterizationRunner — opens a backend once, runs all tests in a campaign.

Typical usage::

    runner  = CharacterizationRunner.from_yaml("test_plan.yaml")
    results = runner.run()
    results.save("results.json")

The runner owns the backend lifecycle.  For Julia server backends this means
the subprocess is started once before the first test and kept alive until the
campaign finishes — all solves share the JIT-compiled kernel.
"""
from __future__ import annotations

import importlib
import logging
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Generator

from numen.characterization.schema import (
    AmplitudeSweepSpec,
    BackendSpec,
    CharacterizationConfig,
    ContinuousChirpSpec,
    DCOperatingPointSweepSpec,
    DiscreteFrequencySweepSpec,
    DOESweepSpec,
    ExcitationSpec,
    FreeDecaySpec,
    HarmonicDistortionSweepSpec,
    ModelSpec,
    ParameterGridSpec,
    ParameterSweepSpec,
    PhasePortraitSpec,
    StochasticExcitationSpec,
    TestSpec,
    TwoToneSpec,
)
from numen.characterization.excitation import (
    ExcitationPortInfo,
    find_excitation_ports,
    inject_excitation,
    set_excitation_params,
)
from numen.characterization.results import CampaignResults
from numen.compiler.flatten import compile_spec

_log = logging.getLogger("numen.characterization.runner")


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def _open_backend(spec: BackendSpec) -> Any:
    if spec.type == "scipy":
        from numen.bridge.scipy_backend import ScipyBackend
        kwargs: dict[str, Any] = {"rtol": spec.rtol, "atol": spec.atol,
                                  "n_save_points": spec.n_save_points, "dtsave": spec.dtsave}
        if spec.method:
            kwargs["method"] = spec.method
        if spec.dtmax is not None:
            kwargs["dtmax"] = spec.dtmax
        return ScipyBackend(**kwargs)

    if spec.type == "jax":
        from numen.bridge.jax_backend import JAXBackend
        kwargs = {"rtol": spec.rtol, "atol": spec.atol}
        if spec.method:
            kwargs["solver"] = spec.method
        return JAXBackend(**kwargs)

    if spec.type == "julia":
        from numen.bridge.runtime import JuliaBackend
        kwargs: dict[str, Any] = {"julia_file": spec.julia_file, "rtol": spec.rtol, "atol": spec.atol,
                                  "n_save_points": spec.n_save_points, "dtsave": spec.dtsave,
                                  "dtmax": spec.dtmax, "maxiters": spec.maxiters}
        if spec.method:
            kwargs["method"] = spec.method
        return JuliaBackend(**kwargs)

    if spec.type == "julia_server":
        from numen.bridge.server_backend import JuliaServerBackend
        kwargs = {"julia_file": spec.julia_file, "rtol": spec.rtol, "atol": spec.atol,
                  "n_save_points": spec.n_save_points, "dtsave": spec.dtsave, "dtmax": spec.dtmax,
                  "maxiters": spec.maxiters, "precompile": spec.precompile}
        if spec.method:
            kwargs["method"] = spec.method
        return JuliaServerBackend(**kwargs)

    raise ValueError(f"Unknown backend type: {spec.type!r}")


@contextmanager
def _backend_context(spec: BackendSpec) -> Generator[Any, None, None]:
    """Context manager that owns the backend lifecycle.

    When ``spec.type == 'julia_server'`` and ``spec.n_workers > 1``, opens a
    ``JuliaServerPool``; sweep runners then dispatch design points across the
    pool in parallel.  All other cases open a single backend sequentially.
    """
    if spec.type == "julia_server" and spec.n_workers > 1:
        from numen.bridge.server_backend import JuliaServerPool
        pool = JuliaServerPool(
            n_workers     = spec.n_workers,
            julia_file    = spec.julia_file,
            method        = spec.method or "Tsit5",
            rtol          = spec.rtol,
            atol          = spec.atol,
            n_save_points = spec.n_save_points,
            dtsave        = spec.dtsave,
            dtmax         = spec.dtmax,
            maxiters      = spec.maxiters,
            precompile    = spec.precompile,
        )
        with pool:
            _log.info("Julia pool started (%d workers)", spec.n_workers)
            yield pool
    else:
        backend = _open_backend(spec)
        if spec.type == "julia_server":
            with backend:
                _log.info("Julia server started")
                yield backend
        else:
            with nullcontext(backend):
                yield backend


# ---------------------------------------------------------------------------
# World / spec helpers
# ---------------------------------------------------------------------------

def _build_world(model_spec: ModelSpec) -> Any:
    module  = importlib.import_module(model_spec.module)
    factory = getattr(module, model_spec.factory)
    return factory(**model_spec.factory_kwargs)


def _make_exc_spec(base_spec: Any, world: Any, exc: ExcitationSpec) -> Any:
    """Inject excitation into a compiled spec with placeholder parameters."""
    ports = find_excitation_ports(world, exc.entity, exc.component)
    if exc.port not in ports:
        raise ValueError(
            f"Entity '{exc.entity}' (component '{exc.component}') has no ExcitationPort '{exc.port}'. "
            f"Available: {list(ports)}"
        )
    port_info: ExcitationPortInfo = ports[exc.port]
    return inject_excitation(
        base_spec,
        entity_id      = exc.entity,
        component_kind = port_info.component_kind,
        port_name      = exc.port,
        target_field   = port_info.targets,
        amp=0.0, freq=1.0, dc=0.0,
        scale_by       = exc.scale_by,
    )


# ---------------------------------------------------------------------------
# CharacterizationRunner
# ---------------------------------------------------------------------------

class CharacterizationRunner:
    """Orchestrates a full characterization campaign from a validated config."""

    def __init__(
        self,
        config: CharacterizationConfig,
        plan_dir: "Path | None" = None,
        global_seed: int | None = None,
    ) -> None:
        self.config      = config
        self._plan_dir   = plan_dir          # for resolving relative file paths
        self._global_seed = global_seed       # CLI --seed override (None = no override)
        self._world      = _build_world(config.model)
        self._base_spec  = compile_spec(self._world)
        self._exc_spec = _make_exc_spec(self._base_spec, self._world, config.excitation)
        exc = config.excitation
        out_component = exc.output_component or exc.component
        self._output_key = f"{exc.entity}.{out_component}.{exc.output_state}"
        # Validate every test's parameter references against the compiled spec.
        # Failing here surfaces typos and stale two-level keys (entity.field) at
        # campaign start instead of midway through a long sweep.
        self._validate_test_parameter_keys(config.tests)
        # The ExcitationPort's target field (e.g. "velocity") and component kind —
        # used by runners that need to inject or override initial conditions.
        ports = find_excitation_ports(self._world, exc.entity, exc.component)
        self._exc_target_field   = ports[exc.port].targets
        self._exc_component_kind = ports[exc.port].component_kind
        _log.info(
            "Runner ready: state_size=%d  param_size=%d  tests=%d",
            self._exc_spec.state_size,
            self._exc_spec.param_size,
            len(config.tests),
        )

    # --- Constructors ---

    @classmethod
    def from_yaml(cls, path: str | Path, global_seed: int | None = None) -> "CharacterizationRunner":
        p = Path(path)
        return cls(CharacterizationConfig.from_yaml(p), plan_dir=p.parent, global_seed=global_seed)

    @classmethod
    def from_json(cls, path: str | Path, global_seed: int | None = None) -> "CharacterizationRunner":
        p = Path(path)
        return cls(CharacterizationConfig.from_json(p), plan_dir=p.parent, global_seed=global_seed)

    @classmethod
    def from_config(cls, config: CharacterizationConfig, global_seed: int | None = None) -> "CharacterizationRunner":
        return cls(config, global_seed=global_seed)

    # --- Public API ---

    def run(self) -> CampaignResults:
        """Execute all tests in the campaign.  Returns collected results."""
        results = CampaignResults(config_version=self.config.version)
        with _backend_context(self.config.backend) as backend:
            for test in self.config.tests:
                if not getattr(test, "enabled", True):
                    _log.info("Skipping test '%s' (enabled=false)", test.name)
                    continue
                _log.info("Running test '%s' (type=%s)", test.name, test.type)
                result = self._run_test(test, backend)
                results.append(test.name, result)
                _log.info("Test '%s' complete", test.name)
        return results

    def compiled_spec_for(
        self, amp: float = 0.0, freq: float = 1.0, dc: float = 0.0,
    ) -> Any:
        """Return an exc_spec with excitation set to the given parameters."""
        from numen.characterization.excitation import set_excitation_params
        return set_excitation_params(
            self._exc_spec,
            self.config.excitation.entity,
            self.config.excitation.port,
            amp=amp, freq=freq, dc=dc,
        )

    # --- Validation ---

    def _validate_test_parameter_keys(self, tests: list[TestSpec]) -> None:
        """Fail fast if any test references a parameter that doesn't exist.

        Walks every ``parameter_sweep``, ``parameter_grid``, and ``doe_sweep``
        and verifies each ``sweep_param`` / ``params`` key resolves against the
        compiled ``param_index_map``. ``excitation.*`` paths are translated the
        same way the runners do at execution time.

        Raises:
            KeyError: with the offending test name, the bad key, and the full
                list of valid model parameter keys.
        """
        from numen.characterization.tests.param_sweep import _resolve_param_key

        exc      = self.config.excitation
        valid    = set(self._exc_spec.param_index_map)
        # Hide internal excitation slots from the error message — they're an
        # implementation detail and would just clutter the suggestion.
        model_params = sorted(k for k in valid if not k.startswith("_exc_"))

        def _check(test_name: str, key: str) -> None:
            resolved = _resolve_param_key(key, exc.entity, exc.port)
            if resolved not in valid:
                raise KeyError(
                    f"Test '{test_name}': parameter '{key}' not found in the "
                    f"compiled spec. Model parameter keys must use the full "
                    f"three-level path 'entity.component_kind.field' (e.g. "
                    f"'piston.pneumatic_dashpot.orifice_area', not "
                    f"'piston.orifice_area').\n"
                    f"Valid model parameters:\n  - "
                    + "\n  - ".join(model_params)
                )

        for test in tests:
            if not getattr(test, "enabled", True):
                continue
            if isinstance(test, ParameterSweepSpec):
                _check(test.name, test.sweep_param)
            elif isinstance(test, (ParameterGridSpec, DOESweepSpec)):
                for key in test.params.keys():
                    _check(test.name, key)

    # --- Internal dispatch ---

    def _run_test(self, test: TestSpec, backend_or_pool: Any) -> Any:
        from numen.bridge.server_backend import JuliaServerPool
        exc    = self.config.excitation
        e_id   = exc.entity
        e_port = exc.port
        out    = self._output_key

        # Sweep tests get the pool/backend so they can parallelise at DOE level.
        if isinstance(test, ParameterSweepSpec):
            return self._run_parameter_sweep(test, backend_or_pool)
        if isinstance(test, ParameterGridSpec):
            return self._run_parameter_grid(test, backend_or_pool)
        if isinstance(test, DOESweepSpec):
            return self._run_doe_sweep(test, backend_or_pool)

        # Leaf tests run on a single server.  When a pool is active, borrow one.
        if isinstance(backend_or_pool, JuliaServerPool):
            with backend_or_pool.borrow() as backend:
                return self._run_leaf_test(test, backend, e_id, e_port, out)
        return self._run_leaf_test(test, backend_or_pool, e_id, e_port, out)

    def _run_leaf_test(self, test: TestSpec, backend: Any,
                       e_id: str, e_port: str, out: str) -> Any:
        if isinstance(test, DiscreteFrequencySweepSpec):
            from numen.characterization.tests.freq_sweep import run_discrete_frequency_sweep
            exc_s = set_excitation_params(self._exc_spec, e_id, e_port, dc=test.dc_offset)
            return run_discrete_frequency_sweep(test, exc_s, e_id, e_port, out, backend)

        if isinstance(test, DCOperatingPointSweepSpec):
            from numen.characterization.tests.dc_sweep import run_dc_operating_point_sweep
            return run_dc_operating_point_sweep(test, self._exc_spec, e_id, e_port, out, backend)

        if isinstance(test, AmplitudeSweepSpec):
            from numen.characterization.tests.amplitude_sweep import run_amplitude_sweep
            exc_s = set_excitation_params(self._exc_spec, e_id, e_port, dc=test.dc_offset)
            return run_amplitude_sweep(test, exc_s, e_id, e_port, out, backend)

        if isinstance(test, ContinuousChirpSpec):
            from numen.characterization.tests.chirp_sweep import run_continuous_chirp
            exc_s = set_excitation_params(self._exc_spec, e_id, e_port, dc=test.dc_offset)
            return run_continuous_chirp(
                test, exc_s, e_id, e_port,
                self._exc_component_kind, self._exc_target_field, out, backend,
            )

        if isinstance(test, TwoToneSpec):
            from numen.characterization.tests.two_tone import run_two_tone
            exc_s = set_excitation_params(self._exc_spec, e_id, e_port, dc=0.0)
            return run_two_tone(
                test, exc_s, e_id, e_port,
                self._exc_component_kind, self._exc_target_field, out, backend,
            )

        if isinstance(test, HarmonicDistortionSweepSpec):
            from numen.characterization.tests.harmonic_distortion import run_harmonic_distortion_sweep
            exc_s = set_excitation_params(self._exc_spec, e_id, e_port, dc=test.dc_offset)
            return run_harmonic_distortion_sweep(test, exc_s, e_id, e_port, out, backend)

        if isinstance(test, FreeDecaySpec):
            from numen.characterization.tests.free_decay import run_free_decay
            return run_free_decay(
                test, self._base_spec, e_id, e_port,
                self._exc_component_kind, self._exc_target_field, out, backend,
            )

        if isinstance(test, PhasePortraitSpec):
            from numen.characterization.tests.phase_portrait import run_phase_portrait
            exc_s = set_excitation_params(self._exc_spec, e_id, e_port, dc=0.0)
            return run_phase_portrait(
                test, exc_s, e_id, e_port,
                self._exc_component_kind, self._exc_target_field, out, backend,
            )

        if isinstance(test, StochasticExcitationSpec):
            from numen.characterization.tests.stochastic_excitation import run_stochastic_excitation
            return run_stochastic_excitation(
                test, self._base_spec, e_id, e_port,
                self._exc_component_kind, self._exc_target_field, out, backend,
                plan_dir=self._plan_dir,
                global_seed=self._global_seed,
            )

        _log.warning("Unknown test type '%s'. Skipping '%s'.", test.type, test.name)
        return None

    def _make_sub_runner_factory(self, sub_spec_obj: Any, context: str) -> Any:
        """Return Callable[[backend, spec_v], result] — backend supplied at call time.

        The factory captures excitation info and sub_spec_obj but NOT the backend,
        so the same factory can be handed to a pool for parallel dispatch (each
        call supplies a different server from the pool).
        """
        e_id   = self.config.excitation.entity
        e_port = self.config.excitation.port
        out    = self._output_key

        def _runner(backend: Any, spec_v: Any) -> Any:
            if isinstance(sub_spec_obj, DiscreteFrequencySweepSpec):
                from numen.characterization.tests.freq_sweep import run_discrete_frequency_sweep
                return run_discrete_frequency_sweep(sub_spec_obj, spec_v, e_id, e_port, out, backend)
            if isinstance(sub_spec_obj, DCOperatingPointSweepSpec):
                from numen.characterization.tests.dc_sweep import run_dc_operating_point_sweep
                return run_dc_operating_point_sweep(sub_spec_obj, spec_v, e_id, e_port, out, backend)
            if isinstance(sub_spec_obj, AmplitudeSweepSpec):
                from numen.characterization.tests.amplitude_sweep import run_amplitude_sweep
                return run_amplitude_sweep(sub_spec_obj, spec_v, e_id, e_port, out, backend)
            if isinstance(sub_spec_obj, ContinuousChirpSpec):
                from numen.characterization.tests.chirp_sweep import run_continuous_chirp
                ck  = self._exc_component_kind
                tgt = self._exc_target_field
                return run_continuous_chirp(sub_spec_obj, spec_v, e_id, e_port, ck, tgt, out, backend)
            if isinstance(sub_spec_obj, TwoToneSpec):
                from numen.characterization.tests.two_tone import run_two_tone
                ck  = self._exc_component_kind
                tgt = self._exc_target_field
                return run_two_tone(sub_spec_obj, spec_v, e_id, e_port, ck, tgt, out, backend)
            if isinstance(sub_spec_obj, HarmonicDistortionSweepSpec):
                from numen.characterization.tests.harmonic_distortion import run_harmonic_distortion_sweep
                return run_harmonic_distortion_sweep(sub_spec_obj, spec_v, e_id, e_port, out, backend)
            if isinstance(sub_spec_obj, PhasePortraitSpec):
                from numen.characterization.tests.phase_portrait import run_phase_portrait
                tgt = self._exc_target_field
                ck  = self._exc_component_kind
                return run_phase_portrait(sub_spec_obj, spec_v, e_id, e_port, ck, tgt, out, backend)
            raise NotImplementedError(
                f"'{context}' sub_test type '{sub_spec_obj.type}' not supported as sub_test. "
                f"Supported: discrete_frequency_sweep, dc_operating_point_sweep, "
                f"amplitude_sweep, continuous_chirp, two_tone, harmonic_distortion_sweep, "
                f"phase_portrait"
            )

        return _runner

    def _get_sub_spec(self, test_name: str, parent: str) -> Any:
        sub = next((t for t in self.config.tests if t.name == test_name), None)
        if sub is None:
            raise ValueError(f"'{parent}': sub_test '{test_name}' not found")
        return sub

    def _run_parameter_sweep(self, test: ParameterSweepSpec, backend_or_pool: Any) -> Any:
        from numen.bridge.server_backend import JuliaServerPool
        sub  = self._get_sub_spec(test.sub_test, test.name)
        exc  = self.config.excitation
        factory = self._make_sub_runner_factory(sub, test.name)
        if isinstance(backend_or_pool, JuliaServerPool):
            from numen.characterization.tests.param_sweep import run_parameter_sweep_parallel
            return run_parameter_sweep_parallel(
                test, self._exc_spec, factory, backend_or_pool,
                entity_id=exc.entity, port_name=exc.port,
            )
        from numen.characterization.tests.param_sweep import run_parameter_sweep
        return run_parameter_sweep(
            test, self._exc_spec,
            lambda spec_v: factory(backend_or_pool, spec_v),
            entity_id=exc.entity, port_name=exc.port,
        )

    def _run_parameter_grid(self, test: ParameterGridSpec, backend_or_pool: Any) -> Any:
        from numen.bridge.server_backend import JuliaServerPool
        sub  = self._get_sub_spec(test.sub_test, test.name)
        exc  = self.config.excitation
        factory = self._make_sub_runner_factory(sub, test.name)
        if isinstance(backend_or_pool, JuliaServerPool):
            from numen.characterization.tests.param_grid import run_parameter_grid_parallel
            return run_parameter_grid_parallel(
                test, self._exc_spec, factory, backend_or_pool,
                entity_id=exc.entity, port_name=exc.port,
            )
        from numen.characterization.tests.param_grid import run_parameter_grid
        return run_parameter_grid(
            test, self._exc_spec,
            lambda spec_v: factory(backend_or_pool, spec_v),
            entity_id=exc.entity, port_name=exc.port,
        )

    def _run_doe_sweep(self, test: DOESweepSpec, backend_or_pool: Any) -> Any:
        from numen.bridge.server_backend import JuliaServerPool
        sub  = self._get_sub_spec(test.sub_test, test.name)
        exc  = self.config.excitation
        factory = self._make_sub_runner_factory(sub, test.name)
        if isinstance(backend_or_pool, JuliaServerPool):
            from numen.characterization.tests.doe_sweep import run_doe_sweep_parallel
            return run_doe_sweep_parallel(
                test, self._exc_spec, factory, backend_or_pool,
                entity_id=exc.entity, port_name=exc.port,
            )
        from numen.characterization.tests.doe_sweep import run_doe_sweep
        return run_doe_sweep(
            test, self._exc_spec,
            lambda spec_v: factory(backend_or_pool, spec_v),
            entity_id=exc.entity, port_name=exc.port,
        )
