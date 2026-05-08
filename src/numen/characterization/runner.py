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
    ModelSpec,
    ParameterGridSpec,
    ParameterSweepSpec,
    TestSpec,
)
from numen.characterization.excitation import find_excitation_ports, inject_excitation
from numen.characterization.results import CampaignResults
from numen.compiler.flatten import compile_spec

_log = logging.getLogger("numen.characterization.runner")


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def _open_backend(spec: BackendSpec) -> Any:
    if spec.type == "scipy":
        from numen.bridge.scipy_backend import ScipyBackend
        kwargs: dict[str, Any] = {"rtol": spec.rtol, "atol": spec.atol}
        if spec.method:
            kwargs["method"] = spec.method
        return ScipyBackend(**kwargs)

    if spec.type == "jax":
        from numen.bridge.jax_backend import JAXBackend
        kwargs = {"rtol": spec.rtol, "atol": spec.atol}
        if spec.method:
            kwargs["solver"] = spec.method
        return JAXBackend(**kwargs)

    if spec.type == "julia":
        from numen.bridge.runtime import JuliaBackend
        kwargs = {"julia_file": spec.julia_file, "rtol": spec.rtol, "atol": spec.atol}
        if spec.method:
            kwargs["method"] = spec.method
        return JuliaBackend(**kwargs)

    if spec.type == "julia_server":
        from numen.bridge.server_backend import JuliaServerBackend
        kwargs = {"julia_file": spec.julia_file, "rtol": spec.rtol, "atol": spec.atol}
        if spec.method:
            kwargs["method"] = spec.method
        return JuliaServerBackend(**kwargs)

    raise ValueError(f"Unknown backend type: {spec.type!r}")


@contextmanager
def _backend_context(spec: BackendSpec) -> Generator[Any, None, None]:
    """Context manager that owns the backend lifecycle.

    julia_server backends are used as context managers (keeps the subprocess
    alive).  All other backends are plain objects — nullcontext wraps them.
    """
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
    ports = find_excitation_ports(world, exc.entity)
    if exc.port not in ports:
        raise ValueError(
            f"Entity '{exc.entity}' has no ExcitationPort '{exc.port}'. "
            f"Available: {list(ports)}"
        )
    return inject_excitation(
        base_spec,
        entity_id    = exc.entity,
        port_name    = exc.port,
        target_field = ports[exc.port].targets,
        amp=0.0, freq=1.0, dc=0.0,
    )


# ---------------------------------------------------------------------------
# CharacterizationRunner
# ---------------------------------------------------------------------------

class CharacterizationRunner:
    """Orchestrates a full characterization campaign from a validated config."""

    def __init__(self, config: CharacterizationConfig) -> None:
        self.config      = config
        self._world      = _build_world(config.model)
        self._base_spec  = compile_spec(self._world)
        self._exc_spec   = _make_exc_spec(self._base_spec, self._world, config.excitation)
        self._output_key = f"{config.excitation.entity}.{config.excitation.output_state}"
        _log.info(
            "Runner ready: state_size=%d  param_size=%d  tests=%d",
            self._exc_spec.state_size,
            self._exc_spec.param_size,
            len(config.tests),
        )

    # --- Constructors ---

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CharacterizationRunner":
        return cls(CharacterizationConfig.from_yaml(path))

    @classmethod
    def from_json(cls, path: str | Path) -> "CharacterizationRunner":
        return cls(CharacterizationConfig.from_json(path))

    @classmethod
    def from_config(cls, config: CharacterizationConfig) -> "CharacterizationRunner":
        return cls(config)

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

    # --- Internal dispatch ---

    def _run_test(self, test: TestSpec, backend: Any) -> Any:
        exc    = self.config.excitation
        e_id   = exc.entity
        e_port = exc.port
        out    = self._output_key

        if isinstance(test, DiscreteFrequencySweepSpec):
            from numen.characterization.tests.freq_sweep import run_discrete_frequency_sweep
            return run_discrete_frequency_sweep(
                test, self._exc_spec, e_id, e_port, out, backend,
            )

        if isinstance(test, DCOperatingPointSweepSpec):
            from numen.characterization.tests.dc_sweep import run_dc_operating_point_sweep
            return run_dc_operating_point_sweep(
                test, self._exc_spec, e_id, e_port, out, backend,
            )

        if isinstance(test, AmplitudeSweepSpec):
            from numen.characterization.tests.amplitude_sweep import run_amplitude_sweep
            return run_amplitude_sweep(
                test, self._exc_spec, e_id, e_port, out, backend,
            )

        if isinstance(test, ContinuousChirpSpec):
            from numen.characterization.tests.chirp_sweep import run_continuous_chirp
            return run_continuous_chirp(
                test, self._exc_spec, e_id, e_port, out, backend,
            )

        if isinstance(test, ParameterSweepSpec):
            return self._run_parameter_sweep(test, backend)

        if isinstance(test, ParameterGridSpec):
            return self._run_parameter_grid(test, backend)

        if isinstance(test, DOESweepSpec):
            return self._run_doe_sweep(test, backend)

        _log.warning("Unknown test type '%s'. Skipping '%s'.", test.type, test.name)
        return None

    def _make_sub_runner(
        self,
        sub_spec_obj: Any,
        backend: Any,
        context: str,
    ) -> Any:
        """Return a closure that runs sub_spec_obj with the given backend/excitation info."""
        e_id   = self.config.excitation.entity
        e_port = self.config.excitation.port
        out    = self._output_key

        def _sub_runner(spec_v: Any) -> Any:
            if isinstance(sub_spec_obj, DiscreteFrequencySweepSpec):
                from numen.characterization.tests.freq_sweep import run_discrete_frequency_sweep
                return run_discrete_frequency_sweep(
                    sub_spec_obj, spec_v, e_id, e_port, out, backend,
                )
            if isinstance(sub_spec_obj, DCOperatingPointSweepSpec):
                from numen.characterization.tests.dc_sweep import run_dc_operating_point_sweep
                return run_dc_operating_point_sweep(
                    sub_spec_obj, spec_v, e_id, e_port, out, backend,
                )
            if isinstance(sub_spec_obj, AmplitudeSweepSpec):
                from numen.characterization.tests.amplitude_sweep import run_amplitude_sweep
                return run_amplitude_sweep(
                    sub_spec_obj, spec_v, e_id, e_port, out, backend,
                )
            if isinstance(sub_spec_obj, ContinuousChirpSpec):
                from numen.characterization.tests.chirp_sweep import run_continuous_chirp
                return run_continuous_chirp(
                    sub_spec_obj, spec_v, e_id, e_port, out, backend,
                )
            raise NotImplementedError(
                f"{context} sub_test type '{sub_spec_obj.type}' not supported. "
                f"Supported: discrete_frequency_sweep, dc_operating_point_sweep, "
                f"amplitude_sweep, continuous_chirp"
            )

        return _sub_runner

    def _get_sub_spec(self, test_name: str, parent: str) -> Any:
        sub = next((t for t in self.config.tests if t.name == test_name), None)
        if sub is None:
            raise ValueError(f"'{parent}': sub_test '{test_name}' not found")
        return sub

    def _run_parameter_sweep(self, test: ParameterSweepSpec, backend: Any) -> Any:
        from numen.characterization.tests.param_sweep import run_parameter_sweep
        sub_spec_obj = self._get_sub_spec(test.sub_test, test.name)
        exc = self.config.excitation
        return run_parameter_sweep(
            test, self._exc_spec,
            self._make_sub_runner(sub_spec_obj, backend, test.name),
            entity_id=exc.entity, port_name=exc.port,
        )

    def _run_parameter_grid(self, test: ParameterGridSpec, backend: Any) -> Any:
        from numen.characterization.tests.param_grid import run_parameter_grid
        sub_spec_obj = self._get_sub_spec(test.sub_test, test.name)
        exc = self.config.excitation
        return run_parameter_grid(
            test, self._exc_spec,
            self._make_sub_runner(sub_spec_obj, backend, test.name),
            entity_id=exc.entity, port_name=exc.port,
        )

    def _run_doe_sweep(self, test: DOESweepSpec, backend: Any) -> Any:
        from numen.characterization.tests.doe_sweep import run_doe_sweep
        sub_spec_obj = self._get_sub_spec(test.sub_test, test.name)
        exc = self.config.excitation
        return run_doe_sweep(
            test, self._exc_spec,
            self._make_sub_runner(sub_spec_obj, backend, test.name),
            entity_id=exc.entity, port_name=exc.port,
        )
