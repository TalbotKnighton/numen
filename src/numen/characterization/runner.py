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
from pathlib import Path
from typing import Any, Generator

from numen.characterization.schema import (
    BackendSpec,
    CharacterizationConfig,
    ExcitationSpec,
    ModelSpec,
    TestSpec,
)
from numen.characterization.excitation import find_excitation_ports, inject_excitation
from numen.compiler.flatten import compile_spec

_log = logging.getLogger("numen.characterization.runner")


# ---------------------------------------------------------------------------
# Result container (grows in Phase 2 with per-test typed results)
# ---------------------------------------------------------------------------

class CampaignResults:
    """Accumulates results across a test campaign.

    In Phase 1 this is a minimal container.  Phase 2 will add per-test typed
    result objects, a to_dataframe() method, and JSON serialisation.
    """

    def __init__(self, config: CharacterizationConfig) -> None:
        self.config = config
        self._results: list[dict[str, Any]] = []

    def append(self, test_name: str, data: Any) -> None:
        self._results.append({"test": test_name, "data": data})

    def __len__(self) -> int:
        return len(self._results)

    def __repr__(self) -> str:
        return f"CampaignResults({len(self)} tests completed)"

    def save(self, path: str | Path) -> None:
        """Stub — full serialisation implemented in Phase 2."""
        raise NotImplementedError("Result serialisation is a Phase 2 feature")


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def _open_backend(spec: BackendSpec) -> Any:
    """Instantiate the correct backend from a BackendSpec.

    Returns the backend object.  For julia_server, the caller is responsible
    for using it as a context manager (or calling close() when done).
    """
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
    alive).  All other backends are plain objects — nullcontext wraps them so
    the calling code is identical in both cases.
    """
    backend = _open_backend(spec)
    if spec.type == "julia_server":
        with backend:
            _log.info("Julia server started (eager=False; starts on first solve)")
            yield backend
    else:
        with nullcontext(backend):
            yield backend


# ---------------------------------------------------------------------------
# World / spec construction from config
# ---------------------------------------------------------------------------

def _build_world(model_spec: ModelSpec) -> Any:
    """Import the user's factory module and call make_world (or the named factory)."""
    module = importlib.import_module(model_spec.module)
    factory = getattr(module, model_spec.factory)
    return factory(**model_spec.factory_kwargs)


def _build_excitation_spec(
    base_spec: Any,
    world: Any,
    exc_spec: ExcitationSpec,
    amp: float,
    freq: float,
    dc: float,
) -> Any:
    """Inject excitation into a compiled spec, resolving the target_field from the annotation."""
    ports = find_excitation_ports(world, exc_spec.entity)
    if exc_spec.port not in ports:
        available = list(ports.keys())
        raise ValueError(
            f"Entity '{exc_spec.entity}' has no ExcitationPort named '{exc_spec.port}'. "
            f"Available ports: {available}"
        )
    port = ports[exc_spec.port]
    return inject_excitation(
        base_spec,
        entity_id    = exc_spec.entity,
        port_name    = exc_spec.port,
        target_field = port.targets,
        amp          = amp,
        freq         = freq,
        dc           = dc,
    )


# ---------------------------------------------------------------------------
# CharacterizationRunner
# ---------------------------------------------------------------------------

class CharacterizationRunner:
    """Orchestrates a full characterization campaign from a validated config.

    Phase 1: validates config, builds world, opens backend, provides a
    compile-and-inject helper.  Individual test runners are added in Phase 2.
    """

    def __init__(self, config: CharacterizationConfig) -> None:
        self.config = config
        self._world = _build_world(config.model)
        self._base_spec = compile_spec(self._world)
        _log.info(
            "Runner ready: %d state fields, %d param fields, %d tests",
            self._base_spec.state_size,
            self._base_spec.param_size,
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
        """Execute all tests in the campaign.  Returns collected results.

        Phase 1: iterates tests and logs what would be run.
        Phase 2: dispatches to concrete test runner implementations.
        """
        results = CampaignResults(self.config)
        with _backend_context(self.config.backend) as backend:
            for test in self.config.tests:
                _log.info("Running test '%s' (type=%s)", test.name, test.type)
                result = self._run_test(test, backend)
                results.append(test.name, result)
                _log.info("Test '%s' complete", test.name)
        return results

    def compiled_spec_for(
        self,
        amp: float = 0.0,
        freq: float = 1.0,
        dc: float = 0.0,
    ) -> Any:
        """Return a compiled spec with excitation injected at the given parameters.

        Convenience method for interactive / scripted use outside of run().
        """
        return _build_excitation_spec(
            self._base_spec,
            self._world,
            self.config.excitation,
            amp=amp, freq=freq, dc=dc,
        )

    # --- Internal dispatch (Phase 2 fills these in) ---

    def _run_test(self, test: TestSpec, backend: Any) -> Any:
        """Dispatch to the appropriate test runner.  Phase 2 implementation."""
        _log.warning(
            "Test type '%s' is not yet implemented (Phase 2). "
            "Returning None for test '%s'.",
            test.type, test.name,
        )
        return None
