"""Parameter sweep test runner — repeats a sub-test for each value of a model parameter."""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Callable

from numen.characterization.results import ParameterFamilyResult
from numen.characterization.schema import ParameterSweepSpec

_log = logging.getLogger("numen.characterization.tests.param_sweep")


_EXC_PARAM_MAP = {
    "dc_offset":  "dc",
    "amplitude":  "amp",
    "frequency":  "freq",
}


def _resolve_param_key(
    key: str,
    entity_id: str | None = None,
    port_name: str | None = None,
) -> str:
    """Translate a user-facing parameter path to the internal param_index_map key.

    User-facing paths:
        ``excitation.dc_offset``  →  ``_exc_{entity_id}_{port_name}.dc``
        ``excitation.amplitude``  →  ``_exc_{entity_id}_{port_name}.amp``
        ``excitation.frequency``  →  ``_exc_{entity_id}_{port_name}.freq``
        ``osc.c1``                →  ``osc.c1``  (unchanged — model parameter)
    """
    if key.startswith("excitation."):
        if entity_id is None or port_name is None:
            raise ValueError(
                f"excitation.* parameter '{key}' requires entity_id and port_name "
                "to be provided to the sweep runner."
            )
        sub = key[len("excitation."):]
        if sub not in _EXC_PARAM_MAP:
            raise KeyError(
                f"Unknown excitation parameter '{sub}'. "
                f"Supported: {list(_EXC_PARAM_MAP)}"
            )
        return f"_exc_{entity_id}_{port_name}.{_EXC_PARAM_MAP[sub]}"
    return key


def _set_model_param(
    spec: Any,
    key: str,
    value: float,
    entity_id: str | None = None,
    port_name: str | None = None,
) -> Any:
    """Return a new spec with one parameter value updated.

    ``key`` may be a model ParameterField path (``"osc.c1"``) or an excitation
    parameter path (``"excitation.dc_offset"``).  The latter requires
    ``entity_id`` and ``port_name`` to resolve the internal key.
    """
    resolved = _resolve_param_key(key, entity_id, port_name)
    if resolved not in spec.param_index_map:
        # This path is normally unreachable: CharacterizationRunner validates
        # all parameter keys at campaign start. Reaching here indicates a key
        # was added via direct API usage (not a YAML campaign).
        raise KeyError(
            f"Parameter '{key}' not found in compiled spec. Use the full "
            f"three-level path 'entity.component_kind.field'."
        )
    new_p = list(spec.p)
    idx   = spec.param_index_map[resolved][0]
    new_p[idx] = value
    return replace(spec, p=new_p)


def run_parameter_sweep_parallel(
    test: ParameterSweepSpec,
    exc_spec: Any,
    factory: Callable[[Any, Any], Any],
    pool: Any,
    entity_id: str | None = None,
    port_name: str | None = None,
) -> ParameterFamilyResult:
    """Parallel variant — dispatches each value to the pool concurrently."""
    all_specs = [
        _set_model_param(exc_spec, test.sweep_param, val, entity_id, port_name)
        for val in test.values
    ]
    sub_results = pool.map(factory, all_specs)
    result_obj = ParameterFamilyResult(
        name         = test.name,
        sweep_param  = test.sweep_param,
        param_values = list(test.values),
    )
    result_obj.sub_results.extend(sub_results)
    _log.info(
        "Parameter sweep '%s' (%s) done [parallel, %d workers]: %d values",
        test.name, test.sweep_param, pool.n_workers, len(test.values),
    )
    return result_obj


def run_parameter_sweep(
    test: ParameterSweepSpec,
    exc_spec: Any,
    sub_test_runner: Callable[[Any], Any],
    entity_id: str | None = None,
    port_name: str | None = None,
) -> ParameterFamilyResult:
    """Run a sub-test for each value of a model ParameterField.

    Args:
        test:            Validated ParameterSweepSpec.
        exc_spec:        CompiledSpec with excitation injected.
        sub_test_runner: Callable that takes a CompiledSpec and returns a result.
                         The runner is responsible for passing the correct
                         backend / entity / output_key arguments via a closure
                         (built in CharacterizationRunner._run_test).

    Returns:
        ParameterFamilyResult with one sub-result per parameter value.
    """
    result_obj = ParameterFamilyResult(
        name        = test.name,
        sweep_param = test.sweep_param,
        param_values= list(test.values),
    )

    for val in test.values:
        spec_v = _set_model_param(exc_spec, test.sweep_param, val, entity_id, port_name)
        sub    = sub_test_runner(spec_v)
        result_obj.sub_results.append(sub)
        _log.debug("%s=%s  sub-result: %s", test.sweep_param, val, type(sub).__name__)

    _log.info(
        "Parameter sweep '%s' (%s) done: %d values",
        test.name, test.sweep_param, len(test.values),
    )
    return result_obj
