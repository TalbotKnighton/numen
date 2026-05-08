"""Parameter sweep test runner — repeats a sub-test for each value of a model parameter."""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Callable

from numen.characterization.results import ParameterFamilyResult
from numen.characterization.schema import ParameterSweepSpec

_log = logging.getLogger("numen.characterization.tests.param_sweep")


def _set_model_param(spec: Any, key: str, value: float) -> Any:
    """Return a new spec with one ParameterField value updated.

    key is a dot-separated path, e.g. "osc.c1".  Raises KeyError if the key
    is not in param_index_map (wrong name, or it's a state field).
    """
    if key not in spec.param_index_map:
        raise KeyError(
            f"Parameter '{key}' not in param_index_map. "
            f"Available: {sorted(spec.param_index_map)}"
        )
    new_p = list(spec.p)
    idx   = spec.param_index_map[key][0]
    new_p[idx] = value
    return replace(spec, p=new_p)


def run_parameter_sweep(
    test: ParameterSweepSpec,
    exc_spec: Any,
    sub_test_runner: Callable[[Any], Any],
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
        spec_v = _set_model_param(exc_spec, test.sweep_param, val)
        sub    = sub_test_runner(spec_v)
        result_obj.sub_results.append(sub)
        _log.debug("%s=%s  sub-result: %s", test.sweep_param, val, type(sub).__name__)

    _log.info(
        "Parameter sweep '%s' (%s) done: %d values",
        test.name, test.sweep_param, len(test.values),
    )
    return result_obj
