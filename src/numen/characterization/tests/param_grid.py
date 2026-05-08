"""Parameter grid test runner — full factorial or pairwise sweep over explicit value lists."""
from __future__ import annotations

import itertools
import logging
from typing import Any, Callable

from numen.characterization.results import ParameterGridResult
from numen.characterization.schema import ParameterGridSpec
from numen.characterization.tests.param_sweep import _set_model_param

_log = logging.getLogger("numen.characterization.tests.param_grid")


def _build_combinations(
    params: dict[str, list[float]],
    mode: str,
) -> list[dict[str, float]]:
    """Return the list of design-point dicts for the given mode.

    full_factorial: Cartesian product of all value lists.
    pairs:          Zip the value lists in order (cycling shorter lists to the
                    length of the longest list).
    """
    param_keys = list(params.keys())
    param_vals = [params[k] for k in param_keys]

    if mode == "full_factorial":
        combinations = [
            dict(zip(param_keys, combo))
            for combo in itertools.product(*param_vals)
        ]
    elif mode == "pairs":
        max_len      = max(len(v) for v in param_vals)
        combinations = [
            {k: vals[i % len(vals)] for k, vals in zip(param_keys, param_vals)}
            for i in range(max_len)
        ]
    else:
        raise ValueError(f"Unknown ParameterGrid mode: {mode!r}")

    return combinations


def run_parameter_grid(
    test: ParameterGridSpec,
    exc_spec: Any,
    sub_test_runner: Callable[[Any], Any],
    entity_id: str | None = None,
    port_name: str | None = None,
) -> ParameterGridResult:
    """Run a sub-test at each point of a parameter grid.

    Args:
        test:            Validated ParameterGridSpec.
        exc_spec:        CompiledSpec with excitation injected.
        sub_test_runner: Callable(CompiledSpec) → result.  Closes over backend
                         and excitation info (built by CharacterizationRunner).

    Returns:
        ParameterGridResult with one sub-result per design point.
    """
    combinations = _build_combinations(test.params, test.mode)
    _log.info(
        "Parameter grid '%s' [%s]: %d points (%d params)",
        test.name, test.mode, len(combinations), len(test.params),
    )

    result_obj = ParameterGridResult(
        name         = test.name,
        param_keys   = list(test.params.keys()),
        combinations = combinations,
        mode         = test.mode,
    )

    for i, combo in enumerate(combinations):
        spec_v = exc_spec
        for key, val in combo.items():
            spec_v = _set_model_param(spec_v, key, val, entity_id, port_name)

        sub = sub_test_runner(spec_v)
        result_obj.sub_results.append(sub)
        _log.debug(
            "Grid point %d/%d: %s → %s",
            i + 1, len(combinations),
            {k: f"{v:.4g}" for k, v in combo.items()},
            type(sub).__name__,
        )

    _log.info("Parameter grid '%s' done: %d points", test.name, len(combinations))
    return result_obj
