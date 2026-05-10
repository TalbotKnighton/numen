"""DOE sweep runner — space-filling and classical designs over continuous parameter ranges.

Supported designs
-----------------
Space-filling (scipy.stats.qmc — always available with scipy):
    latin_hypercube, sobol, halton

Classical (pyDOE3 — optional: pip install pyDOE3):
    central_composite, box_behnken

Grid-based (no extra deps):
    full_factorial   — n_points per axis, where n_points = ceil(n_samples^(1/n_params))
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable

import numpy as np

from numen.characterization.results import DOESweepResult
from numen.characterization.schema import DOEParamSpec, DOESweepSpec
from numen.characterization.tests.param_sweep import _set_model_param

_log = logging.getLogger("numen.characterization.tests.doe_sweep")


# ---------------------------------------------------------------------------
# Design generation
# ---------------------------------------------------------------------------

def _scale_unit_to_param(unit_val: float, spec: DOEParamSpec) -> float:
    """Map a value in [0, 1] to the actual parameter range.

    Linear: val = min + (max - min) * unit_val
    Log:    val = min * (max/min) ** unit_val
    """
    if spec.scale == "log":
        return spec.min * (spec.max / spec.min) ** unit_val
    return spec.min + (spec.max - spec.min) * unit_val


def _scale_coded_to_param(coded_val: float, spec: DOEParamSpec) -> float:
    """Map a coded value in [-1, 1] to the actual parameter range (for CCD / BBD).

    Coded values outside [-1, 1] (CCD axial points) are clipped to the range.
    """
    unit = np.clip((coded_val + 1.0) / 2.0, 0.0, 1.0)
    return _scale_unit_to_param(unit, spec)


def _generate_design(test: DOESweepSpec) -> list[dict[str, float]]:
    """Return a list of design-point dicts for the given DOE spec."""
    param_keys = list(test.params.keys())
    n_params   = len(param_keys)

    # ---- Space-filling designs via scipy.stats.qmc ----
    if test.design in ("latin_hypercube", "sobol", "halton"):
        from scipy.stats import qmc

        if test.design == "latin_hypercube":
            sampler = qmc.LatinHypercube(d=n_params, seed=42)
        elif test.design == "sobol":
            sampler = qmc.Sobol(d=n_params, scramble=True, seed=42)
        else:
            sampler = qmc.Halton(d=n_params, scramble=True, seed=42)

        samples = sampler.random(n=test.n_samples)   # (n_samples, n_params) in [0, 1]
        combinations = []
        for row in samples:
            combinations.append({
                k: _scale_unit_to_param(float(row[i]), test.params[k])
                for i, k in enumerate(param_keys)
            })
        return combinations

    # ---- Full-factorial grid ----
    if test.design == "full_factorial":
        import itertools
        n_per_axis = max(2, round(test.n_samples ** (1.0 / n_params)))
        axes = []
        for k in param_keys:
            spec = test.params[k]
            if spec.scale == "log":
                axes.append(np.geomspace(spec.min, spec.max, n_per_axis).tolist())
            else:
                axes.append(np.linspace(spec.min, spec.max, n_per_axis).tolist())
        combinations = [
            dict(zip(param_keys, combo))
            for combo in itertools.product(*axes)
        ]
        _log.debug(
            "full_factorial: %d axes × %d points = %d design points",
            n_params, n_per_axis, len(combinations),
        )
        return combinations

    # ---- Classical designs via pyDOE3 ----
    try:
        import pyDOE3 as doe
    except ImportError as e:
        raise ImportError(
            f"Design '{test.design}' requires pyDOE3: pip install pyDOE3"
        ) from e

    if test.design == "central_composite":
        coded = doe.ccdesign(n_params, criterion="c", face="ccc")
    elif test.design == "box_behnken":
        if n_params < 3:
            raise ValueError("Box-Behnken requires at least 3 parameters")
        coded = doe.bbdesign(n_params, center=1)
    else:
        raise ValueError(f"Unknown DOE design: {test.design!r}")

    combinations = []
    for row in coded:
        combinations.append({
            k: _scale_coded_to_param(float(row[i]), test.params[k])
            for i, k in enumerate(param_keys)
        })
    return combinations


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_doe_sweep(
    test: DOESweepSpec,
    exc_spec: Any,
    sub_test_runner: Callable[[Any], Any],
    entity_id: str | None = None,
    port_name: str | None = None,
) -> DOESweepResult:
    """Run a sub-test at each DOE design point.

    Args:
        test:            Validated DOESweepSpec.
        exc_spec:        CompiledSpec with excitation injected.
        sub_test_runner: Callable(CompiledSpec) → result.

    Returns:
        DOESweepResult with one sub-result per design point.
    """
    combinations = _generate_design(test)
    _log.info(
        "DOE sweep '%s' [%s]: %d design points (%d params)",
        test.name, test.design, len(combinations), len(test.params),
    )

    result_obj = DOESweepResult(
        name         = test.name,
        design       = test.design,
        param_keys   = list(test.params.keys()),
        combinations = combinations,
    )

    for i, combo in enumerate(combinations):
        spec_v = exc_spec
        for key, val in combo.items():
            spec_v = _set_model_param(spec_v, key, val, entity_id, port_name)

        sub = sub_test_runner(spec_v)
        result_obj.sub_results.append(sub)
        _log.debug(
            "DOE point %d/%d: %s → %s",
            i + 1, len(combinations),
            {k: f"{v:.4g}" for k, v in combo.items()},
            type(sub).__name__,
        )

    _log.info("DOE sweep '%s' done: %d points", test.name, len(combinations))
    return result_obj


def run_doe_sweep_parallel(
    test: DOESweepSpec,
    exc_spec: Any,
    factory: Callable[[Any, Any], Any],
    pool: Any,
    entity_id: str | None = None,
    port_name: str | None = None,
) -> DOESweepResult:
    """Parallel variant — dispatches each design point to the pool concurrently."""
    combinations = _generate_design(test)
    _log.info(
        "DOE sweep '%s' [%s, parallel, %d workers]: %d points (%d params)",
        test.name, test.design, pool.n_workers, len(combinations), len(test.params),
    )

    all_specs = []
    for combo in combinations:
        spec_v = exc_spec
        for key, val in combo.items():
            spec_v = _set_model_param(spec_v, key, val, entity_id, port_name)
        all_specs.append(spec_v)

    sub_results = pool.map(factory, all_specs)

    result_obj = DOESweepResult(
        name         = test.name,
        design       = test.design,
        param_keys   = list(test.params.keys()),
        combinations = combinations,
    )
    result_obj.sub_results.extend(sub_results)
    _log.info("DOE sweep '%s' done [parallel]: %d points", test.name, len(combinations))
    return result_obj
