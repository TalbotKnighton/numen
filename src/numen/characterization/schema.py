"""Pydantic schema for characterization test plans.

YAML and JSON are supported as input formats; both are validated through these
models.  The canonical loading pattern is::

    config = CharacterizationConfig.from_yaml("test_plan.yaml")
    config = CharacterizationConfig.from_json("test_plan.json")
    config = CharacterizationConfig.model_validate(some_dict)

Design principles:
- Test types are signal-level (step, sweep, chirp, DOE) — never domain-specific.
- Domain knowledge lives in the model's metrics registry, not here.
- All test specs use Pydantic discriminated unions on the `type` field.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class BackendSpec(BaseModel):
    """Which solver backend to use for all solves in this campaign."""
    type:       Literal["scipy", "jax", "julia", "julia_server"] = "scipy"
    julia_file: str | None = None    # path to .jl dynamics file (julia / julia_server only)
    method:     str | None = None    # solver name, e.g. "Dopri5", "Rodas5P"
    rtol:       float = 1e-8
    atol:       float = 1e-9

    @model_validator(mode="after")
    def _julia_needs_file(self) -> "BackendSpec":
        if self.type in ("julia", "julia_server") and self.julia_file is None:
            raise ValueError(f"backend.type={self.type!r} requires julia_file to be set")
        return self


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ModelSpec(BaseModel):
    """How to construct the Numen world under test."""
    module:          str                  # importable module path, e.g. "examples.nonlinear_oscillator.world"
    factory:         str                  # callable in that module, e.g. "make_world"
    factory_kwargs:  dict[str, Any] = {}  # keyword arguments forwarded to the factory
    metrics:         str | None = None    # optional module path containing a METRICS dict


# ---------------------------------------------------------------------------
# Excitation
# ---------------------------------------------------------------------------

class ExcitationSpec(BaseModel):
    """Which entity / port to drive and which state to measure."""
    entity:       str   # entity_id in the world, e.g. "osc"
    port:         str   # ExcitationPort field name on the component, e.g. "force"
    output_state: str   # state field to measure, e.g. "position"


# ---------------------------------------------------------------------------
# Shared sub-specs
# ---------------------------------------------------------------------------

class FrequencyGridSpec(BaseModel):
    """Log or linear frequency grid for discrete swept-sine tests."""
    spacing:  Literal["log", "linear"] = "log"
    f_start:  float
    f_end:    float
    n_points: int = 40

    @model_validator(mode="after")
    def _check_range(self) -> "FrequencyGridSpec":
        if self.f_start <= 0:
            raise ValueError("f_start must be > 0")
        if self.f_end <= self.f_start:
            raise ValueError("f_end must be > f_start")
        return self


class DOEParamSpec(BaseModel):
    """Continuous parameter range for a DOE sweep."""
    min:   float
    max:   float
    scale: Literal["linear", "log"] = "linear"

    @model_validator(mode="after")
    def _check_range(self) -> "DOEParamSpec":
        if self.min >= self.max:
            raise ValueError("DOEParamSpec: min must be < max")
        if self.scale == "log" and self.min <= 0:
            raise ValueError("DOEParamSpec: min must be > 0 for log scale")
        return self


# ---------------------------------------------------------------------------
# Test specs
# ---------------------------------------------------------------------------

class DiscreteFrequencySweepSpec(BaseModel):
    """Stepped sine: one solve per frequency, extract amplitude and phase."""
    name:           str
    type:           Literal["discrete_frequency_sweep"]
    frequencies:    FrequencyGridSpec
    amplitude:      float
    dc_offset:      float = 0.0
    settle_periods: int   = 10   # cycles to discard before measuring
    measure_periods: int  = 5    # cycles to use for lock-in extraction


class DCOperatingPointSweepSpec(BaseModel):
    """Sweep DC bias; measure small-signal FRF at each operating point.

    Maps out how effective damping and stiffness shift with bias — the
    first-order fingerprint of a nonlinearity.
    """
    name:            str
    type:            Literal["dc_operating_point_sweep"]
    dc_values:       list[float]
    probe_frequency: float         # Hz — typically near resonance
    probe_amplitude: float         # small relative to operating point
    settle_periods:  int = 10
    measure_periods: int = 5


class AmplitudeSweepSpec(BaseModel):
    """Fixed frequency, varying drive amplitude — nonlinearity signature."""
    name:           str
    type:           Literal["amplitude_sweep"]
    frequency:      float
    amplitudes:     list[float]
    dc_offset:      float = 0.0
    settle_periods: int   = 10
    measure_periods: int  = 5


class ParameterSweepSpec(BaseModel):
    """Repeat a named test for each value of a model ParameterField."""
    name:        str
    type:        Literal["parameter_sweep"]
    sweep_param: str          # dot-separated key, e.g. "osc.c1"
    values:      list[float]
    sub_test:    str          # name of another test in the same plan


class ParameterGridSpec(BaseModel):
    """Full or paired factorial over explicit value lists for multiple params."""
    name:     str
    type:     Literal["parameter_grid"]
    params:   dict[str, list[float]]   # param_key → list of values
    sub_test: str
    mode:     Literal["full_factorial", "pairs"] = "full_factorial"


class DOESweepSpec(BaseModel):
    """Space-filling or classical DOE over continuous parameter ranges.

    Uses scipy.stats.qmc (LHS, Sobol, Halton) or pyDOE3 (CCD, Box-Behnken).
    Install optional extras: ``uv pip install "numen[characterization]"``
    """
    name:    str
    type:    Literal["doe_sweep"]
    design:  Literal[
        "latin_hypercube", "sobol", "halton",
        "central_composite", "box_behnken", "full_factorial",
    ] = "latin_hypercube"
    n_samples: int = 50
    params:    dict[str, DOEParamSpec]
    sub_test:  str
    outputs:   list[str] = []   # scalar metric names to tabulate per design point


class ContinuousChirpSpec(BaseModel):
    """Single solve with a frequency-swept (chirp) input signal.

    Faster than stepped sine (one solve vs N), lower SNR and resolution.
    """
    name:       str
    type:       Literal["continuous_chirp"]
    f_start:    float
    f_end:      float
    duration:   float
    amplitude:  float
    dc_offset:  float                      = 0.0
    chirp_type: Literal["log", "linear"]   = "log"


# ---------------------------------------------------------------------------
# Discriminated union of all test types
# ---------------------------------------------------------------------------

TestSpec = Annotated[
    Union[
        DiscreteFrequencySweepSpec,
        DCOperatingPointSweepSpec,
        AmplitudeSweepSpec,
        ParameterSweepSpec,
        ParameterGridSpec,
        DOESweepSpec,
        ContinuousChirpSpec,
    ],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class CharacterizationConfig(BaseModel):
    """Complete characterization test plan."""
    version:    str            = "1.0"
    backend:    BackendSpec
    model:      ModelSpec
    excitation: ExcitationSpec
    tests:      list[TestSpec]

    @model_validator(mode="after")
    def _unique_test_names(self) -> "CharacterizationConfig":
        names = [t.name for t in self.tests]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"Duplicate test names: {sorted(dupes)}")
        return self

    @model_validator(mode="after")
    def _sub_tests_exist(self) -> "CharacterizationConfig":
        names = {t.name for t in self.tests}
        for t in self.tests:
            ref = getattr(t, "sub_test", None)
            if ref and ref not in names:
                raise ValueError(
                    f"Test '{t.name}' references sub_test '{ref}' which does not exist"
                )
        return self

    # --- Loaders ---

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CharacterizationConfig":
        """Load and validate a YAML test plan."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML is required to load YAML plans: pip install pyyaml") from e
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))

    @classmethod
    def from_json(cls, path: str | Path) -> "CharacterizationConfig":
        """Load and validate a JSON test plan."""
        import json
        with open(path) as f:
            return cls.model_validate(json.load(f))
