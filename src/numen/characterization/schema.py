"""Pydantic schema for characterization test plans.

YAML and JSON are supported as input formats; both are validated through these
models.  The canonical loading pattern is::

    config = CharacterizationConfig.from_yaml("test_plan.yaml")
    config = CharacterizationConfig.from_json("test_plan.json")
    config = CharacterizationConfig.model_validate(some_dict)

Design principles:
- Test types are signal-level (step, sweep, chirp, DOE) — never domain-specific.
- Domain knowledge lives in the model's metrics registry, not here.
- All test specs use Pydantic discriminated unions on the ``type`` field.
- The ``plots:`` section mirrors the ``tests:`` section; both live in the same
  YAML file so the experiment is fully self-contained.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class BackendSpec(BaseModel):
    """Which solver backend to use for all solves in this campaign."""
    type:       Literal["scipy", "jax", "julia", "julia_server"] = "scipy"
    julia_file: str | None = None
    method:     str | None = None
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
    module:         str
    factory:        str
    factory_kwargs: dict[str, Any] = {}
    metrics:        str | None = None


# ---------------------------------------------------------------------------
# Excitation
# ---------------------------------------------------------------------------

class ExcitationSpec(BaseModel):
    """Which entity / port to drive and which state to measure."""
    entity:       str
    port:         str
    output_state: str


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
    name:            str
    type:            Literal["discrete_frequency_sweep"]
    enabled:         bool  = True
    frequencies:     FrequencyGridSpec
    amplitude:       float
    dc_offset:       float = 0.0
    settle_periods:  int   = 10
    measure_periods: int   = 5


class DCOperatingPointSweepSpec(BaseModel):
    name:            str
    type:            Literal["dc_operating_point_sweep"]
    enabled:         bool        = True
    dc_values:       list[float]
    probe_frequency: float
    probe_amplitude: float
    settle_periods:  int  = 10
    measure_periods: int  = 5


class AmplitudeSweepSpec(BaseModel):
    name:            str
    type:            Literal["amplitude_sweep"]
    enabled:         bool        = True
    frequency:       float
    amplitudes:      list[float]
    dc_offset:       float = 0.0
    settle_periods:  int   = 10
    measure_periods: int   = 5


class ParameterSweepSpec(BaseModel):
    name:        str
    type:        Literal["parameter_sweep"]
    enabled:     bool        = True
    sweep_param: str
    values:      list[float]
    sub_test:    str


class ParameterGridSpec(BaseModel):
    name:     str
    type:     Literal["parameter_grid"]
    enabled:  bool                    = True
    params:   dict[str, list[float]]
    sub_test: str
    mode:     Literal["full_factorial", "pairs"] = "full_factorial"


class DOESweepSpec(BaseModel):
    name:    str
    type:    Literal["doe_sweep"]
    enabled: bool = True
    design:  Literal[
        "latin_hypercube", "sobol", "halton",
        "central_composite", "box_behnken", "full_factorial",
    ] = "latin_hypercube"
    n_samples: int = 50
    params:    dict[str, DOEParamSpec]
    sub_test:  str
    outputs:   list[str] = []


class ContinuousChirpSpec(BaseModel):
    name:       str
    type:       Literal["continuous_chirp"]
    enabled:    bool                           = True
    f_start:    float
    f_end:      float
    duration:   float
    amplitude:  float
    dc_offset:  float                          = 0.0
    chirp_type: Literal["log", "linear"]       = "log"


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
# Plot panel specs
# ---------------------------------------------------------------------------

class BodeSeriesSpec(BaseModel):
    """One series on a Bode plot — references a test by name."""
    test:       str
    label:      str | None = None
    show_phase: bool       = True
    db:         bool       = True


class BodePanelSpec(BaseModel):
    """Magnitude + phase Bode diagram; overlays multiple test series."""
    type:    Literal["bode"] = "bode"
    enabled: bool            = True
    title:   str | None      = None
    series:  list[BodeSeriesSpec] = []


class ChirpTimeseriesPanelSpec(BaseModel):
    """Raw time series of the chirp output state."""
    type:    Literal["chirp_timeseries"] = "chirp_timeseries"
    enabled: bool                        = True
    title:   str | None                  = None
    test:    str


class AmplitudeSweepPanelSpec(BaseModel):
    """Transfer function magnitude (and optionally phase) vs drive amplitude."""
    type:       Literal["amplitude_sweep"] = "amplitude_sweep"
    enabled:    bool                       = True
    title:      str | None                 = None
    test:       str
    show_phase: bool = True


class DCSweepPanelSpec(BaseModel):
    """Small-signal gain and phase vs DC operating point."""
    type:    Literal["dc_sweep"] = "dc_sweep"
    enabled: bool                = True
    title:   str | None          = None
    test:    str


class ParameterFamilyPanelSpec(BaseModel):
    """Family of Bode curves from a parameter sweep, coloured by parameter value."""
    type:       Literal["parameter_family"] = "parameter_family"
    enabled:    bool                        = True
    title:      str | None                  = None
    test:       str
    db:         bool = True
    show_phase: bool = True


class DOEScatterPanelSpec(BaseModel):
    """Scatter of one scalar metric vs one parameter from a DOE or parameter sweep."""
    type:        Literal["doe_scatter"] = "doe_scatter"
    enabled:     bool                   = True
    title:       str | None             = None
    test:        str
    x_param:     str
    y_metric:    str        = "f0"    # f0 | Q | damping_ratio | peak_magnitude
    color_param: str | None = None


class ParameterGridHeatmapPanelSpec(BaseModel):
    """Heatmap of a scalar metric over a 2-parameter grid."""
    type:    Literal["parameter_grid_heatmap"] = "parameter_grid_heatmap"
    enabled: bool                              = True
    title:   str | None                        = None
    test:    str
    metric:  str = "peak_magnitude"   # f0 | Q | damping_ratio | peak_magnitude


AnyPanelSpec = Annotated[
    Union[
        BodePanelSpec,
        ChirpTimeseriesPanelSpec,
        AmplitudeSweepPanelSpec,
        DCSweepPanelSpec,
        ParameterFamilyPanelSpec,
        DOEScatterPanelSpec,
        ParameterGridHeatmapPanelSpec,
    ],
    Field(discriminator="type"),
]


class FigureSpec(BaseModel):
    """Top-level figure appearance."""
    title:    str | None                    = None
    subtitle: str | None                    = None
    size:     tuple[float, float] | None    = None   # inches; auto if None


class PlotsSpec(BaseModel):
    """Plot generation configuration — paired with the tests: section."""
    output:  str        = "characterization_summary.png"
    dpi:     int        = 150
    figure:  FigureSpec = Field(default_factory=FigureSpec)
    panels:  list[AnyPanelSpec] = []


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class CharacterizationConfig(BaseModel):
    """Complete characterization campaign: tests + plots in one file."""
    version:    str            = "1.0"
    output:     str            = "results.json"
    backend:    BackendSpec    = Field(default_factory=BackendSpec)
    model:      ModelSpec
    excitation: ExcitationSpec
    tests:      list[TestSpec] = []
    plots:      PlotsSpec      = Field(default_factory=PlotsSpec)

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

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CharacterizationConfig":
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to load YAML plans: pip install pyyaml"
            ) from e
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))

    @classmethod
    def from_json(cls, path: str | Path) -> "CharacterizationConfig":
        import json
        with open(path) as f:
            return cls.model_validate(json.load(f))
