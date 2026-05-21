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
    type:           Literal["scipy", "jax", "julia", "julia_server"] = "scipy"
    julia_file:     str | None = None
    method:         str | None = None
    rtol:           float = 1e-8
    atol:           float = 1e-9
    n_workers:      int          = 1
    n_save_points:  int          = 0
    dtsave:         float | None = None
    dtmax:          float | None = None
    maxiters:       int   | None = None
    precompile:     bool         = True
    # n_workers:     parallel Julia server processes (julia_server only; 1 = sequential).
    # n_save_points: save exactly N uniformly-spaced output points (0 = every adaptive step).
    # dtsave:        save every dtsave time units.  Mutually exclusive with n_save_points.
    # dtmax:         cap the adaptive step size (prevents missing transients / aliasing).
    # maxiters:      raise OrdinaryDiffEq's iteration cap (default 1e5).  Needed for long
    #                chirps or fine dtmax that require millions of steps.  julia/julia_server only.
    # precompile:    if False, skip startup precompile() pass on dynamics functions.
    #                Saves a few seconds at server startup; first solve pays full JIT cost.
    #                Useful for short campaigns or development cycles.  julia_server only.

    @model_validator(mode="after")
    def _validate(self) -> "BackendSpec":
        if self.type in ("julia", "julia_server") and self.julia_file is None:
            raise ValueError(f"backend.type={self.type!r} requires julia_file to be set")
        if self.n_save_points > 0 and self.dtsave is not None:
            raise ValueError("Specify either n_save_points or dtsave, not both.")
        if self.n_workers > 1 and self.type != "julia_server":
            raise ValueError(f"n_workers > 1 requires backend.type='julia_server' (got {self.type!r})")
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
    """Which entity / port to drive and which state to measure.

    Full path for the excitation port:  entity / component (kind) / port (field name).
    Full path for the output state:     entity / output_component (kind) / output_state (field name).
    ``output_component`` defaults to ``component`` when omitted (most models drive and
    measure on the same component).

    The framework is unit-agnostic: the value injected at the ExcitationPort is
    added directly to the target state derivative.  Use ``scale_by`` to apply a
    division (e.g. divide a force by mass to get an acceleration when the target
    is a velocity-like state).  ``scale_by`` must be the full path
    ``"entity.component.field"`` of an existing ParameterField.
    """
    entity:           str
    component:        str          # component kind that owns the ExcitationPort field
    port:             str
    output_state:     str
    output_component: str | None = None  # component kind for output_state; defaults to component
    scale_by:         str  | None = None  # full path of a ParameterField divisor; None = no scaling


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


class TwoToneSpec(BaseModel):
    """Apply two simultaneous sinusoids; extract harmonics and IM products.

    The two tones are injected as independent excitation systems that
    accumulate into the same ExcitationPort target — no extra infrastructure
    is needed.  The simulation runs for ``n_cycles`` cycles of f1, with the
    first 70% discarded as transient.
    """
    name:       str
    type:       Literal["two_tone"]
    enabled:    bool  = True
    f1:         float
    f2:         float
    amplitude1: float
    amplitude2: float
    n_cycles:   int   = 30    # total cycles of f1 to simulate
    max_order:  int   = 3     # highest IM order to extract (1–3)
    dc_offset:  float = 0.0


class HarmonicDistortionSweepSpec(BaseModel):
    """Stepped-sine sweep that measures THD and individual harmonics at each frequency."""
    name:            str
    type:            Literal["harmonic_distortion_sweep"]
    enabled:         bool  = True
    frequencies:     FrequencyGridSpec
    amplitude:       float
    dc_offset:       float = 0.0
    max_harmonic:    int   = 4     # measure H1 through H_max_harmonic
    settle_periods:  int   = 10
    measure_periods: int   = 5


class FreeDecaySpec(BaseModel):
    """Ring-down from a large initial condition; Hilbert-transform backbone extraction.

    The excitation force is set to zero.  The initial state is overridden to
    ``initial_displacement`` (position) and ``initial_velocity`` (velocity).
    """
    name:                  str
    type:                  Literal["free_decay"]
    enabled:               bool        = True
    initial_displacement:  float
    initial_velocity:      float       = 0.0
    t_end:                 float
    bandpass_low:          float | None = None   # Hz; None = no filter
    bandpass_high:         float | None = None   # Hz; None = no filter


class PhasePortraitSpec(BaseModel):
    """Record the steady-state limit cycle in (position, velocity) state space.

    Simulates ``n_transient_cycles + n_record_cycles`` forcing periods.  The
    first ``n_transient_cycles`` are discarded; the rest are returned as the
    limit cycle.  If ``poincare=True``, stroboscopic samples at multiples of
    the forcing period are also returned (one dot per period for a periodic
    response; a cloud for quasi-periodic or chaotic response).
    """
    name:               str
    type:               Literal["phase_portrait"]
    enabled:            bool  = True
    frequency:          float
    amplitude:          float
    dc_offset:          float = 0.0
    n_transient_cycles: int   = 50
    n_record_cycles:    int   = 10
    poincare:           bool  = True


# ---------------------------------------------------------------------------
# Stochastic excitation (random vibration) — signal sub-specs
# ---------------------------------------------------------------------------

class PSDProfileSignalSpec(BaseModel):
    """Inline PSD breakpoints with log-log interpolation."""
    type:        Literal["psd_profile"]
    breakpoints: list[tuple[float, float]]            # [(f_hz, psd_level), …]
    units:       Literal["g_rms", "m_s2"] = "g_rms"  # psd_level units
    target_grms: float | None = None                  # optional RMS normalisation [g]


class PSDFileSignalSpec(BaseModel):
    """External PSD file (CSV / JSON / NPY). Resolved relative to test_plan.yaml."""
    type:        Literal["psd_file"]
    path:        str
    units:       Literal["g_rms", "m_s2"] = "g_rms"
    target_grms: float | None = None


class MultisineSignalSpec(BaseModel):
    """Deterministic multi-sine — PHASE 2, implementation deferred."""
    type:           Literal["multisine"]
    tones:          list[dict] | None = None   # [{frequency, amplitude, phase}, …]
    file:           str | None = None
    optimize_phase: bool = False


class TimeSeriesFileSignalSpec(BaseModel):
    """Replay a pre-computed waveform from a file (CSV / JSON / NPY)."""
    type:     Literal["time_series_file"]
    path:     str
    resample: bool = True   # interpolate to outer spec's dt_sig grid


AnySignalSpec = Annotated[
    Union[
        PSDProfileSignalSpec,
        PSDFileSignalSpec,
        MultisineSignalSpec,
        TimeSeriesFileSignalSpec,
    ],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Stochastic excitation — gating (element-wise modulator on the signal)
# ---------------------------------------------------------------------------

class IntervalsGateSpec(BaseModel):
    """Gate the stochastic signal ON inside the listed time windows, OFF elsewhere.

    Lets you compose burst tests such as ``"random vibe ON 0–5 s, OFF 5–10 s,
    ON 10–15 s"`` by element-wise multiplication on the pre-computed signal.

    ``ramp_s`` applies a half-cosine taper of that duration to each ON-window
    edge; suppresses the spectral splatter of a hard square edge.  Set to 0
    for a hard gate.
    """
    type:         Literal["intervals"]
    on_intervals: list[tuple[float, float]]   # [(t_on, t_off), …] in seconds
    ramp_s:       float = 0.0                  # half-cosine ramp on each edge [s]


class SquareGateSpec(BaseModel):
    """Periodic square-wave gate: ON for ``duty`` of every ``period`` seconds.

    ``phase`` shifts the start of the ON segment as a fraction of the period
    (0 → ON starts at t=0; 0.5 → ON delayed by half a period).
    """
    type:    Literal["square"]
    period:  float                              # full cycle length [s]
    duty:    float = 0.5                        # ON fraction in [0, 1]
    phase:   float = 0.0                        # phase offset in [0, 1)
    ramp_s:  float = 0.0                        # half-cosine ramp on each edge [s]


AnyGateSpec = Annotated[
    Union[IntervalsGateSpec, SquareGateSpec],
    Field(discriminator="type"),
]


class StochasticExcitationSpec(BaseModel):
    """Broadband random vibration test.

    Generates a pre-computed forcing time series from a PSD specification
    (or an external file) and replays it via table interpolation during the
    ODE solve.  The response PSD, RMS, and crest factor are extracted.  A
    Best Linear Approximation (BLA) is computed from the input/output cross-
    spectrum — coherence below 1 identifies nonlinear or noise contributions.

    Seed management:
        ``seed: null`` draws from os.urandom and the used seed is stored in
        the result so the realisation is always replayable.  The CLI
        ``--seed`` flag overrides all per-test seeds in the plan.

    Note:
        JAX backends are not supported (variable-length parameter vector).
        Use ``julia_server`` or ``scipy`` backends.
    """
    name:                str
    type:                Literal["stochastic_excitation"]
    enabled:             bool        = True
    duration:            float                    # total simulation length [s]
    dt_sig:              float                    # signal sample period [s]
    signal:              AnySignalSpec
    seed:                int | None  = None       # None → os.urandom; recorded in result
    transient_fraction:  float       = 0.2        # fraction discarded from PSD/BLA estimate
    n_welch_segments:    int         = 8          # Welch periodogram segments
    dc_offset:           float       = 0.0
    gate:                AnyGateSpec | None = None  # optional on/off modulator multiplied into the signal


TestSpec = Annotated[
    Union[
        DiscreteFrequencySweepSpec,
        DCOperatingPointSweepSpec,
        AmplitudeSweepSpec,
        ParameterSweepSpec,
        ParameterGridSpec,
        DOESweepSpec,
        ContinuousChirpSpec,
        TwoToneSpec,
        HarmonicDistortionSweepSpec,
        FreeDecaySpec,
        PhasePortraitSpec,
        StochasticExcitationSpec,
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


class TwoToneSpectrumPanelSpec(BaseModel):
    """Full FFT of a two-tone result with annotated harmonic and IM product lines."""
    type:               Literal["two_tone_spectrum"] = "two_tone_spectrum"
    enabled:            bool        = True
    title:              str | None  = None
    test:               str
    db_floor:           float       = -120.0
    annotate_products:  bool        = True


class THDSpectrumPanelSpec(BaseModel):
    """THD(f) and individual harmonic magnitudes from a harmonic_distortion_sweep."""
    type:            Literal["thd_spectrum"] = "thd_spectrum"
    enabled:         bool       = True
    title:           str | None = None
    test:            str
    show_harmonics:  list[int]  = [2, 3]


class BackboneCurvePanelSpec(BaseModel):
    """Backbone curve (instantaneous frequency vs amplitude) from a free_decay test."""
    type:       Literal["backbone_curve"] = "backbone_curve"
    enabled:    bool       = True
    title:      str | None = None
    decay_test: str


class PhasePortraitPanelSpec(BaseModel):
    """Phase portrait (position vs velocity limit cycle) with optional Poincaré dots."""
    type:            Literal["phase_portrait_panel"] = "phase_portrait_panel"
    enabled:         bool       = True
    title:           str | None = None
    tests:           list[str]  = []    # overlay multiple portrait results
    show_poincare:   bool       = True


class StochasticResponsePanelSpec(BaseModel):
    """Random vibration response: input PSD vs response PSD + BLA + coherence."""
    type:            Literal["stochastic_response"] = "stochastic_response"
    enabled:         bool       = True
    title:           str | None = None
    test:            str        = ""    # name of a StochasticExcitationSpec test
    show_bla:        bool       = True
    show_coherence:  bool       = True
    psd_db:          bool       = False  # True → dB re 1 g²/Hz (or m²/s⁴)


AnyPanelSpec = Annotated[
    Union[
        BodePanelSpec,
        ChirpTimeseriesPanelSpec,
        AmplitudeSweepPanelSpec,
        DCSweepPanelSpec,
        ParameterFamilyPanelSpec,
        DOEScatterPanelSpec,
        ParameterGridHeatmapPanelSpec,
        TwoToneSpectrumPanelSpec,
        THDSpectrumPanelSpec,
        BackboneCurvePanelSpec,
        PhasePortraitPanelSpec,
        StochasticResponsePanelSpec,
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
