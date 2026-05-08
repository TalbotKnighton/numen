# Plan: First-class Characterization Plots

**Status:** In progress
**Goal:** Make plot generation a first-class `numen characterize` feature driven by the same
YAML file as the test campaign. Eliminate standalone plot scripts.

---

## Motivation

The current workflow has two problems:

1. `--output results.json` is a CLI flag — the output path lives outside the YAML, making
   the YAML incomplete as a self-contained experiment description.
2. Plot generation is a one-off script (`characterize_plot.py`) that hardcodes which results
   go in which panels. Changing what's shown requires editing Python.

The new design: **one YAML file**, two top-level sections (`tests:` and `plots:`), one command
with flags to select which phase to run.

---

## CLI

```
numen characterize plan.yaml          # default: -c -p (run + plot)
numen characterize plan.yaml -c       # characterise only → writes results.json
numen characterize plan.yaml -p       # plot only → reads results.json, writes PNG
numen characterize plan.yaml -c -p    # explicit both
numen characterize plan.yaml --verbose
```

`-c` and `-p` are independent boolean flags. When neither is given, both run.
No backward compatibility required.

---

## YAML structure

```yaml
version: "1.0"
output: results.json          # where JSON is written (-c) / read (-p)
                              # relative to YAML dir, or absolute

backend:
  type: scipy                 # scipy | jax | julia | julia_server
  julia_file: null
  method: null
  rtol: 1.0e-8
  atol: 1.0e-9

model:
  module: world
  factory: make_world
  factory_kwargs: {}

excitation:
  entity: osc
  port: force
  output_state: position

tests:
  - name: chirp_survey
    enabled: true             # NEW: skip without deleting
    type: continuous_chirp
    ...

  - name: baseline_frf
    enabled: true
    type: discrete_frequency_sweep
    ...

plots:                        # NEW top-level section
  output: characterization_summary.png   # relative to YAML dir, or absolute
  dpi: 150
  figure:
    title: "My Model — Characterization"
    subtitle: null            # optional second line
    size: null                # [width, height] in inches; auto if null

  panels:                     # ordered list; layout is auto-computed
    - type: bode
      enabled: true
      title: "Bode — Stepped Sine vs Chirp"
      db: true
      series:
        - test: baseline_frf
          label: stepped sine
          show_phase: true
        - test: chirp_survey
          label: chirp (survey)
          show_phase: false   # noisy for high-Q — toggle per-series

    - type: chirp_timeseries
      enabled: true
      title: "Chirp Time Series"
      test: chirp_survey

    - type: amplitude_sweep
      enabled: true
      title: "Amplitude Sweep — Nonlinearity"
      test: amplitude_sweep_resonance
      show_phase: true

    - type: dc_sweep
      enabled: true
      title: "DC Sweep"
      test: dc_bias_sweep

    - type: parameter_family
      enabled: false
      title: "FRF Family"
      test: c1_family         # references a ParameterSweepResult
      db: true
      show_phase: true

    - type: doe_scatter
      enabled: false
      title: "Sensitivity — f₀ vs c₀"
      test: sensitivity_study
      x_param: osc.c0         # dot-path parameter name
      y_metric: f0            # f0 | Q | damping_ratio | peak_magnitude
      color_param: osc.c1     # optional third dimension

    - type: parameter_grid_heatmap
      enabled: false
      title: "Grid — Peak |H|"
      test: c0_c1_grid
      metric: peak_magnitude  # f0 | Q | damping_ratio | peak_magnitude
```

---

## Schema changes (`schema.py`)

### Root config

```python
class CharacterizationConfig(BaseModel):
    version: str = "1.0"
    output: str = "results.json"        # NEW — was CLI --output
    backend: BackendSpec = BackendSpec()
    model: ModelSpec
    excitation: ExcitationSpec
    tests: list[AnyTestSpec] = []
    plots: PlotsSpec = PlotsSpec()      # NEW
```

### Test specs — add `enabled`

Every `*Spec` class (DiscreteFrequencySweepSpec, ContinuousChirpSpec, etc.) gains:

```python
enabled: bool = True
```

The runner skips any test where `enabled=False`.

### Plot panel specs (new)

```python
class BodeSeriesSpec(BaseModel):
    test: str
    label: str | None = None
    show_phase: bool = True
    db: bool = True

class BodePanelSpec(BaseModel):
    type: Literal["bode"] = "bode"
    enabled: bool = True
    title: str | None = None
    series: list[BodeSeriesSpec] = []

class ChirpTimeseriesPanelSpec(BaseModel):
    type: Literal["chirp_timeseries"] = "chirp_timeseries"
    enabled: bool = True
    title: str | None = None
    test: str

class AmplitudeSweepPanelSpec(BaseModel):
    type: Literal["amplitude_sweep"] = "amplitude_sweep"
    enabled: bool = True
    title: str | None = None
    test: str
    show_phase: bool = True

class DCSweepPanelSpec(BaseModel):
    type: Literal["dc_sweep"] = "dc_sweep"
    enabled: bool = True
    title: str | None = None
    test: str

class ParameterFamilyPanelSpec(BaseModel):
    type: Literal["parameter_family"] = "parameter_family"
    enabled: bool = True
    title: str | None = None
    test: str
    db: bool = True
    show_phase: bool = True

class DOEScatterPanelSpec(BaseModel):
    type: Literal["doe_scatter"] = "doe_scatter"
    enabled: bool = True
    title: str | None = None
    test: str
    x_param: str
    y_metric: str = "f0"        # f0 | Q | damping_ratio | peak_magnitude
    color_param: str | None = None

class ParameterGridHeatmapPanelSpec(BaseModel):
    type: Literal["parameter_grid_heatmap"] = "parameter_grid_heatmap"
    enabled: bool = True
    title: str | None = None
    test: str
    metric: str = "peak_magnitude"

AnyPanelSpec = Annotated[
    Union[BodePanelSpec, ChirpTimeseriesPanelSpec, AmplitudeSweepPanelSpec,
          DCSweepPanelSpec, ParameterFamilyPanelSpec, DOEScatterPanelSpec,
          ParameterGridHeatmapPanelSpec],
    Field(discriminator="type"),
]

class FigureSpec(BaseModel):
    title: str | None = None
    subtitle: str | None = None
    size: tuple[float, float] | None = None

class PlotsSpec(BaseModel):
    output: str = "characterization_summary.png"
    dpi: int = 150
    figure: FigureSpec = FigureSpec()
    panels: list[AnyPanelSpec] = []
```

---

## `plot_runner.py` — new module

`src/numen/characterization/plot_runner.py`

```python
class CharacterizationPlotter:
    def __init__(self, config: CharacterizationConfig, results: CampaignResults,
                 yaml_dir: Path): ...

    def run(self) -> Path:
        """Build figure, render all enabled panels, save PNG."""
        panels = [p for p in self.config.plots.panels if p.enabled]
        rows, cols = _auto_layout(len(panels))
        fig = plt.figure(figsize=self.config.plots.figure.size or _default_size(rows, cols))
        outer_gs = GridSpec(rows, cols, figure=fig, hspace=0.45, wspace=0.38)
        _add_title(fig, self.config.plots.figure)
        results_by_name = {r.name: r for r in self.results.results}
        for idx, panel_spec in enumerate(panels):
            cell = outer_gs[idx // cols, idx % cols]
            self._render_panel(fig, cell, panel_spec, results_by_name)
        out = _resolve_path(self.config.plots.output, self.yaml_dir)
        fig.savefig(out, dpi=self.config.plots.dpi, bbox_inches="tight")
        return out
```

### Layout algorithm

```python
def _auto_layout(n: int) -> tuple[int, int]:
    if n == 0: return 1, 1
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols
```

### Panel renderers (one per type)

Each renderer receives `(fig, subplot_spec, panel_spec, results_by_name)` and uses
`GridSpecFromSubplotSpec` internally for multi-axis panels:

| Panel type | Internal layout | Notes |
|---|---|---|
| `bode` | 2×1 (mag / phase) | `sharex`; phase skipped if no series has `show_phase=True` |
| `chirp_timeseries` | 1×1 | plain time series |
| `amplitude_sweep` | 2×1 (|H| / phase) | phase row optional via `show_phase` |
| `dc_sweep` | 1×2 (|H| / phase) | side-by-side |
| `parameter_family` | 2×1 | colourmap from viridis; colorbar spans both rows |
| `doe_scatter` | 1×1 per param | if multiple params, auto-expands columns |
| `parameter_grid_heatmap` | 1×1 | imshow + colorbar; axes labels from param keys |

### Metric extraction helper

Used by `doe_scatter` and `parameter_grid_heatmap` to pull a scalar from any
nested result:

```python
def _extract_metric(result, metric: str) -> float | None:
    """Return a scalar metric from an FRFResult or the first FRF in a nested result."""
    # Handles FRFResult directly
    # Recurses into ParameterSweepResult, ParameterGridResult, DOESweepResult sub_results
    # Returns None if metric not available
```

Supported metrics: `f0`, `Q`, `damping_ratio`, `peak_magnitude`.

---

## Path resolution

```python
def _resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else base_dir / p
```

Applied to both `config.output` (JSON) and `config.plots.output` (PNG).

---

## Updated `cli.py` — `characterize` command

```python
@app.command()
def characterize(
    plan:    Annotated[Path, typer.Argument(...)],
    do_char: Annotated[bool, typer.Option("-c", is_eager=False)] = False,
    do_plot: Annotated[bool, typer.Option("-p", is_eager=False)] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    # When neither flag given, run both
    if not do_char and not do_plot:
        do_char = do_plot = True

    yaml_dir = plan.resolve().parent
    config   = CharacterizationConfig.from_yaml(plan)
    out_json = _resolve_path(config.output, yaml_dir)

    if do_char:
        runner  = CharacterizationRunner.from_config(config)
        results = runner.run()
        results.save(out_json)

    if do_plot:
        if not out_json.exists():
            _fail(f"results not found: {out_json}  (run with -c first)")
            raise typer.Exit(1)
        results = CampaignResults.load(out_json)
        plotter = CharacterizationPlotter(config, results, yaml_dir)
        out_png = plotter.run()
        _ok(f"Plot saved → {out_png}")
```

---

## Runner changes — respect `enabled`

```python
# runner.py _run_test():
for test_spec in self.config.tests:
    if not getattr(test_spec, "enabled", True):
        continue   # skip disabled tests
    ...
```

---

## `CampaignResults` — add `load()`

```python
@classmethod
def load(cls, path: Path) -> "CampaignResults":
    with open(path) as f:
        data = json.load(f)
    # reconstruct typed result objects from dict
    ...
```

Currently `save()` exists but `load()` does not. This is needed for `-p` without `-c`.

---

## Files changed

| File | Change |
|---|---|
| `src/numen/characterization/schema.py` | Add `output`, `enabled`, `PlotsSpec`, all panel specs |
| `src/numen/characterization/plot_runner.py` | **New** — `CharacterizationPlotter` |
| `src/numen/characterization/results.py` | Add `CampaignResults.load()` |
| `src/numen/characterization/runner.py` | Respect `enabled` flag per test |
| `src/numen/cli.py` | Replace `--output` with `-c`/`-p`; call plotter |
| `examples/nonlinear_oscillator/test_plan.yaml` | Add `output:` and `plots:` sections |
| `examples/nonlinear_oscillator/characterize_plot.py` | Delete (superseded) |
| `src/numen/init_data/CHARACTERIZATION.md` | Update CLI reference + YAML schema |

---

## Phases

### Phase 1 — Foundation ✓ in progress
- [ ] Schema: `output`, `enabled`, `PlotsSpec`, all panel specs
- [ ] `CampaignResults.load()` (JSON → typed objects)
- [ ] `CharacterizationPlotter.run()` with auto-layout
- [ ] Panel renderers: `bode`, `chirp_timeseries`, `amplitude_sweep`, `dc_sweep`
- [ ] CLI: `-c`/`-p` flags, path resolution, call plotter
- [ ] Runner: respect `enabled`
- [ ] Update `test_plan.yaml` with `output:` + `plots:` sections
- [ ] Delete `characterize_plot.py`

### Phase 2 — Parameter result panels ✓ complete
- [x] `parameter_family` panel renderer
- [x] `doe_scatter` panel renderer + `_extract_metric()`
- [x] `parameter_grid_heatmap` panel renderer
- [x] Update CHARACTERIZATION.md with new YAML schema

### Phase 3 — Excitation parameter outer-loop ✓ complete

**Goal:** Let `parameter_sweep`, `parameter_grid`, and `doe_sweep` vary excitation
inputs (DC offset, amplitude, frequency) as the outer dimension over any inner test.

**Problem:** The excitation parameters (`amp`, `freq`, `dc`) live in
`spec.param_index_map` under internal keys like `_exc_osc_force.dc` — not user-facing
paths.  A model parameter like `osc.c1` is accessed directly; excitation needs a
translation layer.

**Solution:** `excitation.*` parameter paths in `param_sweep.py`:

```python
_EXC_PARAM_MAP = {"dc_offset": "dc", "amplitude": "amp", "frequency": "freq"}

def _resolve_param_key(key, entity_id=None, port_name=None) -> str:
    if key.startswith("excitation."):
        sub = key[len("excitation."):]
        return f"_exc_{entity_id}_{port_name}.{_EXC_PARAM_MAP[sub]}"
    return key  # plain model param path, unchanged
```

The `entity_id` and `port_name` come from `CharacterizationRunner` (the excitation
config) and are threaded down through every sweep runner.

**YAML usage:**

```yaml
# Outer loop: 6 DC offsets
# Inner test: baseline_frf (discrete frequency sweep)
# Result: ParameterFamilyResult with 6 FRFResults
- name: frf_vs_dc
  type: parameter_sweep
  sweep_param: excitation.dc_offset   # translates to _exc_osc_force.dc
  values: [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
  sub_test: baseline_frf

# Same pattern for chirp
- name: chirp_vs_dc
  type: parameter_sweep
  sweep_param: excitation.dc_offset
  values: [0.0, 0.3, 0.6, 1.0]
  sub_test: chirp_survey
```

**`_render_parameter_family` sub-result dispatch:**

| `subs[0]` type | Layout | Description |
|---|---|---|
| `FRFResult` | 2×1 stacked Bode | mag + phase (phase optional via `show_phase`) |
| `ChirpResult` | 1×1 | magnitude-only semilog; chirp phase unreliable |
| `AmplitudeSweepResult` | 1×1 | |H| vs drive amplitude |

All three are coloured by sweep parameter using viridis with a colorbar.

**Files changed:**
- `src/numen/characterization/tests/param_sweep.py` — `_EXC_PARAM_MAP`, `_resolve_param_key()`, updated `_set_model_param()` and `run_parameter_sweep()` signatures
- `src/numen/characterization/tests/param_grid.py` — propagate `entity_id`/`port_name`
- `src/numen/characterization/tests/doe_sweep.py` — same
- `src/numen/characterization/runner.py` — `_make_sub_runner()` + `_get_sub_spec()` consolidation
- `src/numen/characterization/plot_runner.py` — `_render_parameter_family` extended for all 3 sub-result types
- `examples/nonlinear_oscillator/test_plan.yaml` — `frf_vs_dc` + `chirp_vs_dc` tests + panels

---

## Open questions

1. **`doe_scatter` with multiple params** — should each param get its own subplot within
   the panel cell (auto-expanding columns), or should the user pick one `x_param` and
   one optional `color_param`? Current design: one axis per panel, user picks axes.

2. **`parameter_grid_heatmap` with >2 params** — for N>2 parameters, a single heatmap
   is impossible. Options: show all 2D projections, or raise an error and require the
   user to add multiple panels. Current design: require exactly 2 params; raise if not.

3. **Figure size with many panels** — `_default_size` scales with rows/cols but tall
   panels (Bode = 2 stacked axes) may feel cramped at small sizes. Consider a minimum
   cell height.
