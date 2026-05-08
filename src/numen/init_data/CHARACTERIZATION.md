# Numen Characterization Framework

A domain-agnostic test campaign engine. Write a YAML test plan, run it against
any model, get structured results and plots — no Python scripting required.

```bash
uv run numen characterize test_plan.yaml        # run tests + generate plot
uv run numen characterize test_plan.yaml -c     # characterize only → results.json
uv run numen characterize test_plan.yaml -p     # plot only → reads results.json
uv run numen characterize test_plan.yaml -c -p  # explicit both
```

---

## 1  Add an ExcitationPort to your model

An `ExcitationPort` marks a component field as an input port where the framework
injects time-varying forcing without touching the dynamics function.

```python
# components.py
from numen.fields import IntegratedField, ParameterField, ExcitationPort
from typing import Annotated, Literal
from numen.spec.component import Component

class MassComponent(Component):
    kind:     Literal["mass"] = "mass"
    position: Annotated[float, IntegratedField()] = 0.0
    velocity: Annotated[float, IntegratedField()] = 0.0
    mass:     Annotated[float, ParameterField()]  = 1.0

    # ExcitationPort — effort-source (force) input.
    # targets="velocity" means F(t) is added to d(velocity)/dt.
    force: Annotated[float, ExcitationPort(
        targets   = "velocity",   # IntegratedField whose derivative gets F(t)
        port_type = "effort",     # "effort" (force/pressure/voltage) or "flow"
        units     = "N",
    )] = 0.0
```

The framework injects `F(t) = amp·sin(2π·freq·t) + dc` automatically.
The dynamics function does not reference `force` at all.

---

## 2  Test plan YAML — full schema

```yaml
version: "1.0"
output: results.json          # where JSON is written (-c) / read (-p)
                              # relative to YAML dir, or absolute path

backend:
  type: scipy                 # scipy | jax | julia | julia_server
  julia_file: null            # required for julia / julia_server
  method: null                # RK45 | Dopri5 | Rodas5P | Tsit5 | …
  rtol: 1.0e-8
  atol: 1.0e-9

model:
  module: world               # importable module (YAML dir added to sys.path)
  factory: make_world         # callable that returns a World object
  factory_kwargs: {}          # kwargs forwarded to the factory

excitation:
  entity: osc                 # entity_id in the world
  port: force                 # ExcitationPort field name on that component
  output_state: position      # state field to measure (response signal)

tests:
  - name: my_test
    type: <test_type>
    enabled: true             # set false to skip without deleting
    ...

plots:
  output: characterization_summary.png   # relative to YAML dir, or absolute
  dpi: 150
  figure:
    title: "My Model — Characterization"
    subtitle: null            # optional second line under title
    size: null                # [width_in, height_in]; auto if null
  panels:
    - type: <panel_type>
      enabled: true
      ...
```

---

## 3  Test types

### 3.1  `discrete_frequency_sweep`

Stepped sine: one solve per frequency, lock-in detection.  Most accurate FRF
method.  Use for final Bode plots and Q measurement.

```yaml
- name: baseline_frf
  type: discrete_frequency_sweep
  enabled: true
  frequencies:
    spacing: log              # log | linear
    f_start: 0.1
    f_end: 10.0
    n_points: 30
  amplitude: 0.01             # drive amplitude (keep small for linear regime)
  dc_offset: 0.0
  settle_periods: 50          # cycles before measurement — see §5 on choosing
  measure_periods: 10
```

### 3.2  `continuous_chirp`

Single solve with a frequency-swept input.  Fast survey; lower SNR than stepped
sine.  Use first to locate resonance, then follow up with a stepped sine.

```yaml
- name: chirp_survey
  type: continuous_chirp
  enabled: true
  f_start: 0.1
  f_end: 10.0
  duration: 120.0             # s — longer → finer frequency resolution
  amplitude: 0.05
  dc_offset: 0.0
  chirp_type: log             # log | linear
```

### 3.3  `amplitude_sweep`

Fixed frequency, varying drive amplitude.  Reveals amplitude-dependent
nonlinearities — a linear system shows flat |H|.

```yaml
- name: amplitude_sweep_resonance
  type: amplitude_sweep
  enabled: true
  frequency: 1.0              # Hz — set to f₀
  amplitudes: [0.001, 0.01, 0.1, 0.5, 1.0]
  dc_offset: 0.0
  settle_periods: 30
  measure_periods: 5
```

### 3.4  `dc_operating_point_sweep`

Sweep DC bias; measure small-signal FRF at each operating point.  Shows how
effective stiffness and damping change with bias.

```yaml
- name: dc_bias_sweep
  type: dc_operating_point_sweep
  enabled: true
  dc_values: [0.0, 0.5, 1.0, 2.0]
  probe_frequency: 1.0        # Hz — probe near resonance
  probe_amplitude: 0.005
  settle_periods: 50
  measure_periods: 10
```

### 3.5  `parameter_sweep`

Repeat a named sub-test for each value of one parameter.  The `sweep_param`
can be a **model parameter** or an **excitation input**:

```yaml
# ── Model parameter (dot-path: entity_id.field_name) ──
- name: damping_family
  type: parameter_sweep
  sweep_param: osc.c1         # ParameterField on the component
  values: [0.5, 1.0, 2.0, 5.0]
  sub_test: baseline_frf      # must match another test name in this plan

# ── Excitation input (outer DC loop over inner FRF) ──
- name: frf_vs_dc
  type: parameter_sweep
  sweep_param: excitation.dc_offset   # dc_offset | amplitude | frequency
  values: [0.0, 0.2, 0.5, 1.0]
  sub_test: baseline_frf

# ── Excitation input (outer DC loop over inner chirp) ──
- name: chirp_vs_dc
  type: parameter_sweep
  sweep_param: excitation.dc_offset
  values: [0.0, 0.5, 1.0]
  sub_test: chirp_survey
```

**Supported `excitation.*` paths:**

| Path | What it varies |
|---|---|
| `excitation.dc_offset` | DC bias on the forcing signal |
| `excitation.amplitude` | Probe amplitude |
| `excitation.frequency` | Fixed probe frequency (for amplitude_sweep sub-tests) |

**Sub-test types supported inside a sweep:** `discrete_frequency_sweep`,
`continuous_chirp`, `amplitude_sweep`, `dc_operating_point_sweep`.

### 3.6  `parameter_grid`

Full factorial or pairwise grid over explicit value lists for multiple params.
Supports the same `excitation.*` paths as `parameter_sweep`.

```yaml
- name: c0_c1_grid
  type: parameter_grid
  params:
    osc.c0: [0.05, 0.1, 0.2]
    osc.c1: [0.5, 1.0, 2.0]
  sub_test: baseline_frf
  mode: full_factorial        # full_factorial | pairs
```

### 3.7  `doe_sweep`

Space-filling or classical DOE over continuous parameter ranges.
Supports the same `excitation.*` paths.

```yaml
- name: sensitivity_study
  type: doe_sweep
  design: latin_hypercube     # latin_hypercube | sobol | halton
                              # central_composite | box_behnken (needs pyDOE3)
                              # full_factorial
  n_samples: 50
  params:
    osc.c0: { min: 0.05, max: 0.5, scale: linear }
    osc.c1: { min: 0.1,  max: 5.0, scale: log    }
  sub_test: baseline_frf
```

Install optional extras for classical designs:
```bash
uv pip install "numen[characterization]"   # adds pyDOE3, SALib, pandas
```

---

## 4  Plot panel types

The `plots:` section controls what `numen characterize -p` renders.  Panels are
laid out automatically in a roughly-square grid.

### 4.1  `bode`

Overlay any mix of FRF and chirp results on one Bode diagram.  Phase is shown
per-series via `show_phase`.

```yaml
- type: bode
  enabled: true
  title: "Bode — Stepped Sine vs Chirp"
  series:
    - test: baseline_frf
      label: stepped sine
      show_phase: true
      db: true
    - test: chirp_survey
      label: chirp (survey)
      show_phase: false       # chirp phase unreliable for high-Q — toggle off
      db: true
```

### 4.2  `chirp_timeseries`

Raw output time series from a chirp solve.

```yaml
- type: chirp_timeseries
  title: "Chirp Time Series"
  test: chirp_survey
```

### 4.3  `amplitude_sweep`

|H| (and optional phase) vs drive amplitude.

```yaml
- type: amplitude_sweep
  title: "Amplitude Sweep — Nonlinearity"
  test: amplitude_sweep_resonance
  show_phase: false
```

### 4.4  `dc_sweep`

|H| and phase vs DC offset (stacked, shared x-axis).

```yaml
- type: dc_sweep
  title: "DC Sweep"
  test: dc_bias_sweep
```

### 4.5  `parameter_family`

A family of curves from a `parameter_sweep` result, coloured by sweep parameter
(viridis colormap).  Handles FRF, chirp, and amplitude-sweep sub-results.

```yaml
- type: parameter_family
  title: "FRF vs DC Offset"
  test: frf_vs_dc             # ParameterSweepResult containing FRFResults
  show_phase: false           # phase row optional for FRF families
  db: true

- type: parameter_family
  title: "Chirp Survey vs DC Offset"
  test: chirp_vs_dc           # ParameterSweepResult containing ChirpResults
  db: true
```

### 4.6  `doe_scatter`

Scatter plot of a scalar metric vs one DOE parameter, with optional colour axis.

```yaml
- type: doe_scatter
  title: "Sensitivity — f₀ vs c₀"
  test: sensitivity_study
  x_param: osc.c0
  y_metric: f0                # f0 | Q | damping_ratio | peak_magnitude
  color_param: osc.c1         # optional third dimension
```

### 4.7  `parameter_grid_heatmap`

2D heatmap over a `parameter_grid` result (requires exactly 2 parameters).

```yaml
- type: parameter_grid_heatmap
  title: "Grid — Peak |H|"
  test: c0_c1_grid
  metric: peak_magnitude      # f0 | Q | damping_ratio | peak_magnitude
```

---

## 5  Choosing `settle_periods`

The most common mistake is under-settling near resonance.

| Q | Rule of thumb |
|---|---|
| Q < 5 | `settle_periods = 10–20` is fine |
| Q = 5–20 | `settle_periods ≈ 2–3 × Q` |
| Q > 20 | `settle_periods ≈ 5 × Q` |

**Example:** Q = 63 at f₀ = 1 Hz → need ~315 settle periods for a clean measurement.

**DC sweep caveat:** a DC step from x=0 → x_eq takes the same ~5τ to decay.
Insufficient settle causes beat-frequency leakage from residual ringing at f₀
into the probe measurement.  Symptom: |H| varies with DC even when the system
is actually linear.

**Practical workflow:**
1. Run a chirp with short `settle_periods` (10–20) to locate resonance and estimate Q.
2. Use `settle_periods ≈ 5 × Q` for the final accurate stepped-sine sweep.

---

## 6  Stepped sine vs chirp — when to use which

| Criterion | Stepped sine | Chirp |
|---|---|---|
| Accuracy | ✓✓ Best | ✓ Rough estimate |
| Speed | Slower (N solves) | Fast (1 solve) |
| High-Q systems | ✓ Works (with long settle) | ✗ Poor SNR at resonance |
| First survey | Overkill | ✓ Ideal |
| Q measurement | ✓ | Approximate only |

---

## 7  Full example — nonlinear oscillator

See `examples/nonlinear_oscillator/` for a complete worked example including
`excitation.*` parameter sweeps and all panel types:

```
examples/nonlinear_oscillator/
├── components.py     NLOscillatorComponent with ExcitationPort
├── dynamics.py       JAX-compatible dynamics
├── world.py          make_world() factory
├── run.py            free-decay simulation
└── test_plan.yaml    6-test campaign: chirp, FRF, amplitude sweep,
                      DC sweep, FRF-vs-DC family, chirp-vs-DC family
```

```bash
cd examples/nonlinear_oscillator
uv run numen characterize test_plan.yaml        # run all + plot
uv run numen characterize test_plan.yaml -c     # tests only
uv run numen characterize test_plan.yaml -p     # re-plot saved results
```

---

## 8  CLI reference

```
numen characterize PLAN [-c] [-p] [--verbose]

  PLAN      Path to YAML test plan.
  -c        Run characterization tests → writes output (default if neither flag given).
  -p        Generate plots → reads saved results, writes PNG (default if neither flag given).
  --verbose Enable DEBUG logging (per-solve timing, lock-in values, Julia output).

When neither -c nor -p is given, both run (characterize then plot).
```

---

## 9  Loading results in Python

```python
from numen.characterization.results import CampaignResults

results = CampaignResults.load("results.json")

for r in results.all_results():
    print(r.name, type(r).__name__)

# Access a specific result
frf = results.get("baseline_frf")   # returns typed FRFResult, ChirpResult, etc.
print(frf.f0, frf.Q)
```
