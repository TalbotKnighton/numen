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
    )]
    # No default needed — ExcitationPort is metadata only; not compiled into x or p
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
  n_workers: 1                # julia_server only: N parallel solver processes
  n_save_points: 0            # 0 = save every adaptive step; N = save N uniform points
  dtsave: null                # save every dtsave seconds (mutually exclusive with n_save_points)
  dtmax: null                 # cap adaptive step size (prevents aliasing / missing transients)

model:
  module: world               # importable module (YAML dir added to sys.path)
  factory: make_world         # callable that returns a World object
  factory_kwargs: {}          # kwargs forwarded to the factory

excitation:
  entity: osc                 # entity_id in the world
  component: my_component     # component kind that owns the ExcitationPort
  port: force                 # ExcitationPort field name on that component
  output_state: position      # state field to measure (response signal)
  output_component: null      # component kind for output_state; defaults to component if null

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

## 2a  Parallel execution (`n_workers`)

Set `n_workers: N` to launch N Julia server processes and distribute DOE-level work
across them.  Each worker JIT-compiles all dynamics on its first solve; subsequent
solves within the same worker are warm.  Single-server sequential is the default.

```yaml
backend:
  type: julia_server
  julia_file: dynamics.jl
  method: Tsit5
  n_workers: 4          # 4 parallel processes
  n_save_points: 2000   # cap output size per solve
```

CLI override (always wins over YAML):

```bash
numen characterize test_plan.yaml --workers 4
numen characterize test_plan.yaml --workers 1   # force sequential
```

**What parallelises:** `parameter_sweep`, `parameter_grid`, `doe_sweep` — each design
point is dispatched to a free worker.  Top-level leaf tests (chirp, FRF, etc.) are
not parallelised; they run on one worker sequentially.

**Output density knobs (all backends):**

| Key | Meaning |
|---|---|
| `n_save_points: N` | Save exactly N uniformly-spaced output points |
| `dtsave: 0.001` | Save every 1 ms (mutually exclusive with `n_save_points`) |
| `dtmax: 0.0005` | Cap adaptive step size at 0.5 ms (independent of save density) |

Rule of thumb for `dtmax` / `dtsave`: set to `1 / (10 × f_max)` for 10 samples per
period of the highest-frequency content you care about.

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
# ── Model parameter (3-part path: entity_id.component_kind.field_name) ──
- name: damping_family
  type: parameter_sweep
  sweep_param: osc.my_component.c1   # ParameterField: entity.component_kind.field
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
`continuous_chirp`, `amplitude_sweep`, `dc_operating_point_sweep`,
`two_tone`, `harmonic_distortion_sweep`, `phase_portrait`.

### 3.6  `parameter_grid`

Full factorial or pairwise grid over explicit value lists for multiple params.
Supports the same `excitation.*` paths as `parameter_sweep`.

```yaml
- name: c0_c1_grid
  type: parameter_grid
  params:
    osc.my_component.c0: [0.05, 0.1, 0.2]   # 3-part: entity.component_kind.field
    osc.my_component.c1: [0.5, 1.0, 2.0]
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
    osc.my_component.c0: { min: 0.05, max: 0.5, scale: linear }   # 3-part path
    osc.my_component.c1: { min: 0.1,  max: 5.0, scale: log    }
  sub_test: baseline_frf
```

Install optional extras for classical designs:
```bash
uv pip install "numen[characterization]"   # adds pyDOE3, SALib, pandas
```

### 3.8  `two_tone`

Apply two simultaneous sinusoids at f1 and f2; extract intermodulation products,
THD, IMD3 (dB), and IP3 estimate.  Identifies the order and magnitude of
nonlinearity without sweeping.

```yaml
- name: two_tone_near_resonance
  type: two_tone
  f1: 0.9          # Hz — lower tone (near f₀)
  f2: 1.1          # Hz — upper tone (near f₀)
  amplitude1: 0.05
  amplitude2: 0.05
  n_cycles: 50     # total cycles of f1 to simulate; 70% discarded as transient
  max_order: 3     # extract IM products up to order 3 (standard IM2, IM3)
  dc_offset: 0.0
```

**What it tells you:**
- **IMD3 (dB)**: ratio of strongest 3rd-order sideband to fundamental.  More negative = more linear.
- **IP3**: input amplitude at which the extrapolated fundamental and IM3 lines would intersect.  Higher = more headroom.
- **THD (%)**: all distortion relative to the fundamental.

**Frequency placement:** put f1 and f2 near (but not at) resonance so the response
is amplified.  The 3rd-order in-band sidebands are at 2f1−f2 and 2f2−f1; these
appear between or just outside f1 and f2.

### 3.9  `harmonic_distortion_sweep`

Stepped sine that measures H1, H2, H3, … vs frequency.  THD(f) localises where
the nonlinearity is active; the harmonic orders identify the type.

```yaml
- name: thd_sweep
  type: harmonic_distortion_sweep
  frequencies:
    spacing: log
    f_start: 0.1
    f_end: 10.0
    n_points: 30
  amplitude: 0.1
  dc_offset: 0.0
  max_harmonic: 4     # measure H1 through H4
  settle_periods: 30
  measure_periods: 5
```

### 3.10  `free_decay`

Release from a large initial condition; apply the Hilbert transform to extract:
- **Backbone curve**: instantaneous frequency f(A) as amplitude decays
- **Damping curve**: instantaneous damping ratio ζ(A)

For a linear system, f(A) is flat at f₀ and ζ(A) is constant.  Any slope or
curvature reveals nonlinearity.

```yaml
- name: free_decay_large_ic
  type: free_decay
  initial_displacement: 2.0
  initial_velocity: 0.0
  t_end: 60.0              # s — long enough to ring down to small amplitude
  bandpass_low: 0.3        # Hz — pre-filter to isolate the mode of interest
  bandpass_high: 5.0       # Hz
```

### 3.11  `phase_portrait`

Record the steady-state limit cycle in (position, velocity) state space.  If
`poincare: true`, also records stroboscopic samples at multiples of the forcing
period T:
- **Periodic response**: Poincaré dots cluster at one point
- **Period-doubled**: two clusters
- **Quasi-periodic / chaotic**: ring or area of points

```yaml
- name: portrait_at_resonance
  type: phase_portrait
  frequency: 1.0
  amplitude: 0.5
  n_transient_cycles: 50   # discard to reach steady state
  n_record_cycles: 10      # record limit cycle
  poincare: true
  dc_offset: 0.0
```

### 3.12  `stochastic_excitation`

Apply a broadband random forcing signal synthesised from a PSD specification
(or an external file) and measure the response.  The signal is pre-computed on
the Python side using inverse FFT with random phases, then stored in the
parameter vector and replayed via table interpolation during the solve (the ODE
is deterministic).

Three input formats are supported:

**`psd_profile`** — inline log-log breakpoints:
```yaml
- name: mil_std_810
  type: stochastic_excitation
  duration: 60.0           # simulation length [s]
  dt_sig: 0.001            # signal sample period [s]  (use 1/(10·f_max) as rule of thumb)
  seed: 42                 # reproducible realisation; null → os.urandom (seed stored in result)
  transient_fraction: 0.2  # discard first 20% from PSD/BLA analysis
  n_welch_segments: 8      # Welch periodogram segments
  dc_offset: 0.0
  signal:
    type: psd_profile
    breakpoints:
      # [frequency Hz, psd_level g²/Hz]
      - [20.0,   0.01]
      - [80.0,   0.04]
      - [350.0,  0.04]
      - [2000.0, 0.007]
    units: g_rms            # g²/Hz → signal in m/s² (×9.80665)
    target_grms: null       # optional RMS normalisation [g]; omit for natural RMS
```

**`psd_file`** — external CSV or JSON:
```yaml
  signal:
    type: psd_file
    path: specs/customer_psd.csv   # resolved relative to test_plan.yaml
    units: g_rms
```

File formats: two-column CSV `frequency,psd` (header optional); JSON
`{"breakpoints": [[f, psd], …], "units": "g_rms"}`.

**Seed management:**
- `seed: null` → samples from `os.urandom`; the used seed is always stored in
  the result JSON so the realisation is replayable.
- `numen characterize test_plan.yaml --seed 42` → global CLI override that
  sets the seed for every stochastic test in the plan.

**Outputs:** `StochasticExcitationResult` with:
- Response PSD (Welch estimate)
- Input PSD (estimated from generated signal)
- BLA gain `|Sxy|/Sxx` — Best Linear Approximation
- BLA coherence `|Sxy|²/(Sxx·Syy)` — coherence below 1 indicates nonlinearity
- RMS, crest factor, seed used

**Backend note:** JAX is not supported (variable-length parameter vector).
Use `julia_server` or `scipy`.

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
  x_param: osc.my_component.c0   # 3-part: entity.component_kind.field
  y_metric: f0                   # f0 | Q | damping_ratio | peak_magnitude
  color_param: osc.my_component.c1   # optional third dimension
```

### 4.7  `parameter_grid_heatmap`

2D heatmap over a `parameter_grid` result (requires exactly 2 parameters).

```yaml
- type: parameter_grid_heatmap
  title: "Grid — Peak |H|"
  test: c0_c1_grid
  metric: peak_magnitude      # f0 | Q | damping_ratio | peak_magnitude
```

### 4.8  `two_tone_spectrum`

Full one-sided FFT of a `two_tone` result, with annotated vertical lines for
each identified harmonic and IM product.

```yaml
- type: two_tone_spectrum
  title: "Two-Tone Spectrum"
  test: two_tone_near_resonance
  db_floor: -80.0          # floor for the dB axis
  annotate_products: true  # draw and label component lines
```

### 4.9  `thd_spectrum`

Two stacked subplots: THD(f) on top, H1 and selected harmonics on bottom.

```yaml
- type: thd_spectrum
  title: "Harmonic Distortion vs Frequency"
  test: thd_sweep
  show_harmonics: [2, 3, 4]   # which Hn rows to overlay
```

### 4.10  `backbone_curve`

Two side-by-side subplots: backbone curve (instantaneous frequency vs amplitude)
and damping ratio vs time.

```yaml
- type: backbone_curve
  title: "Backbone Curve"
  decay_test: free_decay_large_ic
```

### 4.11  `phase_portrait_panel`

Overlay multiple `phase_portrait` limit cycles in one state-space plot, coloured
by amplitude.  Poincaré dots are shown as small filled circles.

```yaml
- type: phase_portrait_panel
  title: "Phase Portrait"
  tests: [portrait_small, portrait_large]   # list of phase_portrait test names
  show_poincare: true
```

### 4.12  `stochastic_response`

Random vibration response panel.  Shows:
- Row 1: Input signal preview (first 5 s)
- Row 2: PSD comparison — input (grey) vs response (blue), log-scale
- Row 3 (optional): BLA gain |H_BLA(f)| — the best linear approximation of the FRF
- Row 4 (optional): BLA coherence γ²(f) — values < 1 identify nonlinear contributions

```yaml
- type: stochastic_response
  enabled: true
  title: "Random Vibe — MIL-STD-810"
  test: mil_std_810        # name of a stochastic_excitation test
  show_bla: true
  show_coherence: true
  psd_db: false            # true → dB re 1 m²/s⁴/Hz
```

An info box in the input preview shows: input RMS, response RMS, crest factor, seed used.

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
