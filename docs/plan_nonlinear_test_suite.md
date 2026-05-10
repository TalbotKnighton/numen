# Plan: Extended Nonlinear Characterization Test Suite

**Status:** Planned
**Reference:** [nonlinear_characterization_methods.md](nonlinear_characterization_methods.md)

This plan adds five new test types and three new plot panels to the Numen
characterization framework, targeting nonlinear system identification tasks
beyond the current FRF/chirp/amplitude-sweep baseline.

---

## New test types (priority order)

### 1. `two_tone`

**What it does:**  
Apply two simultaneous sinusoids at f₁ and f₂, run to steady state, FFT the
output, and extract:
- Fundamental components at f₁, f₂
- Harmonic distortion components at 2f₁, 2f₂, 3f₁, 3f₂, …
- Intermodulation products at |m·f₁ ± n·f₂| for m+n ≤ `max_order` (default 3)
- Third-order intercept point (IP3): extrapolate fundamental (slope 1) and 3rd-order
  IM product (slope 3) on log-log; intersection is IP3
- Intermodulation Distortion ratio: IMD₃ = IM₃ / H₁ (dB)

**YAML schema:**
```yaml
- name: two_tone_near_resonance
  type: two_tone
  f1: 10.0          # Hz — lower tone
  f2: 11.0          # Hz — upper tone
  amplitude1: 0.5   # ExcitationPort amplitude for tone 1
  amplitude2: 0.5
  n_cycles: 30      # simulate for 30 periods of f1 to reach steady state
  max_order: 3      # extract IM products up to order 3
  output_field: "osc.position"
  dc_offset: 0.0
```

**Result type:** `TwoToneResult`
```python
@dataclass
class TwoToneResult:
    f1: float
    f2: float
    spectrum_freq: np.ndarray      # full one-sided FFT frequency axis
    spectrum_mag: np.ndarray       # full one-sided FFT magnitude
    fundamentals: dict[str, float] # {"f1": mag, "f2": mag}
    harmonics: dict[str, float]    # {"2f1": mag, "3f1": mag, "2f2": mag, ...}
    im_products: dict[str, float]  # {"2f1-f2": mag, "2f2-f1": mag, ...}
    thd: float                     # Total Harmonic Distortion (%)
    imd3: float                    # 3rd-order IMD ratio (dB)
    ip3_estimate: float | None     # extrapolated IP3 amplitude (None if slope fit fails)
```

**Implementation notes:**
- Build excitation as `A₁·sin(2πf₁t) + A₂·sin(2πf₂t)`.  The ExcitationPort
  currently handles one sinusoid.  Two-tone needs either: (a) a new dual-port
  injection path, or (b) a raw `force_override` that bypasses ExcitationPort
  and supplies `p[excitation_idx]` directly from the test runner.
  Option (b) is simpler and does not require new component machinery.
- Extract tone amplitudes from the FFT using a window-corrected peak-search
  (Hann window + zero-padding to 4× signal length for interpolated peak).
- IP3 fit: run the test at 3–5 amplitude levels (inner loop), fit log-log slope
  to confirm it is ~1 for fundamental and ~3 for IM₃, then extrapolate intersection.
  Alternatively compute analytically from the Volterra kernel approximation.

**Supported backends:** all (scipy, JAX, Julia)

---

### 2. `harmonic_distortion_sweep`

**What it does:**  
At each frequency in a stepped grid, apply a single sinusoid, run to steady state,
and measure the THD and individual harmonic magnitudes (H₂, H₃, H₄, …).  The
result is THD(f) across the sweep band, and H₂(f), H₃(f) individually.

This localises the nonlinearity in frequency: a stiffness nonlinearity produces
large H₃ near resonance; an asymmetric geometry produces large H₂; velocity-
squared drag produces large H₂ wherever velocity is large.

**YAML schema:**
```yaml
- name: thd_sweep
  type: harmonic_distortion_sweep
  freq_min: 1.0
  freq_max: 50.0
  n_freqs: 40
  amplitude: 0.3
  n_cycles: 20        # cycles per frequency
  max_harmonic: 4     # measure H2, H3, H4
  output_field: "osc.position"
  dc_offset: 0.0
```

**Result type:** `HarmonicDistortionResult`
```python
@dataclass
class HarmonicDistortionResult:
    freqs: np.ndarray           # shape (n_freqs,)
    H1: np.ndarray              # fundamental amplitude at each freq
    Hn: np.ndarray              # shape (max_harmonic-1, n_freqs): H2, H3, ...
    thd: np.ndarray             # THD(%) at each freq = sqrt(sum H2..Hn²) / H1
```

**Implementation notes:**
- Structurally identical to `discrete_frequency_sweep` but the post-processing
  extracts multiple harmonics instead of only H₁.
- Reuse the existing `FreqSweepRunner`; add a `harmonic_mode=True` flag that
  returns the full spectrum at each frequency instead of only the fundamental peak.
- Parallelises trivially across Julia server pool (same as FRF sweep).

---

### 3. `free_decay`

**What it does:**  
Set the system to an initial displacement (or velocity) and let it ring down from
that initial condition with zero forcing.  Apply the Hilbert transform to the
output time series to extract:
- Instantaneous amplitude envelope A(t)
- Instantaneous frequency f(t) = (1/2π) dφ/dt
- Instantaneous damping ratio ζ(t) from log-decrement of A(t)
- Backbone curve: plot f vs A (frequency as a function of amplitude)
- Damping curve: plot ζ vs A (damping ratio as a function of amplitude)

**YAML schema:**
```yaml
- name: free_decay_large_amplitude
  type: free_decay
  initial_displacement: 2.0    # initial condition on the output_field's position
  initial_velocity: 0.0
  t_end: 10.0                  # simulate for this long (seconds)
  output_field: "osc.position"
  bandpass:                    # optional pre-HT bandpass filter
    f_low: 5.0
    f_high: 25.0
```

**Result type:** `FreeDecayResult`
```python
@dataclass
class FreeDecayResult:
    t: np.ndarray
    x: np.ndarray               # raw displacement time series
    envelope: np.ndarray        # instantaneous amplitude A(t)
    inst_freq: np.ndarray       # instantaneous frequency f(t) in Hz
    inst_damping: np.ndarray    # instantaneous damping ratio ζ(t)
    backbone_amplitude: np.ndarray   # A values (sorted, decreasing)
    backbone_frequency: np.ndarray   # f(A) — backbone curve
    damping_amplitude: np.ndarray    # A values
    damping_ratio: np.ndarray        # ζ(A)
```

**Implementation notes:**
- Uses `scipy.signal.hilbert` for the analytic signal.  No Julia needed for
  post-processing — the simulation runs on whichever backend is configured.
- Differentiate instantaneous phase with `np.gradient` (not `np.diff`) for
  uniform-spacing accuracy.
- The `initial_displacement` field needs to override the component's initial
  state before compiling; the runner should call `world.components[eid].position
  = initial_displacement` before `compile_spec`.
- Bandpass filter: apply a Butterworth or Hann-windowed FIR before the Hilbert
  transform to isolate one mode.

**Supported backends:** all

---

### 4. `phase_portrait`

**What it does:**  
For a periodic forcing scenario, run to steady state and then record N cycles of
the limit cycle in state space (x, ẋ).  Optionally record a Poincaré section
(stroboscopic samples at multiples of T = 1/f_forcing).

**YAML schema:**
```yaml
- name: limit_cycle_portrait
  type: phase_portrait
  frequency: 12.5
  amplitude: 0.8
  n_transient_cycles: 50      # discard this many cycles at the start
  n_record_cycles: 10         # record these cycles
  output_field: "osc.position"
  velocity_field: "osc.velocity"   # optional — computed by diff if omitted
  dc_offset: 0.0
  poincare: true              # also record stroboscopic section
```

**Result type:** `PhasePortraitResult`
```python
@dataclass
class PhasePortraitResult:
    x: np.ndarray               # displacement (limit cycle only)
    xdot: np.ndarray            # velocity
    t: np.ndarray
    poincare_x: np.ndarray | None      # stroboscopic points
    poincare_xdot: np.ndarray | None
```

**Implementation notes:**
- Straightforward: run a long simulation, discard transient, return the
  (position, velocity) trajectory.
- Poincaré section: collect state at t = t_start + n/f_forcing for integer n.
  Either run with `tstops` at all multiples, or interpolate from the dense
  output.
- A `phase_portrait_amplitude_family` compound test (multiple amplitudes overlaid)
  is a natural extension — see §Plot panels below.

---

### 5. `broadband_noise`

**What it does:**  
Apply a bandlimited white noise excitation and record a long output time series.
Post-process to estimate:
- Best Linear Approximation (BLA) — the linear FRF that minimises the mean-square
  difference between the linear prediction and the actual output (Schoukens et al.)
- Nonlinear distortion spectrum — variance of output not explained by the BLA,
  plotted vs. frequency
- Optionally: bicoherence (normalized bispectrum) to identify quadratic coupling

**YAML schema:**
```yaml
- name: broadband_bla
  type: broadband_noise
  f_low: 1.0
  f_high: 100.0
  amplitude: 0.2
  t_total: 60.0
  n_averages: 10          # split into segments for averaging
  output_field: "osc.position"
  compute_bicoherence: false   # expensive; set true to diagnose quadratic NL
```

**Result type:** `BroadbandResult`
```python
@dataclass
class BroadbandResult:
    freq: np.ndarray
    bla_magnitude: np.ndarray    # best linear approximation |H_BLA(f)|
    bla_phase: np.ndarray
    nl_distortion: np.ndarray    # stochastic NL distortion level (same units)
    snr: np.ndarray              # signal-to-noise ratio (dB)
    bicoherence: np.ndarray | None   # shape (n_freq, n_freq) if computed
```

**Implementation notes:**
- Generate noise in Python with `numpy.random`; feed as a precomputed array
  into the ExcitationPort (requires a lookup-table excitation mode — new feature
  needed on the ExcitationPort path or a separate `table_excitation` port type).
- BLA estimation: segment the record, Welch-average cross-spectrum and
  auto-spectrum: `H_BLA(f) = Syu(f) / Suu(f)`.
- Bicoherence: scipy has no built-in; use `pyspectrum` or implement the FFT
  accumulation directly.  Mark as optional/experimental.

---

## New plot panel types

### `two_tone_spectrum`

Full FFT of the two-tone output with labelled tone, harmonic, and IM product
components.  Log-scale magnitude vs. frequency.

```yaml
- type: two_tone_spectrum
  test: two_tone_near_resonance
  db_floor: -120
  annotate_products: true
```

### `backbone_curve`

Overlays the backbone curve (frequency vs. amplitude from free decay) on the
amplitude-sweep FRF family.  The backbone should intersect each FRF curve at
its peak — this is the visual confirmation that the identification is consistent.

```yaml
- type: backbone_curve
  decay_test: free_decay_large_amplitude
  frf_family_test: frf_vs_amplitude    # a parameter_sweep over amplitude
  frequency_axis: [5, 25]
```

### `phase_portrait_family`

Multiple phase portraits (limit cycles) overlaid in (x, ẋ) space, coloured by
amplitude or frequency parameter.  Poincaré dots plotted if available.

```yaml
- type: phase_portrait_family
  tests: [portrait_amp_01, portrait_amp_03, portrait_amp_06]
  color_by: amplitude
```

### `thd_spectrum`

THD(f) and individual harmonic magnitudes H₂(f), H₃(f) vs. frequency — the
result of a `harmonic_distortion_sweep`.

```yaml
- type: thd_spectrum
  test: thd_sweep
  show_harmonics: [2, 3]
```

---

## Implementation priority and dependencies

```
Phase 1 — no new framework machinery required (all four implemented)
  ├── two_tone                    (inject_excitation called twice with different port
  │                                names; both accumulate via += into the same target
  │                                slot — architecture already supports this natively)
  ├── harmonic_distortion_sweep   (variant of existing freq_sweep runner)
  ├── free_decay                  (new runner; scipy.signal.hilbert only)
  └── phase_portrait              (new runner; trivial post-processing)

Phase 2 — superseded by dedicated plan
  └── broadband_noise             → see docs/plan_random_vibe_testing.md
```

Phase 1 tests are self-contained in the `characterization/` runner module;
no changes needed to the Julia backend, the compiler, or the ExcitationPort
infrastructure.

`two_tone` works because `inject_excitation` appends an independent `CompiledSystem`
to `spec.systems` and accumulates with `+=` into the target derivative slot.  Two
calls with different `port_name` values (e.g., `"force"` and `"force_tone2"`)
produce two separate synthetic entities (`_exc_osc_force` and `_exc_osc_force_tone2`),
both writing to the same velocity slot.  No code changes required.

The `broadband_noise` / random vibe test type has been broken out into its own
design plan (`docs/plan_random_vibe_testing.md`) because it requires a new
`inject_table_excitation` function and covers a much wider set of input formats
(PSD profiles, external files, multisine, time-series replay) than originally
sketched here.  The table excitation approach stores the pre-computed signal
directly in `p` — no new `ExcitationPort` field type required.

---

## Test file additions

Each new test type should have a full entry in the `nonlinear_oscillator/test_plan.yaml`
example (which is the canonical reference for the characterization framework).  The
pneumatic_dashpot example can also use `free_decay` once Phase 1 is implemented.

---

## Open questions

1. **Two-tone injection**: resolved — call `inject_excitation` twice with different
   `port_name` values.  Each call appends an independent `CompiledSystem`; both
   accumulate via `+=`.  No new API needed.  N-tone generalises trivially.

2. **Hilbert transform accuracy**: for lightly-damped systems (Q > 50), the HT
   instantaneous frequency estimate is noisy.  Consider offering an EMD (Empirical
   Mode Decomposition) pre-filter, or the wavelet ridge method (Staszewski 1997)
   as an alternative to HT.

3. **IP3 from a single two-tone run**: the single-run estimate requires that the
   system is weakly nonlinear (small perturbation regime).  For strongly nonlinear
   systems the extrapolation breaks down; the multi-amplitude inner loop gives a
   more reliable estimate.  Document this in the YAML schema as `ip3_mode: single |
   multi_amplitude`.
