# Random Vibration Testing — Design Plan

**Status:** Planned (Phase 1 ready for implementation)

---

## 1. Motivation

Real test environments subject hardware to broadband stochastic excitation described
by a Power Spectral Density (PSD) specification (e.g. MIL-STD-810, DO-160, customer
vibration envelope). Being able to run a simulation under the same spec — and compare
the simulated response PSD to measured data — closes the gap between characterization
campaigns and acceptance testing.

Three distinct signal-input formats are needed:

| # | Type | When to use |
|---|------|-------------|
| 1 | `psd_profile` / `psd_file` | Standard test spec: log-log breakpoints in Hz vs g²/Hz |
| 2 | `multisine` | Deterministic multi-sine (phase-optimised crest factor); defer impl. |
| 3 | `time_series_file` | Replay a measured or pre-synthesised waveform from a file |

All three produce a **pre-computed time series on the Python side** that is stored in
the parameter vector and interpolated by the ODE right-hand side during integration.
No new solver infrastructure is needed.

---

## 2. High-level approach — "Table excitation"

### Why pre-compute in Python

The ODE RHS must be a pure function of `(t, x, p)`. Generating random numbers inside
the RHS is incompatible with deterministic replay and Julia's threading model. Instead:

1. Python generates the signal array `f_sig[0..N-1]` at uniform time steps `dt_sig`.
2. The array is appended to the parameter vector `p` via a new `inject_table_excitation`
   call that mirrors the existing `inject_excitation` / `inject_chirp_excitation` API.
3. The Julia (and Python) ODE function does **linear table interpolation** at each call.

### Parameter layout in `p`

New entries appended by `inject_table_excitation`:

```
_table_{entity_id}_{port_name}.n_samples   → int stored as float (cast back to Int on Julia side)
_table_{entity_id}_{port_name}.t_start     → first sample time [s]
_table_{entity_id}_{port_name}.dt_sig      → uniform time step [s]
_table_{entity_id}_{port_name}.target_idx  → state index of driven field (float, same as existing pattern)
_table_{entity_id}_{port_name}.signal[0]   → f_sig[0]
_table_{entity_id}_{port_name}.signal[1]   → f_sig[1]
…
_table_{entity_id}_{port_name}.signal[N-1] → f_sig[N-1]
```

`param_index_map` stores these as a range:
```python
new_param_map[f"{exc_eid}.signal"] = (start, start + N)   # slice, not scalar
```

### Julia interpolation

```julia
function table_excitation_dynamics!(
    dx :: AbstractVector{T}, x :: AbstractVector{S}, p :: Vector{Float64},
    t  :: Real, spec :: CompiledSpec, sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    exc_eid = only(sys.entity_ids)
    n      = Int(p[param_idx(spec, exc_eid * ".n_samples")])
    t0     = p[param_idx(spec, exc_eid * ".t_start")]
    dt     = p[param_idx(spec, exc_eid * ".dt_sig")]
    tgt_i  = Int(p[param_idx(spec, exc_eid * ".target_idx")]) + 1   # 1-based
    sig_r  = param_slice(spec, exc_eid * ".signal")                  # UnitRange{Int}

    # Clamp t to table bounds
    tc     = clamp(float(t) - t0, 0.0, (n - 1) * dt)
    frac   = tc / dt
    k      = clamp(Int(floor(frac)) + 1, 1, n)
    k1     = clamp(k + 1, 1, n)
    alpha  = frac - floor(frac)

    F = p[sig_r[k]] * (1 - alpha) + p[sig_r[k1]] * alpha
    dx[tgt_i] += F
end
```

The Python closure mirrors this with `np.interp`.

---

## 3. Signal input formats

### 3.1 `psd_profile` — inline YAML breakpoints

```yaml
- name: mil_std_810_cat10
  type: stochastic_excitation
  duration: 60.0
  dt_sig: 0.001          # 1 kHz sample rate — 10× f_max
  seed: 42               # reproducible realisation; null → system clock
  transient_fraction: 0.2   # discard first 20% of response for PSD estimate
  signal:
    type: psd_profile
    breakpoints:
      # [frequency Hz, psd_level g²/Hz]
      - [20.0,   0.01]
      - [80.0,   0.04]
      - [350.0,  0.04]
      - [2000.0, 0.007]
    units: g_rms       # g²/Hz → divide by g² before ifft; result in g
```

**Generation algorithm:**

```
df   = 1 / duration
freqs = 0, df, 2df, …, f_max              (rfft grid)
S(f)  = piecewise log-log interp of breakpoints (S=0 outside range)
amp(f) = sqrt(2 · S(f) · df)             (one-sided → two-sided, Parseval)
phase  = rng.uniform(0, 2π, len(freqs))   (seeded RNG)
X(f)   = amp(f) · exp(j·phase(f))
f_sig  = irfft(X)                          (real-valued time series)
normalise: f_sig *= target_grms / rms(f_sig)   # if target_grms specified
```

The last normalisation step is optional and only triggered when `target_grms` is
given; otherwise the PSD breakpoints fully determine the signal power.

`units` may be `g_rms` (g² in spec, signal in g) or `m_s2` (m²/s⁴, signal in m/s²).

### 3.2 `psd_file` — external file

```yaml
signal:
  type: psd_file
  path: specs/mil_std_810_cat10.csv    # resolved relative to test_plan.yaml
  # CSV format: two columns, no header OR header "frequency,psd"
  # JSON format: {"breakpoints": [[f0, psd0], [f1, psd1], …], "units": "g_rms"}
  units: g_rms
```

The file is loaded and parsed into the same breakpoint list as `psd_profile`, then
processed identically. Supported formats: `.csv`, `.json`, `.npy` (two-column).

### 3.3 `multisine` — deterministic multi-sine (Phase 2, defer implementation)

```yaml
signal:
  type: multisine
  # Option A: inline table
  tones:
    - {frequency: 10.0,  amplitude: 0.05, phase: 0.0}
    - {frequency: 42.7,  amplitude: 0.10, phase: 1.57}
    - {frequency: 100.0, amplitude: 0.05, phase: 3.14}
  # Option B: external file (same schema as inline, CSV or JSON)
  # file: tones.csv

  optimize_phase: false   # if true: Schroeder phase optimisation (future)
```

**Why defer:** Multi-sine is fully deterministic — no seed needed. The signal
generation is trivial (`sum of sinusoids`), but crest-factor optimisation
(Schroeder / random phase) is a separate design question. The YAML schema and
result type are designed to accommodate it; the runner simply returns
`NotImplementedError` until Phase 2.

**Design note:** The discriminated union on `signal.type` keeps the outer test
spec unchanged; adding `multisine` is a purely additive change.

### 3.4 `time_series_file` — arbitrary pre-computed waveform

```yaml
signal:
  type: time_series_file
  path: measured_input.csv
  # CSV: two columns [time_s, force_N], or single column (uniform dt inferred)
  # JSON: {"t": [...], "f": [...]}  or  {"dt": 0.001, "f": [...]}
  # NPY: shape (N,) for uniform dt (requires dt_sig on outer spec) or (2, N) for [t; f]
  resample: true          # interpolate to outer spec's dt_sig grid (default true)
```

When `resample: false`, the file's native `dt` is used and `dt_sig` on the outer
spec is ignored (but must be consistent within ~0.1% or a warning is emitted).

---

## 4. Seed management

### Per-test (YAML)

```yaml
- name: vibe_run_1
  type: stochastic_excitation
  seed: 42       # reproducible
  …

- name: vibe_run_2
  type: stochastic_excitation
  seed: null     # null → os.urandom-seeded; recorded in result for replay
  …
```

`seed: null` samples one 64-bit integer from `os.urandom` and records the used
seed in the result JSON so the run can always be replayed.

### Global CLI override

```bash
numen characterize test_plan.yaml --seed 7
```

`--seed` (integer) overrides the per-test seed for every stochastic test in the
plan, identical in pattern to `--workers`:

```python
# cli.py  (characterize command)
seed: int = typer.Option(-1, "--seed",
                         help="Override random seed for all stochastic tests (-1 = use YAML values)")

# … after config load:
if seed >= 0:
    for test in config.tests:
        if isinstance(test, StochasticExcitationSpec):
            config = replace_test_seed(config, test.name, seed)
```

`replace_test_seed` returns a new `CharacterizationConfig` with the named test's
seed replaced — same pure-function replace pattern used for `n_workers`.

---

## 5. Test specification schema

```python
class PSDProfileSignalSpec(BaseModel):
    type:        Literal["psd_profile"]
    breakpoints: list[tuple[float, float]]   # [(f_hz, psd_level), …]
    units:       Literal["g_rms", "m_s2"] = "g_rms"
    target_grms: float | None = None         # optional RMS normalisation

class PSDFileSignalSpec(BaseModel):
    type:    Literal["psd_file"]
    path:    str
    units:   Literal["g_rms", "m_s2"] = "g_rms"
    target_grms: float | None = None

class MultisineSignalSpec(BaseModel):
    type:            Literal["multisine"]
    tones:           list[dict] | None = None    # [{frequency, amplitude, phase}, …]
    file:            str | None = None
    optimize_phase:  bool = False

class TimeSeriesFileSignalSpec(BaseModel):
    type:    Literal["time_series_file"]
    path:    str
    resample: bool = True

AnySignalSpec = Annotated[
    Union[PSDProfileSignalSpec, PSDFileSignalSpec, MultisineSignalSpec, TimeSeriesFileSignalSpec],
    Field(discriminator="type"),
]

class StochasticExcitationSpec(BaseModel):
    """Broadband random vibration test."""
    name:                str
    type:                Literal["stochastic_excitation"]
    duration:            float                          # simulation length [s]
    dt_sig:              float                          # signal sample period [s]
    signal:              AnySignalSpec
    seed:                int | None = None              # null → os.urandom
    transient_fraction:  float = 0.2                   # fraction discarded from PSD estimate
    n_welch_segments:    int   = 8                      # segments for Welch estimate
    dc_offset:           float = 0.0
```

---

## 6. Result type

```python
@dataclass
class StochasticExcitationResult:
    name:                str
    seed_used:           int                  # always recorded, even if null was given

    # Time domain
    t:                   np.ndarray           # shape (n_response,)
    response:            np.ndarray           # shape (n_response,) — after transient

    # Input signal (first few seconds, for diagnostic)
    t_input:             np.ndarray           # shape (n_preview,), capped at 5s
    input_signal:        np.ndarray           # shape (n_preview,)

    # PSD estimates (Welch)
    input_psd_freq:      np.ndarray           # shape (n_fft,)
    input_psd:           np.ndarray           # shape (n_fft,)  [input signal PSD]
    response_psd_freq:   np.ndarray           # shape (n_fft,)
    response_psd:        np.ndarray           # shape (n_fft,)  [output response PSD]

    # Scalar metrics
    input_rms:           float
    response_rms:        float
    crest_factor:        float                # peak / rms of response
    bla_gain:            np.ndarray           # BLA: |Sxy(f)| / Sxx(f) (approximate FRF)
    bla_coherence:       np.ndarray           # coherence γ²(f) = |Sxy|² / (Sxx·Syy)

    # Metadata
    duration:            float
    dt_sig:              float
    dc_offset:           float
    signal_type:         str                  # "psd_profile", "psd_file", etc.
```

**BLA (Best Linear Approximation)** is computed as:

```python
from scipy.signal import csd, welch
f, Sxx = welch(input_signal, fs=1/dt_sig, nperseg=n_seg)
f, Syy = welch(response,     fs=1/dt_sig, nperseg=n_seg)
f, Sxy = csd(input_signal, response, fs=1/dt_sig, nperseg=n_seg)
bla_gain      = np.abs(Sxy) / np.maximum(Sxx, 1e-300)
bla_coherence = np.abs(Sxy)**2 / np.maximum(Sxx * Syy, 1e-300)
```

For a linear system the BLA equals the FRF exactly. Deviations in coherence
identify nonlinear or noise contributions.

---

## 7. Plot panels

### `stochastic_response` panel

```yaml
- type: stochastic_response
  enabled: true
  title: "Random Vibe — MIL-STD-810 Cat 10"
  test: mil_std_810_cat10
  show_bla: true
  show_coherence: true
  psd_db: false          # true → dB re 1 g²/Hz
```

Layout (3 or 4 rows):

```
┌──────────────────────────────────────────────────┐
│ Row 1: Input signal preview (first 5 s)          │
│        + text box: RMS, crest factor, seed       │
├──────────────────────────────────────────────────┤
│ Row 2: PSD — input spec (grey) vs response       │
│        (blue);  if psd_file, overlay target      │
├──────────────────────────────────────────────────┤
│ Row 3: BLA gain [optional, if show_bla]          │
├──────────────────────────────────────────────────┤
│ Row 4: BLA coherence [optional, if show_coherence│
└──────────────────────────────────────────────────┘
```

---

## 8. Changes required

### Python

| File | Change |
|------|--------|
| `characterization/schema.py` | Add `StochasticExcitationSpec`, signal specs, `StochasticResponsePanelSpec`; extend `TestSpec` and `AnyPanelSpec` unions |
| `characterization/results.py` | Add `StochasticExcitationResult` with `to_dict()` / `_dict_to_result()` round-trip |
| `characterization/excitation.py` | Add `inject_table_excitation(spec, entity_id, port_name, target_field, signal_array, dt_sig, t_start)` |
| `characterization/tests/stochastic_excitation.py` | New file: `run_stochastic_excitation(test, base_spec, …)` |
| `characterization/runner.py` | Dispatch `StochasticExcitationSpec`; pass `--seed` override |
| `characterization/plot_runner.py` | Add `_render_stochastic_response` renderer |
| `cli.py` | Add `--seed` option to `characterize` command; propagate to runner |
| `init_data/CHARACTERIZATION.md` | Add section 3.12 (stochastic_excitation) and 4.12 (stochastic_response panel) |

### Julia (NumenCharacterization module)

| File | Change |
|------|--------|
| Julia characterization module | Add `table_excitation_dynamics!` with linear interpolation from `p` |

The table excitation function is loaded from the user's `dynamics.jl` if they define
it there, or from the built-in `NumenCharacterization` module automatically — the same
lookup logic used for `excitation_dynamics!` and `chirp_dynamics!`.

### Scaffold templates

| File | Change |
|------|--------|
| `init_data/test_plan_generic.yaml` | Add stochastic section (commented out) |
| `init_data/test_plan_mechanical.yaml` | Add MIL-STD-810 profile example (commented out) |

---

## 9. `inject_table_excitation` API sketch

```python
def inject_table_excitation(
    spec:         CompiledSpec,
    entity_id:    str,
    port_name:    str,
    target_field: str,
    signal:       np.ndarray,   # pre-computed, shape (N,), uniform dt_sig
    dt_sig:       float,
    t_start:      float = 0.0,
) -> CompiledSpec:
    """Return a new CompiledSpec with a table-lookup forcing system added.

    The signal is appended to the parameter vector in the layout:
        _table_{entity_id}_{port_name}.n_samples  → N
        _table_{entity_id}_{port_name}.t_start    → t_start
        _table_{entity_id}_{port_name}.dt_sig     → dt_sig
        _table_{entity_id}_{port_name}.target_idx → state index (float)
        _table_{entity_id}_{port_name}.signal     → signal[0..N-1]

    The Python dynamics closure uses np.interp (not jnp — stochastic tests
    run on Julia or scipy backends only; JAX is not supported for this test
    type due to the variable-length parameter layout).
    """
```

**JAX compatibility note:** The table size N varies per test, making it incompatible
with JAX's static-shape requirement at trace time. The runner raises
`NumenFeatureError("variable_table_excitation")` when a JAX backend is requested
for a stochastic test. Scipy and Julia backends support it fully.

---

## 10. PSD generation — worked example

```python
import numpy as np
from numpy.fft import rfft, irfft, rfftfreq

def generate_psd_signal(
    breakpoints: list[tuple[float, float]],   # [(f_hz, psd_g2hz), …]
    duration:    float,
    dt_sig:      float,
    seed:        int | None,
    units:       str = "g_rms",
    target_grms: float | None = None,
) -> tuple[np.ndarray, int]:
    """Return (signal, seed_used)."""
    rng = np.random.default_rng(seed)
    seed_used = int(rng.bit_generator.state["state"]["state"]) if seed is None else seed

    N    = int(round(duration / dt_sig))
    fs   = 1.0 / dt_sig
    freq = rfftfreq(N, d=dt_sig)        # one-sided freq grid

    # Log-log interpolate PSD onto freq grid
    bkf = np.array([b[0] for b in breakpoints])
    bkp = np.array([b[1] for b in breakpoints])
    psd = np.zeros_like(freq)
    mask = (freq >= bkf[0]) & (freq <= bkf[-1])
    psd[mask] = np.exp(np.interp(
        np.log(freq[mask]), np.log(bkf), np.log(bkp)
    ))

    # PSD → amplitude spectrum  (one-sided, bin width = 1/duration)
    df   = freq[1] - freq[0]            # = 1/duration
    amp  = np.sqrt(2.0 * psd * df)      # factor 2: one-sided → two-sided energy
    amp[0]  = 0.0                        # zero DC
    amp[-1] = 0.0                        # zero Nyquist

    # Random phases
    phases = rng.uniform(0.0, 2.0 * np.pi, len(freq))
    X      = amp * np.exp(1j * phases)
    signal = irfft(X, n=N)              # real-valued time series

    # Optional RMS normalisation
    if target_grms is not None:
        actual_rms = np.sqrt(np.mean(signal**2))
        signal    *= target_grms / max(actual_rms, 1e-300)

    # Unit conversion
    G = 9.80665
    if units == "g_rms":
        signal = signal / G             # g²/Hz spec → signal in g → convert to m/s²
        # Actually keep in the unit the ODE expects (m/s²):
        signal = signal * G             # signal was already in g, convert to m/s²
        # (net: no-op if the PSD spec is already in g²/Hz and the ODE expects m/s²)
        # Callers must match units to ODE force units — document this clearly.

    return signal.astype(np.float64), seed_used
```

**Unit responsibility:** The PSD spec and ODE force units must match. Document in
CHARACTERIZATION.md that `units: g_rms` means the PSD spec is in g²/Hz and the
generated signal is in m/s² (after ×9.80665), which is the SI unit expected by
the dynamics function. `units: m_s2` means the spec and signal are both in SI
units throughout.

---

## 10b. Gating (composition by multiplication) — shipped

`StochasticExcitationSpec` accepts an optional `gate:` block.  The gate is
built on the Python side as an array of values in `[0, 1]` and multiplied
element-wise into the pre-computed signal before `inject_table_excitation` is
called — no Julia or scipy runtime change.

Two gate types are supported:

- `intervals` — explicit list of ON windows `[(t_on, t_off), …]`
- `square`    — periodic duty cycle (`period`, `duty`, `phase`)

Both honour `ramp_s` for a half-cosine taper at each ON/OFF edge.  This is the
minimum-viable "compose excitations by multiplication" feature; the random
vibe → gate path proves out the pattern.  General composition (multiply two
arbitrary excitations, AM modulation, envelope filters) is deferred until the
second concrete use case shows up.

Code: `signal_gen.build_gate_signal`; applied in
`tests/stochastic_excitation.run_stochastic_excitation` immediately after
`_build_signal`.

---

## 11. Implementation phases

### Phase 1 (implement now)

- `inject_table_excitation` in `excitation.py`
- `generate_psd_signal` in new `characterization/signal_gen.py`
- `StochasticExcitationSpec` with `psd_profile` and `psd_file` signal types
- `StochasticExcitationResult` with BLA/coherence/crest factor
- `run_stochastic_excitation` runner
- `stochastic_response` plot panel
- `table_excitation_dynamics!` Julia function
- `--seed` CLI option
- Updated `CHARACTERIZATION.md`

### Phase 2 (defer)

- `multisine` signal type (Schroeder phase optimisation optional)
- `time_series_file` signal type (file loading + resample)
- JAX-compatible static-size table excitation (pad to max_samples, clip at runtime)

---

## 12. Open questions

1. **Phase 2 — `time_series_file` priority:** If a user has measured lab data they
   want to replay, `time_series_file` is more urgent than `multisine`. Should these
   swap priority?  Recommendation: implement `time_series_file` first in Phase 2
   (same `inject_table_excitation` path, just different generation step), then `multisine`.

2. **Max table size and memory:** A 60-second signal at 10 kHz = 600,000 floats =
   4.8 MB. This is appended to `p` and serialised to JSON for Julia. At 1 kHz
   (60,000 floats, ~480 KB) it is comfortably small. Document the dt_sig guidance:
   `dt_sig ≤ 1 / (10 · f_max)`, where f_max is the highest breakpoint frequency.

3. **Multiple stochastic inputs:** The table excitation appends a unique key
   `_table_{entity_id}_{port_name}` so two stochastic excitations on different ports
   are fully independent. Same call pattern as two-tone.

4. **Parallel sweeps over seeds:** A `parameter_sweep` over `seed` values (to
   estimate response statistics) works naturally with the existing `n_workers` pool.
   The `sweep_param: seed` path requires a thin adapter in the runner that calls
   `replace(test, seed=int(val))` instead of modifying the compiled spec.

---

## 13. Example YAML (complete)

```yaml
version: "1.0"
output: results_vibe.json

backend:
  type: julia_server
  julia_file: dynamics.jl
  method: Tsit5
  rtol: 1.0e-8
  atol: 1.0e-9
  n_save_points: 8000

model:
  module: world
  factory: make_world
  factory_kwargs:
    x0: 0.0
    v0: 0.0
    omega: 62.832   # 10 Hz natural frequency
    c0: 0.5
    c1: 0.0         # linear system for baseline

excitation:
  entity: osc
  port: force
  output_state: position

tests:
  - name: mil_std_psd
    type: stochastic_excitation
    duration: 60.0
    dt_sig: 0.0001       # 10 kHz — well above 2 kHz PSD ceiling
    seed: 12345
    transient_fraction: 0.25
    n_welch_segments: 8
    dc_offset: 0.0
    signal:
      type: psd_profile
      breakpoints:
        - [20.0,   0.01]
        - [80.0,   0.04]
        - [350.0,  0.04]
        - [2000.0, 0.007]
      units: g_rms

  - name: psd_from_file
    type: stochastic_excitation
    duration: 60.0
    dt_sig: 0.0001
    seed: null          # random; seed recorded in result
    signal:
      type: psd_file
      path: specs/customer_vibe_spec.csv
      units: g_rms

plots:
  output: vibe_characterization.png
  dpi: 150
  figure:
    title: "Random Vibration Response"
    subtitle: "MIL-STD-810 Cat 10 excitation — ω₀=10 Hz"

  panels:
    - type: stochastic_response
      enabled: true
      title: "PSD Response — MIL-STD-810"
      test: mil_std_psd
      show_bla: true
      show_coherence: true
      psd_db: false
```
