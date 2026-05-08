# Numen Characterization Framework

A domain-agnostic test campaign engine built on top of Numen.
Write a YAML test plan, run it against any model, get structured results.

```bash
uv run numen characterize test_plan.yaml --output results.json
uv run numen characterize test_plan.yaml --verbose     # show per-solve timing
```

---

## 1  Add an ExcitationPort to your model

An `ExcitationPort` marks a component field as an **input port** — a place where
the framework can inject time-varying forcing without modifying the model's dynamics
function.  The port is compiled as a `ParameterField`; `inject_excitation()` appends
three parameter slots (`amp`, `freq`, `dc`) and adds a synthetic forcing system at
run time.

```python
# components.py
from numen.fields import IntegratedField, ParameterField, ExcitationPort
from typing import Annotated, Literal
from numen.spec.component import Component

class MassComponent(Component):
    kind:     Literal["mass"] = "mass"
    position: Annotated[float, IntegratedField()] = 0.0   # state
    velocity: Annotated[float, IntegratedField()] = 0.0   # state
    mass:     Annotated[float, ParameterField()]  = 1.0   # param

    # ExcitationPort — effort-source (force) input.
    # targets="velocity" means F(t) is added to d(velocity)/dt.
    force: Annotated[float, ExcitationPort(
        targets   = "velocity",   # IntegratedField whose derivative gets F(t)
        port_type = "effort",     # "effort" (force/pressure/voltage) or "flow"
        units     = "N",          # informational only
    )] = 0.0
```

**Rules:**
- `targets` must name an `IntegratedField` on the same component.
- The dynamics function **does not need to reference `force`** — the framework
  injects `F(t) = amp·sin(2π·freq·t) + dc` via a synthetic system appended to
  the compiled spec at test time.
- Multiple `ExcitationPort` fields on the same component are allowed.

---

## 2  Test plan YAML schema

```yaml
version: "1.0"

backend:
  type: scipy              # scipy | jax | julia | julia_server
  julia_file: null         # required for julia / julia_server backends
  method: null             # solver name: Dopri5, Rodas5P, RK45, …
  rtol: 1.0e-8
  atol: 1.0e-9

model:
  module: world            # importable module OR filename relative to YAML dir
  factory: make_world      # callable in that module
  factory_kwargs: {}       # kwargs forwarded to the factory

excitation:
  entity: osc              # entity_id in the world
  port: force              # ExcitationPort field name on the component
  output_state: position   # state field to measure (response)

tests:
  - ...                    # see test types below
```

**Module resolution:** if the YAML file lives in the same directory as `world.py`,
set `module: world`.  `numen characterize` automatically adds the YAML file's
directory to `sys.path` so bare-name modules are importable.

---

## 3  Test types

### 3.1  `discrete_frequency_sweep`

Stepped sine: one solve per frequency, lock-in detection for amplitude and phase.
Most accurate FRF method.  Use for final Bode plots and Q measurement.

```yaml
- name: baseline_frf
  type: discrete_frequency_sweep
  frequencies:
    spacing: log         # log | linear
    f_start: 0.1         # Hz
    f_end: 10.0          # Hz
    n_points: 30
  amplitude: 0.01        # drive amplitude (keep small for linear regime)
  dc_offset: 0.0
  settle_periods: 50     # cycles before measurement starts — see §4 on choosing this
  measure_periods: 10    # cycles used for lock-in integration
```

### 3.2  `dc_operating_point_sweep`

Sweep DC bias; measure small-signal FRF at each operating point.
Maps how effective stiffness and damping change with bias — the first-order
fingerprint of a nonlinearity.

```yaml
- name: dc_bias_sweep
  type: dc_operating_point_sweep
  dc_values: [0.0, 0.5, 1.0, 2.0]
  probe_frequency: 1.0   # Hz — probe near resonance
  probe_amplitude: 0.005 # small relative to expected response
  settle_periods: 50     # must be long enough for DC step response to decay
  measure_periods: 10
```

### 3.3  `amplitude_sweep`

Fixed frequency, varying drive amplitude — reveals amplitude-dependent
nonlinearities.  A linear system shows flat |H|; softening or hardening shows
as a monotone slope.

```yaml
- name: amplitude_sweep_resonance
  type: amplitude_sweep
  frequency: 1.0         # Hz — usually set to f₀
  amplitudes: [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
  dc_offset: 0.0
  settle_periods: 30
  measure_periods: 5
```

### 3.4  `continuous_chirp`

Single solve with a frequency-swept input.  Much faster than a stepped sine but
with lower SNR and frequency resolution.  Use as a **quick survey** to locate
the resonance before running a targeted stepped-sine sweep.

```yaml
- name: chirp_survey
  type: continuous_chirp
  f_start: 0.1           # Hz
  f_end: 10.0            # Hz
  duration: 120.0        # s — longer gives finer frequency resolution
  amplitude: 0.05
  dc_offset: 0.0
  chirp_type: log        # log | linear
```

### 3.5  `parameter_sweep`

Repeat a named sub-test for each value of one model `ParameterField`.

```yaml
- name: c1_family
  type: parameter_sweep
  sweep_param: osc.c1    # dot-path: entity_id.field_name
  values: [0.0, 0.5, 1.0, 2.0, 5.0]
  sub_test: baseline_frf # must match another test name in the same plan
```

### 3.6  `parameter_grid`

Full factorial or pairwise grid over explicit value lists for multiple params.

```yaml
- name: c0_c1_grid
  type: parameter_grid
  params:
    osc.c0: [0.05, 0.1, 0.2]
    osc.c1: [0.5, 1.0, 2.0]
  sub_test: baseline_frf
  mode: full_factorial   # full_factorial | pairs
```

### 3.7  `doe_sweep`

Space-filling or classical DOE over **continuous** parameter ranges.

```yaml
- name: sensitivity_study
  type: doe_sweep
  design: latin_hypercube  # latin_hypercube | sobol | halton
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

## 4  Choosing `settle_periods`

The most common mistake is under-settling near resonance.  A forced oscillator
near its natural frequency approaches steady state on the timescale of the
**free-decay envelope**, not the forcing period.

| System | Time constant | Rule of thumb |
|---|---|---|
| Low Q (< 5) | τ ≈ 2Q/ω₀ | settle_periods = 10–20 is fine |
| Medium Q (5–20) | τ ≈ 2Q/ω₀ | settle_periods = 2–3 × Q |
| High Q (> 20) | τ ≈ 2Q/ω₀ | settle_periods ≈ 5 × Q |

**Example:** Q = 63 at f₀ = 1 Hz → τ ≈ 20 s → need ~100 s of settle at 1 Hz →
`settle_periods ≈ 100`.

**Practical workflow:**
1. Run a chirp or stepped-sine with short `settle_periods` (10–20) to *locate*
   the resonance and estimate Q.
2. Use `settle_periods ≈ 5 × Q` for the final accurate measurement.

**DC sweep caveat:** if the system has a high Q and you're probing near resonance,
the DC step response (from x=0 → x_eq) takes the same ~5τ to decay.  With
insufficient `settle_periods`, the residual transient at ω₀ will contaminate the
probe measurement via beat-frequency leakage into the lock-in.  Symptom: |H|
varies with DC even when ∂c/∂x is negligible.

---

## 5  Stepped sine vs chirp — when to use which

| Criterion | Stepped sine | Chirp |
|---|---|---|
| Accuracy | ✓✓ Best | ✓ Rough estimate |
| Speed | Slower (N solves) | Fast (1 solve) |
| High-Q systems | ✓ Works (with long settle) | ✗ Poor SNR |
| Low-Q / broadband | ✓ | ✓ Comparable |
| First look / survey | Overkill | ✓ Ideal |
| Q measurement | ✓ | Approximate only |

**Recommendation:** run a chirp first to locate the resonance and bandwidth, then
target a stepped sine sweep over a narrower frequency range with appropriate
`settle_periods`.

---

## 6  Loading and plotting results

```python
import json
import numpy as np
from numen.characterization.plots import plot_bode, plot_amplitude_sweep, plot_chirp_frf

# Load JSON
with open("results.json") as f:
    data = json.load(f)["results"]

# Or reload via runner (not yet implemented — use JSON for now)

# Plot a Bode diagram
for r in data:
    if r["type"] == "frf":
        import matplotlib.pyplot as plt
        freqs = np.array(r["frequencies"])
        mags  = np.array(r["magnitudes"])
        # manual plot or use FRFResult reconstruction:
        from numen.characterization.results import FRFResult
        frf = FRFResult(
            name=r["name"], frequencies=freqs, magnitudes=mags,
            phases_deg=np.array(r["phases_deg"]),
            f0=r["f0"], Q=r["Q"], damping_ratio=r["damping_ratio"],
            amplitude=r["amplitude"], dc_offset=r["dc_offset"],
        )
        fig, ax_m, ax_p = plot_bode(frf, db=True)
        plt.savefig("bode.png", dpi=150, bbox_inches="tight")
```

### Flatten to DataFrame

```python
# Requires: pip install pandas
from numen.characterization.runner import CharacterizationRunner
# ... or reconstruct CampaignResults from JSON manually ...

# If you have a CampaignResults object:
df = campaign_results.to_dataframe()
print(df.columns)
# Each row is one measurement; grid/DOE sweeps add parameter columns automatically.
```

---

## 7  Full example — nonlinear oscillator

See `examples/nonlinear_oscillator/` for a complete worked example:

```
examples/nonlinear_oscillator/
├── components.py          NLOscillatorComponent with ExcitationPort
├── dynamics.py            JAX-compatible dynamics
├── world.py               make_world() factory
├── run.py                 free-decay simulation + 4-panel plot
├── test_plan.yaml         4-test characterization campaign
├── characterize_plot.py   load results.json and generate summary plots
└── results.json           output from: numen characterize test_plan.yaml
```

Run the campaign:
```bash
cd examples/nonlinear_oscillator
uv run numen characterize test_plan.yaml --output results.json
uv run python characterize_plot.py
```

---

## 8  CLI reference

```
numen characterize PLAN [--output FILE] [--verbose]

  PLAN     Path to a YAML or JSON test plan.
  --output Save results to a JSON file.
  --verbose Enable DEBUG logging (shows per-solve timing, lock-in values, etc.)
```
