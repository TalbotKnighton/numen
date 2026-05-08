# Characterization Framework — Design Plan

---

## North Star

Build a **domain-agnostic characterization framework** that can run systematic
test campaigns against any Numen physics model — oscillators, thermal systems,
fluid networks, diffusion problems, chemical kinetics — without the framework
itself knowing anything about those domains.

The framework operates at the level of **signals and ports**. It applies waveforms
(steps, sinusoids, chirps, DOE-sampled inputs) to model input ports and records
responses from output states. Domain-specific knowledge lives entirely in the model
and its associated metric extractors; the framework is just the orchestration layer.

Long-term, computation lives in Julia (via the Julia server backend), and Python
handles schema, orchestration, and visualization only. This keeps the hot path in
the fastest available solver ecosystem without sacrificing the ergonomics of
Pydantic-validated declarative test plans.

### What does not exist yet

Nothing in the open-source ecosystem does this. The closest analogues are:

| Tool | Gap |
|---|---|
| MATLAB Control Toolbox | Linear systems only, closed ecosystem, Simulink-coupled |
| ModelingToolkit.jl / SciML | Modeling toolkit, not a characterization framework |
| DynamicalSystems.jl | Nonlinear analysis only (Lyapunov, attractors), not general characterization |
| Modelica / OpenModelica | Separate language, not composable with Python/Pydantic |
| AMESim / GT-Suite | Commercial, locked ecosystem |

This is genuinely novel. Design decisions should be made with that in mind.

---

## Overview

This document captures the design decisions made for a general-purpose characterization
test framework built on top of Numen. The goal is to run systematic frequency-domain
and operating-point experiments on any physical model, using a declarative YAML/JSON
test plan validated by Pydantic, with a Julia server backend that stays open across
the entire test campaign.

---

## Motivation

When you build a physics model, the first questions are always:
- What is the natural frequency?
- How much damping is present?
- How do those properties change with amplitude or operating point?

For linear systems these questions have clean analytical answers. For nonlinear
systems (like the position-dependent damping oscillator we built) they require
numerical experiments. This framework provides a structured, reusable way to run
those experiments on **any** Numen model without writing bespoke run scripts each time.

---

## Physics Background

### Linear systems: homogeneous + particular solution

For a linear ODE such as `ẍ + 2ζω·ẋ + ω²x = F(t)`, the general solution decomposes
cleanly:

```
x(t) = x_h(t)  +  x_p(t)
        free         forced
        response     response
```

The free response `x_h` decays to zero (for ζ > 0). The forced response `x_p` is a
steady-state sinusoid at the driving frequency. Superposition holds: you can solve
each part independently and add them. This is why Bode plots work — each frequency
is independent.

### Nonlinear systems: superposition breaks

For a nonlinear ODE such as `ẍ + (c₀ + c₁x²)ẋ + ω²x = F(t)`, **superposition does
not hold**. The free and forced responses are coupled — the damping coefficient
`c(x) = c₀ + c₁x²` depends on the instantaneous displacement, so the presence of
any forcing changes the effective damping seen by the free response, and vice versa.

You cannot separate the responses. You must solve the full ODE including all forcing
and read off the steady-state behavior directly.

### Operating-point linearization (the DC sweep technique)

Even though global superposition fails, nonlinear systems can be **locally linearized**
around an equilibrium point. This is the key technique behind the DC sweep test:

1. Apply a constant force `F_dc`. The system settles to a new equilibrium `x_eq`
   where the restoring force balances `F_dc`:
   `ω²·x_eq = F_dc`  →  `x_eq = F_dc / ω²`

2. At `x_eq`, the effective damping is `c(x_eq) = c₀ + c₁·x_eq²`.

3. Apply a small probe sine on top: `F(t) = F_dc + ε·sin(2πft)` with `ε ≪ 1`.
   The response is approximately linear with the modified parameters.

4. Sweep `F_dc` through a range of values → you map out how effective damping and
   effective stiffness change with operating point. This is the **first-order
   fingerprint of the nonlinearity**.

A purely linear system's frequency response function (FRF) does not change with DC
bias. Any change you observe is a direct measurement of the nonlinear character.

### Key measurement quantities

| Quantity | Symbol | How extracted |
|---|---|---|
| Natural (resonant) frequency | f₀ | Peak of magnitude FRF |
| Quality factor | Q | Q = f₀ / Δf₋₃dB (bandwidth method) |
| Damping ratio | ζ | ζ = 1 / (2Q) |
| Transfer function magnitude | \|H(f)\| | Output amplitude / input amplitude |
| Transfer function phase | ∠H(f) | Phase of output relative to input |
| Effective damping vs bias | c(x_eq) | From DC sweep → FRF peak at each bias |

---

## Domain-Agnostic Design Principle

### The temptation to avoid

As the framework grows to cover thermal, fluid, diffusion, and other domains, the
temptation is to add domain-specific test types: `oscillator_bode`, `thermal_step`,
`fluid_pressure_drop`. This leads to a combinatorial explosion of
`test_type × physics_domain` specializations and a framework that needs to be
rewritten every time a new domain is added.

### The right abstraction: ports and signals

Every physical domain shares the same port structure, captured by bond graph theory:

| Domain | Effort variable | Flow variable |
|---|---|---|
| Mechanical | Force [N] | Velocity [m/s] |
| Electrical | Voltage [V] | Current [A] |
| Thermal | Temperature [K] | Heat flux [W/m²] |
| Fluid | Pressure [Pa] | Volume flow [m³/s] |
| Chemical | Chemical potential [J/mol] | Molar flux [mol/s] |

Every test type — step, sine sweep, chirp, DOE — is simply applying a waveform to
an effort or flow port and recording the response at another port. The framework
never needs to know which domain it is in.

**Domain-specific knowledge lives in exactly two places:**

1. The model itself (components, dynamics, Julia functions)
2. **Metric extractors** — small functions registered by the model author that know
   how to compute physically meaningful scalars from a time series

```python
# metrics.py — written by the model author, not the framework
METRICS = {
    "f0":            extract_resonant_frequency,   # oscillator domain
    "Q":             extract_quality_factor,
    "settling_time": extract_settling_time,        # universal
    "overshoot":     extract_overshoot,            # universal
}
```

```python
# A thermal model registers different metrics
METRICS = {
    "time_constant":      extract_time_constant,
    "thermal_resistance": extract_steady_state_gain,
    "peak_temperature":   extract_peak,
}
```

The test plan names which metrics to extract; the framework resolves them against
the model's registry at runtime. The framework itself is unchanged.

### Port typing on ExcitationPort

`ExcitationPort` should carry physical metadata so the framework can generate
correct axis labels and perform basic dimensional checks without knowing the domain:

```python
force: Annotated[float, ExcitationPort(
    targets  = "velocity",
    domain   = "mechanical",   # optional, for documentation
    quantity = "force",        # effort or flow variable name
    units    = "N",            # SI units string
)] = 0.0
```

---

## Julia-First Architecture

### Division of responsibilities

The long-term target is for all computation to live in Julia, with Python handling
only schema, orchestration, and visualization:

```
Python                              Julia
─────────────────────────────────   ────────────────────────────────────
Schema definition (Pydantic)        Dynamics (OrdinaryDiffEq.jl)
Test plan parsing (YAML → dict)     Stiff/DAE solvers (Rodas5P, FBDF)
Test orchestration (runner.py)      Sensitivity (ForwardDiff.jl)
DOE sampling (scipy / pyDOE3)       Frequency analysis (DSP.jl)
Sensitivity indices (SALib)         Control analysis (ControlSystems.jl)
Visualization (matplotlib)          Bifurcation (BifurcationKit.jl)
Result serialization (JSON)
```

Metric extraction — currently imagined on the Python side — can migrate to Julia
using **DSP.jl** and **ControlSystems.jl** as the framework matures. This would
eliminate a Python round-trip per solve and keep the entire hot path in Julia.

### SciML ecosystem integration (future)

The Julia SciML ecosystem provides building blocks we should plan to integrate:

| Package | Capability |
|---|---|
| `ModelingToolkit.jl` | Symbolic model manipulation, automatic Jacobians |
| `SciMLSensitivity.jl` | Adjoint-based parameter sensitivity (faster than finite diff for large models) |
| `BifurcationKit.jl` | Bifurcation diagrams, stability analysis |
| `ControlSystems.jl` | Transfer function extraction, Bode, Nyquist, root locus |
| `DynamicalSystems.jl` | Lyapunov exponents, attractors, chaos characterization |

None of these require changes to the Python schema — they are Julia-side analysis
tools that consume the same solve results the current framework already produces.

---

## Architecture

```
src/numen/characterization/
├── schema.py        # Pydantic models for all spec types; YAML loads to these
├── runner.py        # CharacterizationRunner — opens backend once, runs all tests
├── excitation.py    # ExcitationPort field marker, ExcitationSystem, world wrapping
├── analysis.py      # Lock-in extraction, FFT, Q-factor fitting, peak finding
├── results.py       # Typed result containers (FRFResult, DCSweptFRF, etc.)
└── plots.py         # Bode, Nyquist, operating-point waterfall plots
```

### CharacterizationRunner

The runner is the top-level orchestrator. It:

1. Parses and validates the test plan (YAML → dict → Pydantic)
2. Opens the backend **once** (critical for Julia server: avoids repeated JIT)
3. Iterates through the test list in order
4. For each test, builds a world variant (modified parameters), compiles a spec,
   runs the solve, and collects results
5. Passes the open backend handle to every test so the server is never closed
   between runs

```python
runner = CharacterizationRunner.from_yaml("test_plan.yaml")
results = runner.run()    # Julia server opens here and stays open
results.save("results.json")
results.plot()
```

---

## ExcitationPort — Field Annotation Design

### Rationale

Excitation injection points should be **declared by the model author**, not discovered
or guessed by the test framework. This is the physically correct approach: in real
systems, input channels are structural — a force port on a mass, a pressure inlet on
a control volume. The model author knows where external forcing can be applied; the
framework should respect that.

This also constrains the test framework to valid experiments. Injecting an arbitrary
force into an arbitrary derivative would often be physically meaningless.

### Usage

```python
from numen.fields import IntegratedField, ParameterField, ExcitationPort

class NLOscillatorComponent(Component):
    kind:     Literal["nl_oscillator"] = "nl_oscillator"
    position: Annotated[float, IntegratedField()] = 1.0
    velocity: Annotated[float, IntegratedField()] = 0.0
    omega:    Annotated[float, ParameterField()]  = 1.0
    c0:       Annotated[float, ParameterField()]  = 0.1
    c1:       Annotated[float, ParameterField()]  = 1.0
    # Declares this as a force input port that adds to d(velocity)/dt
    force:    Annotated[float, ExcitationPort(targets="velocity")] = 0.0
```

`ExcitationPort(targets="velocity")` means: "the value of `force` is added to
`d(velocity)/dt` by the framework's ExcitationSystem."

At the component level, `force` behaves like a `ParameterField` — it is constant
for any single solve. The characterization framework varies it across solves by
setting `force_amp`, `force_freq`, and `force_dc` in the compiled parameter vector.

### What the framework adds automatically

When `compile_spec` (or a wrapper) detects an `ExcitationPort`, it:
1. Adds three `ParameterField`s to the component: `{port}_amp`, `{port}_freq`, `{port}_dc`
2. Adds a generic `ExcitationSystem` to the world that evaluates
   `F(t) = amp·sin(2π·freq·t) + dc` and injects it into the target derivative

The dynamics function itself does not need to be modified to support excitation.

### Multiple ports

Components can declare multiple excitation ports. Multi-mass systems can have a
force port on each mass:

```python
class MassComponent(Component):
    ...
    force: Annotated[float, ExcitationPort(targets="velocity")] = 0.0
```

The test schema specifies which entity's port to drive: `entity: mass_1, port: force`.

---

## Schema Design

### YAML → Pydantic pattern

YAML is the human-readable authoring format. Pydantic is the validation layer.
They compose without any magic:

```python
import yaml
config = CharacterizationConfig.model_validate(yaml.safe_load(open("test_plan.yaml")))
```

We never bypass Pydantic. The YAML is just a serialization of the dict that
`model_validate` would otherwise receive.

### Full example test plan

```yaml
version: "1.0"

backend:
  type: julia_server          # scipy | jax | julia | julia_server
  julia_file: dynamics.jl
  rtol: 1e-8
  atol: 1e-9

model:
  module: examples.nonlinear_oscillator.world
  factory: make_world
  factory_kwargs:
    omega: 6.283              # 1 Hz natural frequency
    c0: 0.1
    c1: 2.0
  metrics: examples.nonlinear_oscillator.metrics   # optional METRICS registry

excitation:
  entity: osc
  port: force                 # must match an ExcitationPort field name
  output_state: position      # field to measure

tests:

  - name: baseline_frf
    type: discrete_frequency_sweep
    frequencies:
      spacing: log
      f_start: 0.1
      f_end: 20.0
      n_points: 40
    amplitude: 0.01           # small — linear regime
    dc_offset: 0.0
    settle_periods: 10        # wait this many cycles before measuring
    measure_periods: 5        # integrate over this many cycles

  - name: dc_bias_sweep
    type: dc_operating_point_sweep
    dc_values: [0.0, 0.3, 0.6, 1.0, 1.5, 2.0]
    probe_frequency: 1.0      # Hz — probe near resonance
    probe_amplitude: 0.001    # small-signal (differential) probe
    settle_periods: 10
    measure_periods: 5

  - name: amplitude_sweep
    type: amplitude_sweep
    frequency: 1.0            # Hz — fixed at resonance
    amplitudes: [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
    dc_offset: 0.0
    settle_periods: 10
    measure_periods: 5

  - name: parameter_family
    type: parameter_sweep
    sweep_param: osc.c1       # any ParameterField in the model
    values: [0.0, 0.5, 1.0, 2.0, 5.0]
    sub_test: baseline_frf    # reference a named test to repeat at each value

  - name: chirp_sweep
    type: continuous_chirp
    f_start: 0.1
    f_end: 20.0
    duration: 120.0
    amplitude: 0.01
    dc_offset: 0.0
    chirp_type: log           # log | linear
```

### Pydantic model sketch

```python
class BackendSpec(BaseModel):
    type: Literal["scipy", "jax", "julia", "julia_server"]
    julia_file: str | None = None
    rtol: float = 1e-8
    atol: float = 1e-9

class ModelSpec(BaseModel):
    module: str
    factory: str
    factory_kwargs: dict[str, Any] = {}
    metrics: str | None = None   # optional module path to METRICS registry

class ExcitationSpec(BaseModel):
    entity: str
    port: str
    output_state: str

class DiscreteFrequencySweepSpec(BaseModel):
    name: str
    type: Literal["discrete_frequency_sweep"]
    frequencies: FrequencyGridSpec
    amplitude: float
    dc_offset: float = 0.0
    settle_periods: int = 10
    measure_periods: int = 5

# ... other test spec types ...

TestSpec = Annotated[
    DiscreteFrequencySweepSpec | DCOperatingPointSweepSpec | AmplitudeSweepSpec
    | ParameterSweepSpec | ContinuousChirpSpec,
    Field(discriminator="type"),
]

class CharacterizationConfig(BaseModel):
    version: str
    backend: BackendSpec
    model: ModelSpec
    excitation: ExcitationSpec
    tests: list[TestSpec]
```

---

## Test Types — Detailed Descriptions

### 1. Discrete Frequency Sweep (stepped sine)

**What it does:** Runs one solve per frequency point. At each frequency `f`:
- Build world with `force_amp = A`, `force_freq = f`, `force_dc = DC`
- Solve for `t_settle + t_measure` seconds where `t_settle = settle_periods / f`
- Discard the settling transient; extract amplitude and phase from the last
  `measure_periods` cycles using lock-in detection (multiply by sin/cos, lowpass)
- Record `|H(f)| = output_amp / input_amp` and `∠H(f)`

**Output:** Full Bode plot. Peak gives f₀; -3dB bandwidth gives Q.

**When to use:** When you want clean, high-SNR measurements at each frequency.
Slower than chirp (N separate solves) but more accurate.

### 2. DC Operating Point Sweep

**What it does:** Sweeps the DC bias through a list of values. At each bias:
- Settle to the new equilibrium under `F_dc` alone (no probe)
- Apply a small-amplitude probe sine at the specified frequency
- Extract the local (linearized) gain and phase

**Output:** How the small-signal transfer function changes with operating point.
For the nonlinear oscillator, this maps out `c(x_eq) = c₀ + c₁·x_eq²` directly.

**When to use:** First diagnostic for nonlinearity. If the FRF changes with DC
bias, the system is nonlinear. If it doesn't, it's linear (in that range).

### 3. Amplitude Sweep

**What it does:** Fixed frequency, vary drive amplitude. Measures whether
output amplitude scales linearly with input (linear regime) or deviates (nonlinear
regime — amplitude-dependent resonance, harmonic generation).

**Output:** Output amplitude vs input amplitude at a fixed frequency.
Deviation from a straight line is the nonlinearity signature.

**Note:** This is the entry point for the more advanced characterization work
(harmonic analysis, Volterra kernels) planned for a future phase.

### 4. Parameter Sweep (wrapper)

**What it does:** Repeats any named test for multiple values of a model
ParameterField. For example, run `baseline_frf` for `c1 ∈ {0, 0.5, 1, 2, 5}`.

**Output:** A family of FRFs or other results showing how system character
depends on a physical parameter. Useful for design sweeps and sensitivity analysis.

### 5. Multi-Parameter Grid

**What it does:** Full (or paired) factorial over explicit value lists for two or
more parameters. The `full_factorial` mode generates every combination; `pairs`
only sweeps the diagonal (N points instead of N^k).

```yaml
- type: parameter_grid
  params:
    osc.c0: [0.1, 0.5, 1.0]
    osc.c1: [0.0, 1.0, 2.0]
  sub_test: baseline_frf
  mode: full_factorial          # or pairs
```

**Output:** Same as a single-parameter sweep but indexed by a multi-dimensional
parameter tuple. Results are tabulated as `(c₀, c₁, f₀, Q, ...)` rows suitable
for response surface plotting or surrogate fitting.

### 6. DOE Sweep

**What it does:** Generates a design matrix using a statistical sampling strategy,
then runs the sub-test at each design point. Suited for exploring a continuous
parameter space efficiently without specifying explicit value lists.

```yaml
- type: doe_sweep
  design: latin_hypercube       # sobol | halton | central_composite | full_factorial
  n_samples: 50                 # for space-filling designs
  params:
    osc.c0: { min: 0.01, max: 2.0, scale: log }
    osc.c1: { min: 0.0,  max: 5.0, scale: linear }
  sub_test: baseline_frf
  outputs: [f0, Q, settling_time]   # scalar summaries to tabulate per design point
```

The `outputs` list extracts scalar summaries from each sub-test result into a flat
table `(c₀, c₁, ...) → (f₀, Q, ...)` that can be passed directly to a surrogate
model or sensitivity analysis.

### 7. Continuous Chirp Sweep

**What it does:** One long solve with a frequency-swept input:
`F(t) = A · sin(φ(t))` where `φ(t)` is a log or linear chirp phase.

**Output:** Spectrogram or short-time FFT gives FRF estimate in a single solve.
Faster than stepped sine but lower frequency resolution and SNR.

**When to use:** Quick first look, or when the number of frequency points makes
stepped sine too slow.

---

## Signal Analysis

### Lock-in detection (for discrete swept sine)

For each solved time series, extract amplitude and phase at the drive frequency `f`:

```python
def lock_in(t, y, f, t_start):
    mask = t >= t_start
    t_m, y_m = t[mask], y[mask]
    I = 2 * np.trapz(y_m * np.sin(2*np.pi*f*t_m), t_m) / (t_m[-1] - t_m[0])
    Q = 2 * np.trapz(y_m * np.cos(2*np.pi*f*t_m), t_m) / (t_m[-1] - t_m[0])
    amplitude = np.sqrt(I**2 + Q**2)
    phase     = np.arctan2(Q, I)
    return amplitude, phase
```

This is numerically cleaner than FFT for single-frequency extraction and
handles non-integer numbers of cycles correctly.

### Q-factor extraction from FRF

Given a magnitude FRF `|H(f)|`:
1. Find the peak at f₀ (resonant frequency)
2. Find the two -3dB frequencies `f₁ < f₀ < f₂` where `|H| = |H_peak| / √2`
3. `Q = f₀ / (f₂ - f₁)`

For the nonlinear oscillator, Q will be amplitude-dependent. The DC sweep maps
out how Q changes with operating point.

---

## Julia Server Integration

The Julia server must be opened **once** before the test loop and reused for all
solves. This is the key to performance: Julia JIT-compiles the dynamics on the
first solve; all subsequent solves hit the compiled kernel (~14 ms each).

```python
# Pseudocode — runner.py
with JuliaServerBackend(julia_file=config.backend.julia_file, ...) as backend:
    for test_spec in config.tests:
        for world_variant in test_spec.iter_worlds(base_world):
            spec   = compile_spec(world_variant)
            result = backend.solve(spec, tspan=...)
            results.append(analyze(result, test_spec))
```

For a 40-point frequency sweep at ~14 ms/solve, the full campaign takes
~560 ms warm. With scipy it would be ~360 seconds.

---

## DOE & Parameter Sweep Dependencies

The sampling math and sensitivity analysis are solved problems — we use existing
packages rather than writing our own. All are either already available (scipy) or
small, well-maintained libraries added as optional extras.

| Responsibility | Package | How we use it |
|---|---|---|
| LHS, Sobol, Halton sampling | `scipy.stats.qmc` | Already a dependency — free |
| Full/fractional factorial, CCD, Box-Behnken | `pyDOE3` | Classical DOE design matrices |
| Sobol sensitivity indices | `SALib` | Which parameters drive f₀ and Q most |
| Response surface / surrogate fitting | `scikit-learn` | GP, RBF after the sweep |

**What we write ourselves** is only the glue layer (~100 lines):
1. Translate the YAML spec into a call to the appropriate sampler
2. Iterate design points, call the backend, collect scalar outputs
3. Return a `pandas.DataFrame` of `(param_1, param_2, ..., f₀, Q, ...)`

The math of generating design matrices and computing sensitivity indices is entirely
delegated to those packages.

### pyproject.toml optional extras

```toml
[project.optional-dependencies]
characterization = [
    "pyDOE3",
    "SALib",
]
```

Install with: `uv pip install "numen[characterization]"`

### Typical post-sweep workflow

```python
# 1. Run the DOE sweep → get a DataFrame
df = results.to_dataframe()   # columns: c0, c1, f0, Q, settling_time

# 2. Sensitivity analysis — which parameter drives Q most?
from SALib.analyse import sobol
Si = sobol.analyze(problem, df["Q"].values)

# 3. Surrogate model for fast interpolation
from sklearn.gaussian_process import GaussianProcessRegressor
gp = GaussianProcessRegressor().fit(df[["c0", "c1"]], df["Q"])
```

---

## Implementation Roadmap

### Phase 1 — Foundation

- [ ] `ExcitationPort` field marker in `src/numen/fields.py`
- [ ] `ExcitationSystem` in `src/numen/characterization/excitation.py`
- [ ] Update nonlinear oscillator to declare force port
- [ ] Pydantic schema: `BackendSpec`, `ModelSpec`, `ExcitationSpec`, all test specs
      in `src/numen/characterization/schema.py`
- [ ] `CharacterizationRunner` skeleton with Julia server context manager

### Phase 2 — Test Implementations

- [ ] `DiscreteFrequencySweep` runner
- [ ] Lock-in analysis and Q-factor extraction in `analysis.py`
- [ ] `DCOperatingPointSweep` runner
- [ ] `ParameterSweep` wrapper
- [ ] Bode plot and operating-point waterfall in `plots.py`
- [ ] Result serialization to JSON

### Phase 3 — Extended Tests & DOE

- [ ] `AmplitudeSweep` runner
- [ ] `ContinuousChirpSweep` runner (chirp signal generation + STFT analysis)
- [ ] `ParameterGrid` runner (full/paired factorial over explicit value lists)
- [ ] `DOESweep` runner using `scipy.stats.qmc` (LHS, Sobol) and `pyDOE3` (CCD, BBD)
- [ ] `results.to_dataframe()` — flatten scalar outputs to pandas DataFrame
- [ ] SALib integration for Sobol sensitivity indices on DOE results
- [ ] YAML test plan loader + CLI: `uv run numen characterize test_plan.yaml`

### Phase 4 — Nonlinearity Characterization (future)

- [ ] Harmonic distortion measurement (THD, individual harmonics)
- [ ] Amplitude-dependent resonance tracking
- [ ] Volterra kernel identification (second-order frequency response)
- [ ] Modulation / intermodulation tests

---

## Open Questions

1. **Julia dynamics for ExcitationSystem** — the generic ExcitationSystem
   needs a corresponding `dynamics.jl` function. Should this be auto-generated
   or hand-written once and shipped with the framework?

2. **Settle detection vs fixed settle time** — currently settling is specified
   as `settle_periods` (a fixed number of cycles). An adaptive approach that
   monitors when the transient has decayed below a threshold would be more
   robust, especially for heavily damped systems where the true settling time
   is much shorter.

3. **Result schema versioning** — results saved as JSON should carry the full
   test spec and model parameters so they are self-documenting. Design the
   result schema before implementing serialization.

4. **CLI integration** — `uv run numen characterize test_plan.yaml` should
   display a Rich progress table (one row per test, live-updating) consistent
   with the existing CLI style.
