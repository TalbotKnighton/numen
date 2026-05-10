# Characterization Framework

The characterization system runs test campaigns from a single YAML file and
generates plots without any Python scripting.

## Quick start

```bash
uv run numen characterize test_plan.yaml          # run tests + plot
uv run numen characterize test_plan.yaml -c       # run tests only → results.json
uv run numen characterize test_plan.yaml -p       # plot only → reads results.json
```

## Adding an ExcitationPort

The characterization framework identifies which component fields accept excitation
signals via `ExcitationPort` annotations.

```python
from numen.fields import ExcitationPort

class MassComponent(Component):
    kind:     Literal["mass"] = "mass"
    position: Annotated[float, IntegratedField()] = 0.0
    velocity: Annotated[float, IntegratedField()] = 0.0
    mass:     Annotated[float, ParameterField()]  = 1.0
    force:    Annotated[float, ExcitationPort(
                  targets   = "velocity",   # derivative of this field receives F(t)
                  port_type = "effort",     # "effort" (force/pressure) or "flow" (velocity)
                  units     = "N",
              )] = 0.0
```

`ExcitationPort` fields are **not** compiled into `p`. They are pure metadata
used by `inject_excitation()` to add a time-varying forcing system post-compilation.

## Test plan YAML structure

```yaml
version: "1.0"

backend:
  type: julia_server          # scipy | jax | julia | julia_server
  julia_file: dynamics.jl
  n_save_points: 2000
  n_workers: 4               # optional: parallel workers for sweep tests

model:
  module: world              # Python module with make_world factory
  factory: make_world

excitation:
  entity: osc               # entity_id that has the ExcitationPort
  port: force               # field name of the ExcitationPort
  output_state: position    # field name to observe

tests:
  - name: baseline_frf
    type: discrete_frequency_sweep
    frequencies:
      spacing: log
      f_start: 0.1
      f_end: 50.0
      n_points: 40
    amplitude: 0.01
    settle_periods: 50
    measure_periods: 10

  - name: frf_survey
    type: continuous_chirp
    f_start: 0.1
    f_end: 100.0
    amplitude: 0.01
    duration: 30.0

plots:
  - name: bode
    type: bode
    test: baseline_frf

  - name: chirp_plot
    type: chirp_timeseries
    test: frf_survey
```

## Test types

| Type | Description |
|---|---|
| `discrete_frequency_sweep` | Stepped sine; one solve per frequency; lock-in FRF detection |
| `continuous_chirp` | Single chirp solve; fast frequency survey |
| `amplitude_sweep` | Fixed frequency, varying amplitude; nonlinearity signature |
| `dc_operating_point_sweep` | DC bias sweep; small-signal FRF at each operating point |
| `parameter_sweep` | Outer loop over one parameter; inner loop is any sub-test |
| `parameter_grid` | Full factorial or pairwise grid over multiple parameters |
| `doe_sweep` | Space-filling DOE (LHS, Sobol, Halton) or classical designs (CCD, BBD) |

DOE sweeps require `pip install "numen[characterization]"` (pyDOE3, SALib, pandas).

## `excitation.*` parameter paths

`parameter_sweep`, `parameter_grid`, and `doe_sweep` can vary excitation inputs
as the outer sweep dimension:

```yaml
- name: frf_vs_amplitude
  type: parameter_sweep
  sweep_param: excitation.amplitude   # also: excitation.frequency, excitation.dc_offset
  values: [0.001, 0.01, 0.1, 1.0]
  sub_test: baseline_frf
```

This generates a `ParameterFamilyResult` rendered as a family of Bode curves
coloured by amplitude.

## Parallel execution

Add `n_workers: N` to `backend:` to run sweep tests in parallel:

```yaml
backend:
  type: julia_server
  julia_file: dynamics.jl
  n_workers: 4
  n_save_points: 2000
```

CLI override: `numen characterize test_plan.yaml --workers 4`

Each worker precompiles all dynamics at startup so the first real solve
carries no JIT latency.

## Plot panel types

| Type | Renders |
|---|---|
| `bode` | Magnitude + phase from `discrete_frequency_sweep` |
| `chirp_timeseries` | Time-domain chirp with spectrogram |
| `amplitude_sweep` | Peak response vs. amplitude |
| `dc_sweep` | Small-signal FRF vs. DC operating point |
| `parameter_family` | Overlaid curves from `parameter_sweep` |
| `doe_scatter` | Scatter matrix from `doe_sweep` |
| `parameter_grid_heatmap` | Heatmap from `parameter_grid` |

Use `enabled: false` on any test or plot panel to skip it without deleting it.
