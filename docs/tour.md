# A guided tour of Numen

A 10-minute walkthrough you can read end-to-end — what Numen is for, what makes it different, and how a real engineering simulation gets written. No prior context required.

If a piece sparks a question, every section links to the deep dive.

---

## What's Numen for?

Numen is a Python-first framework for **engineering dynamics simulation** — the ODEs and DAEs that come out of mechanical, fluid, thermal, electrical, and multi-domain systems. It exists because the usual options force a trade-off:

- **Modelica / Simulink** — great for modelling, hostile for scripting, automation, CI, and optimization loops
- **Hand-rolled scipy** — friendly to write, but slow at scale and you write everything from scratch each time
- **JAX/diffrax** — fast and differentiable, but explicit-only solvers diverge on stiff problems, and there's no DAE path
- **Bare Julia** — the right solver ecosystem, but every model needs hand-built infrastructure

Numen splits the problem in two:
- **Modelling** stays in **Python** with Pydantic-typed components
- **Solving** happens through whichever backend makes sense — scipy for dev, **Julia for production**, JAX for autodiff

The Julia backend is a thin wrapper over [`OrdinaryDiffEq.jl`](https://docs.sciml.ai/OrdinaryDiffEq/stable/) — ~150 solvers including stiff Rosenbrock methods (`Rodas5P`), implicit RK (`KenCarp4`), multistep (`FBDF`, `QNDF`), and a mass-matrix DAE path. That's the whole reason this framework exists: keep authoring ergonomic, get production-grade solvers for free.

---

## Five-minute tour: a damped oscillator

Let's build a 1D damped harmonic oscillator end-to-end. This is what `numen run oscillator` does.

### 1. Components — your data

A `Component` is a Pydantic model that describes one piece of the world. Field annotations declare which numbers go into the integrated state vector and which are constant parameters.

```python
# components.py
from typing import Annotated, Literal
from numen.spec.component import Component
from numen.fields import IntegratedField, ParameterField

class OscillatorComponent(Component):
    kind:     Literal["oscillator"] = "oscillator"
    position: Annotated[float, IntegratedField()] = 1.0   # in state vector x
    velocity: Annotated[float, IntegratedField()] = 0.0   # in state vector x
    omega:    Annotated[float, ParameterField()]  = 6.28  # in parameter vector p
    damping:  Annotated[float, ParameterField()]  = 0.05  # in parameter vector p
```

That's it. Numen's compiler walks every component on every entity, packs the integrated fields into one flat state vector `x`, the parameters into `p`, and builds an index map so dynamics functions can read and write each named slot.

### 2. Systems — your physics

A `System` is a stateless function that writes derivatives. It receives the state vector, parameter vector, and a `spec` object that hides indexing detail.

```python
# dynamics.py
from typing import ClassVar, Literal
from numen.spec.system import System, DynamicsFn
from components import OscillatorComponent

def oscillator_dynamics(dx, x, p, t, spec, system):
    # ẋ = v,  v̇ = -ω²x - 2ζωv
    for (eid,) in system.entity_groups:
        o  = spec.view(eid, OscillatorComponent, x, p)   # read state + params
        do = spec.dx_view(eid, OscillatorComponent, dx)  # write derivatives
        do.position += o.velocity
        do.velocity += -o.omega**2 * o.position - 2*o.damping*o.omega*o.velocity

class OscillatorSystem(System):
    component_types: ClassVar = (OscillatorComponent,)
    python_fn:       ClassVar[DynamicsFn] = staticmethod(oscillator_dynamics)
    kind:            Literal["oscillator"] = "oscillator"
    dynamics_fn:     str = "OscillatorDynamics.oscillator_dynamics!"
```

Two things to notice:

- `spec.view(eid, ComponentType, x, p)` returns a typed accessor — `o.position` is a real number, not a magic index. Numen looks the index up once.
- `system.entity_groups` is a list of tuples — for a single-entity system each tuple has one element. For a multi-entity coupled system (like a spring connecting two masses) it has more, and you destructure: `for (mass_a, spring, mass_b) in system.entity_groups:`.

### 3. World — your model instance

```python
# world.py
from numen.spec.world import GenericWorld
from components import OscillatorComponent
from dynamics import OscillatorSystem

World = GenericWorld[OscillatorComponent, OscillatorSystem, None]

def make_world():
    return World(
        components={"osc": {"oscillator": OscillatorComponent(position=1.0, omega=6.28, damping=0.05)}},
        systems={"sys":  OscillatorSystem()},
    )
```

The `components` dict is keyed by entity ID, then by component kind — this lets one entity carry multiple component types (a sensor on a body, an orifice on a control volume, etc.).

### 4. Solve

```python
# run.py
from numen.compiler.flatten import compile_spec
from numen.bridge.scipy_backend import ScipyBackend
from numen.reconstruction.collector import SnapshotCollector
from world import make_world

world  = make_world()
spec   = compile_spec(world)                                    # flatten to vectors
result = ScipyBackend(rtol=1e-9, atol=1e-9).solve(spec, (0.0, 5.0))

collector = SnapshotCollector(world, spec, result)
t, position = collector.field_series("osc", "oscillator", "position")
```

Run it: `python run.py`. That's the whole framework loop in 30 lines.

---

## Multi-entity topology — coupling

Real models couple entities. Three masses connected by two springs:

```
   m1 ── s1 ── m2 ── s2 ── m3
```

The trick: declare the **slot pattern** on the system, then provide explicit topology when you instantiate it:

```python
class SpringForceSystem(System):
    component_types: ClassVar = ()                 # no auto-population
    entity_slots:    ClassVar = EntityGroup(MassComponent, SpringComponent, MassComponent)
    python_fn:       ClassVar[DynamicsFn] = staticmethod(spring_force_dynamics)
    kind:            Literal["spring_force"] = "spring_force"
    dynamics_fn:     str = "SpringDynamics.spring_force_dynamics!"

# In make_world:
spring_sys = SpringForceSystem(entity_groups=[
    ["m1", "s1", "m2"],
    ["m2", "s2", "m3"],     # m2 appears in both groups — forces accumulate via +=
])
```

Inside `spring_force_dynamics`:

```python
def spring_force_dynamics(dx, x, p, t, spec, system):
    for id_a, id_s, id_b in system.entity_groups:
        ma = spec.view(id_a, MassComponent, x, p)
        sp = spec.view(id_s, SpringComponent, x, p)
        mb = spec.view(id_b, MassComponent, x, p)
        force = sp.k * ((mb.position - ma.position) - sp.rest_length)
        spec.dx_view(id_a, MassComponent, dx).velocity +=  force / ma.mass
        spec.dx_view(id_b, MassComponent, dx).velocity += -force / mb.mass
```

Run `numen run coupled_spring` to see this in action — energy conservation drift is around 1e-8 J over 10 seconds.

---

## A real engineering problem: pneumatic poppet check valve

Now the kind of model Numen is actually built for. A pneumatic check valve has:

- Four control volumes (inlet tank → inlet pipe → outlet pipe → outlet tank)
- Compressible orifice flow between them (choked / unchoked branches)
- A spring-loaded poppet that lifts off its seat at a cracking pressure
- Hard stops at the closed seat and at maximum travel
- Variable flow area through the poppet as a function of position

This is a **stiff, coupled, contact-rich** problem with 6 state variables and 5 systems. It's the kind of thing JAX struggles with and `Rodas5P` handles cleanly.

The model lives at [`src/numen/examples/fluid_poppet/`](https://github.com/TalbotKnighton/numen/tree/main/src/numen/examples/fluid_poppet) — `numen run fluid_poppet` runs it with scipy in about 9 seconds. Switching to the Julia backend with `Rodas5P` brings that to ~14 ms warm.

A few patterns worth seeing:

**Smooth contact.** A bare `max(0, -pos)` stop has a C0 kink that wrecks adaptive ODE solvers (thousands of rejected steps near contact). Numen's examples use a C1-smooth ramp over 1 µm:

```python
_STOP_DELTA = 1e-6   # 1 µm

def _soft_pen(x):
    return jnp.where(
        x <= 0.0, 0.0,
        jnp.where(x >= _STOP_DELTA, x - 0.5 * _STOP_DELTA, 0.5 * x * x / _STOP_DELTA)
    )
```

**Isentropic compressible orifice flow** — the standard helper, with both choked and unchoked branches and `jnp.where` to keep it JAX-traceable.

**Multi-domain coupling** — the poppet's mechanics system reads pressures from neighbouring control volumes (`F_pressure = (P_in - P_out) * seat_area`), and the flow systems read the poppet's position to compute opening area. All of it is just `+=` on `dx` slots.

---

## Killer feature: characterization without Python

Most simulation frameworks make you write a Python script every time you want to sweep a parameter, run a frequency response, or build a Bode plot. Numen has a domain-agnostic characterization engine you drive from a YAML file.

### Step 1 — Mark an excitation port on your component

Add an `ExcitationPort` field to whichever component receives the test signal. It declares which integrated field's derivative the framework should add `F(t)` into. The port itself is metadata only — it isn't compiled into `x` or `p` and you never reference it from the dynamics function:

```python
# components.py
from typing import Annotated, Literal
from numen.spec.component import Component
from numen.fields import IntegratedField, ParameterField, ExcitationPort

class PistonComponent(Component):
    kind:     Literal["pneumatic_dashpot"] = "pneumatic_dashpot"
    position: Annotated[float, IntegratedField()] = 0.0
    velocity: Annotated[float, IntegratedField()] = 0.0
    mass:     Annotated[float, ParameterField()]  = 0.5

    # Excitation input port — effort source (force).
    # targets="velocity": F(t) is added to d(velocity)/dt during the solve.
    force: Annotated[float, ExcitationPort(
        targets   = "velocity",   # IntegratedField whose derivative receives F(t)
        port_type = "effort",     # "effort" (force/pressure/voltage) or "flow"
        units     = "N",
    )] = 0.0
```

That's the entire change — no edit to the dynamics function. The framework injects `F(t) = amp·sin(2π·f·t) + dc` (or a chirp, two-tone, stochastic PSD, etc.) into the targeted derivative for you.

### Step 2 — Write the test plan

```yaml
# yaml-language-server: $schema=test_plan.schema.json

backend:
  type: julia_server
  julia_file: dynamics.jl
  method: Rodas5P
  n_workers: 4
  n_save_points: 2000

model:
  module: world
  factory: make_world

excitation:
  entity: piston                       # entity_id in your World
  component: pneumatic_dashpot         # component kind that owns the port
  port: force                          # ExcitationPort field name
  output_state: position               # state field to measure (response)

tests:
  - name: chirp_survey
    type: continuous_chirp
    f_start: 0.5
    f_end: 200.0
    duration: 10.0
    amplitude: 4.0

  - name: frf_vs_orifice
    type: parameter_sweep
    sweep_param: piston.pneumatic_dashpot.orifice_area
    values: [1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 5.0e-4]
    sub_test: chirp_survey

plots:
  output: characterization_summary.png
  panels:
    - type: parameter_family
      test: frf_vs_orifice
```

### Step 3 — Run it

```bash
numen characterize test_plan.yaml
```

The framework:

1. Adds force excitation `F(t) = amp·sin(2π·f·t) + dc` to the entity's port
2. Sweeps the orifice area across five decades using 4 parallel pre-warmed Julia workers
3. Lock-in detects FRF magnitude and phase from the chirp response
4. Renders a 5-curve Bode family with a colour bar

Available test types: stepped sine FRF, continuous chirp, amplitude sweep (nonlinearity), DC operating point sweep, parameter sweep, parameter grid (full factorial), DOE sweep (Latin hypercube / Sobol / CCD / BBD), two-tone IMD, harmonic distortion, free-decay backbone, phase portrait, stochastic excitation (random vibration). Each has a YAML schema documented in [CHARACTERIZATION.md](https://github.com/TalbotKnighton/numen/blob/main/src/numen/init_data/CHARACTERIZATION.md).

The parameter key `piston.pneumatic_dashpot.orifice_area` is the full three-level path `entity_id.component_kind.field_name`. Validation runs at campaign start — typos fail immediately with the list of valid keys.

---

## Why Julia is the strategic backend

Numen ships three solver backends:

| Backend | Use for | Stiff? | DAE? |
|---|---|:---:|:---:|
| **`JuliaServerBackend`** ⭐ | Production, stiff problems, parameter sweeps | ✓ | ✓ |
| `ScipyBackend` | Development, debugging | (LSODA only) | — |
| `JAXBackend` | Autodiff *through* the solve | weak | — |

Three things make Julia the strategic choice:

1. **Solver ecosystem.** The `method=` argument forwards directly to `OrdinaryDiffEq.jl`. You get `Tsit5`, `Dopri5`, `Vern7`, `Vern9` for non-stiff; `Rodas5P`, `Rosenbrock23` for stiff; `KenCarp4`, `TRBDF2`, `FBDF`, `QNDF` for very stiff or large; symplectic, IMEX, and exponential families. JAX/diffrax has a much smaller set and its implicit solvers are slow.

2. **Stiff problems and DAEs.** Real engineering systems are almost always stiff. JAX's explicit solvers diverge on them; Numen's `Rodas5P` path handles them with a sparse Jacobian (Jacobian cost is O(group_coupling_width), not O(state_size)) and an explicit time gradient (avoids OrdinaryDiffEq's `FunctionWrapper` issue with `Dual` `t`). DAEs via `ContinuousField(algebraic=True)` and the mass-matrix path are Julia-only.

3. **JIT amortisation.** `JuliaServerBackend` keeps a hot Julia process alive across the whole session — pay JIT compilation **once**, every solve after is warm. `JuliaServerPool` runs N pre-warmed workers in parallel; the framework's `precompile()` finishes before the first real solve.

Don't trust the "JAX 1500× / Julia 600×" benchmark on the home page in isolation — it's a non-stiff toy problem and JAX wins it. Real engineering systems flip the ranking, and JAX often fails outright. **Always benchmark your own model.**

---

## Where to go next

**Build something.** `numen init my_project --domain mechanical` scaffolds a CLAUDE-friendly project with all the context files (`CLAUDE.md`, `JULIA.md`, `CHARACTERIZATION.md`, `DESIGN.md`, `test_plan.schema.json`) plus a working starter model.

**Read the deep dives.**

- [Getting Started](getting-started.md) — installation, first model, CLI reference
- [User Guide → Backends](guide/backends.md) — full backend rundown, when to pick which solver
- [User Guide → Components & Fields](guide/components.md) — every field type, vector fields, ExcitationPort
- [User Guide → Systems & Dynamics](guide/systems.md) — coupling, JAX rules, hot-loop tips
- [User Guide → Characterization](guide/characterization.md) — every test type, plot panel, DOE schema
- [Julia Reference](julia.md) — `dynamics.jl` writing guide, helpers, performance notes

**See the worked examples.**

- [Oscillator](examples/oscillator.md) — minimal end-to-end (Python + Julia)
- [Coupled Spring](examples/coupled_spring.md) — multi-entity, energy conservation
- [Fluid Poppet](examples/fluid_poppet.md) — full stiff multi-domain model
- [Pneumatic Dashpot Julia](examples/pneumatic_dashpot_julia.md) — stiff Rodas5P + DAE

**Or just**: `numen run fluid_poppet` and read the source.
