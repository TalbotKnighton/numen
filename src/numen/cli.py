"""
numen — command-line interface for the Numen simulation framework.

Commands
--------
  numen init [dir]         Bootstrap a new project: CLAUDE.md + first model scaffold
  numen check              Verify installation: scipy, JAX, Julia backends
  numen list               List built-in examples with descriptions
  numen run <example>      Run a built-in example (scipy only, no plot window)
  numen new <name>         Scaffold a new model directory inside an existing project
  numen info               Print framework cheat-sheet
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

_INIT_CLAUDE_MD = '''\
# {project_name} — Numen Physics Simulation Project

This project uses the **Numen** framework (`pip install numen`) for
engineering dynamics simulation.  Models are defined in Python and
solved by scipy, JAX, or Julia backends.

Run `numen check` to verify your installation, then `numen info` for a
quick reference.

---

## Core pattern

```python
# 1. Components — data (state + parameters)
from numen.spec.component import Component
from numen.fields import IntegratedField, ParameterField
from typing import Annotated, Literal

class MyComponent(Component):
    kind:     Literal["my"] = "my"
    position: Annotated[float, IntegratedField()] = 0.0   # state (integrated)
    mass:     Annotated[float, ParameterField()]  = 1.0   # param (constant)

# 2. Systems — stateless dynamics functions
import jax.numpy as jnp   # always jnp, never np, inside dynamics
from numen.spec.system import System, DynamicsFn
from typing import ClassVar

def my_dynamics(dx, x, p, t, spec, system):
    for (eid,) in system.entity_groups:
        c  = spec.view(eid, MyComponent, x, p)     # read
        dc = spec.dx_view(eid, MyComponent, dx)    # write
        dc.position += c.velocity                  # accumulate with +=

class MySystem(System):
    component_types: ClassVar[tuple[type, ...]] = (MyComponent,)
    python_fn:       ClassVar[DynamicsFn]       = staticmethod(my_dynamics)
    kind:            Literal["my_sys"]          = "my_sys"
    dynamics_fn:     str = "MyDynamics.my_dynamics!"   # Julia function name

# 3. Assemble world + solve
from numen.spec.world import GenericWorld
from numen.compiler.flatten import compile_spec
from numen.bridge.scipy_backend import ScipyBackend

World  = GenericWorld[MyComponent, MySystem, None]
world  = World(components={{"e": MyComponent()}}, systems={{"s": MySystem()}})
spec   = compile_spec(world)
result = ScipyBackend(rtol=1e-8, atol=1e-10).solve(spec, tspan=(0.0, 1.0))
```

---

## JAX rules  ⚠️

Inside any dynamics function, always use `jnp.*` — never `np.*`.
Never use `if`/`else` on state values; use `jnp.where(cond, a, b)`.
Guard both branches of every `jnp.where` against NaN/Inf.
Use solver `Dopri5` (not `Tsit5`) when absolute tolerance is tight.

## Smooth contact  ⚠️

Hard-stop forces (`max(0,-pos)`) cause catastrophic step rejection.
Use a C1-smooth 1 µm ramp instead:

```python
_D = 1e-6   # 1 µm
def _soft_pen(x):
    return jnp.where(x <= 0, 0.0, jnp.where(x >= _D, x - 0.5*_D, 0.5*x*x/_D))

pen   = _soft_pen(-pos)                            # penetration at closed stop
alpha = jnp.clip(-pos / _D, 0.0, 1.0)             # contact activation (0→1)
F_stop = k_stop * pen + c_stop * jnp.maximum(0,-vel) * alpha
```

---

## Backends

| Backend | Warm speed | Use when |
|---|---|---|
| `ScipyBackend(rtol, atol)` | baseline | development, debugging |
| `JAXBackend(solver="Dopri5", max_steps=100_000)` | ~1500× faster | repeated solves, Monte Carlo, differentiable |
| `JuliaBackend(julia_file="dynamics.jl", method, rtol, atol)` | ~300–600× faster | long runs, stiff systems, one-off solves |
| `JuliaServerBackend(julia_file, method, rtol, atol)` | ~300–600× faster | parameter sweeps — pays JIT cost once |

JAX requires `jnp.*` dynamics (see rules above).
Julia backends require a `.jl` file that mirrors the Python dynamics (see below).

For stiff problems (multiple timescales, high-frequency oscillations), use an
implicit Julia solver: `method="Rodas5P"` or `method="FBDF"`.  These take far
larger steps than explicit methods (Tsit5, Vern7) on stiff systems.

---

## Parameter sweeps — JuliaServerBackend

`JuliaBackend` spawns a fresh Julia process per call (~6–12 s startup + JIT).
`JuliaServerBackend` keeps the process alive so every solve after the first is
a warm call (no recompilation).

```python
from numen.bridge.server_backend import JuliaServerBackend

# Start once — pays boot + JIT on first .solve() call
with JuliaServerBackend(
    julia_file="dynamics.jl",
    method="Rodas5P",   # implicit solver — best for stiff problems
    rtol=1e-6,
    atol=1e-8,
) as server:
    for params in parameter_grid:
        world  = make_world(params)          # rebuild with new parameters
        spec   = compile_spec(world)
        result = server.solve(spec, tspan=(0.0, 3600.0))
        # process result...
```

The server is safe to reuse with different specs and tspans.
Use it as a plain object (not a context manager) if the lifetime spans
multiple functions — just call `server.close()` when done.

```python
server = JuliaServerBackend(julia_file="dynamics.jl", eager=True)
# ... later ...
server.close()
```

`eager=True` starts the Julia process immediately (at construction) rather
than on the first `solve()` call, so startup happens at a predictable point.

---

## Multi-entity topology

When a system couples multiple entity types (spring between two masses,
orifice between two control volumes), declare slots in `entity_slots`
and provide the connections at instantiation:

```python
from numen.fields import EntityGroup

class SpringSystem(System):
    entity_slots: ClassVar[EntityGroup] = EntityGroup(
        MassComponent, SpringComponent, MassComponent   # group_size = 3
    )
    ...

SpringSystem(entity_groups=[["m1", "spring", "m2"], ["m2", "spring2", "m3"]])
```

---

## Accessing results

```python
from numen.reconstruction.collector import SnapshotCollector

collector = SnapshotCollector(world, spec, result)

# Time series for one field
t, pos = collector.field_series("entity_id", "position")

# Typed snapshot at a specific time
snap  = collector.at(t=1.5)
state = snap.components["entity_id"]   # read .position, .velocity, etc.
```

---

## Writing Julia dynamics

```julia
# dynamics.jl
module MyDynamics
import Main: CompiledSpec, CompiledSystemSpec, state_idx, param_idx

function my_dynamics!(dx, x, p, t, spec, sys)
    for id_e in sys.entity_ids
        i = state_idx(spec, id_e * ".position")
        dx[i] += x[state_idx(spec, id_e * ".velocity")]
    end
end

end  # module MyDynamics
```

Pass to the subprocess backend:
`JuliaBackend(julia_file="dynamics.jl", method="Tsit5", rtol=1e-8, atol=1e-10)`

Pass to the persistent server backend (parameter sweeps):
`JuliaServerBackend(julia_file="dynamics.jl", method="Rodas5P", rtol=1e-6, atol=1e-8)`

Available solvers: `Tsit5` (default, fast explicit), `Vern7` (higher-order explicit),
`Rodas5P` (stiff, implicit — best for multi-timescale problems), `FBDF` (stiff, implicit).

---

## Project layout

```
{project_name}/
{model_dirs}\\
├── CLAUDE.md          this file
└── (add more models with: numen new <name> --domain mechanical|fluid|generic)
```

---

## CLI reference

```bash
numen check                          # verify scipy + JAX + Julia
numen new <name> --domain <domain>   # scaffold a model (mechanical / fluid / generic)
numen info                           # framework cheat-sheet
```
'''


def cmd_init(args) -> int:
    target = Path(args.directory).resolve() if args.directory else Path.cwd()
    target.mkdir(parents=True, exist_ok=True)

    claude_md = target / "CLAUDE.md"
    if claude_md.exists() and not args.force:
        print(f"CLAUDE.md already exists in {target}.  Use --force to overwrite.")
        return 1

    # Optionally scaffold a first model
    model_dirs = ""
    if args.model:
        domain = args.domain or "generic"
        model_path = target / args.model
        if model_path.exists() and not args.force:
            print(f"Model directory '{model_path}' already exists.  Use --force to overwrite.")
            return 1
        model_path.mkdir(parents=True, exist_ok=True)
        model_name = args.model.replace("-", "_").replace(" ", "_").title().replace("_", "")
        tmpl = TEMPLATES.get(domain, TEMPLATES["generic"])
        for filename, content in tmpl.items():
            (model_path / filename).write_text(content.replace("{{MODEL_NAME}}", model_name))
        model_dirs = f"├── {args.model}/        ({domain} model)\n"
        print(f"  Scaffolded model: {model_path}/")

    project_name = target.name
    claude_content = _INIT_CLAUDE_MD.format(
        project_name=project_name,
        model_dirs=model_dirs,
    )
    claude_md.write_text(claude_content)
    print(f"  Created: {claude_md}")

    print()
    print(f"Project '{project_name}' initialized in {target}/")
    print()
    print("For a new Claude Code session on this project:")
    print("  1. Open this directory in Claude Code")
    print("  2. CLAUDE.md is loaded automatically — Claude knows the framework")
    print("  3. Run: numen check   — to verify the install")
    if args.model:
        print(f"  4. Run: cd {args.model} && python run.py")
    else:
        print(f"  4. Run: numen new <model_name> --domain mechanical|fluid|generic")
    return 0


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def cmd_check(args) -> int:
    """Smoke-test each backend.  Returns 0 if scipy passes, 1 if scipy fails."""
    print("Numen backend check")
    print("=" * 50)

    # --- minimal inline model (no file I/O) ---
    ok = True

    # scipy
    try:
        from numen._check_model import _CheckOsc, _CheckOscSys, CheckWorld
        from numen.compiler.flatten import compile_spec
        from numen.bridge.scipy_backend import ScipyBackend

        world  = CheckWorld(components={"o": _CheckOsc()}, systems={"s": _CheckOscSys()})
        spec   = compile_spec(world)
        result = ScipyBackend(rtol=1e-8, atol=1e-10).solve(spec, (0.0, 1.0))
        final  = result.x[spec.state_index_map["o.position"][0], -1]
        assert abs(final - 1.0) < 1e-4, f"wrong answer: {final}"
        print("  scipy   ✓  (RK45, oscillator x(1s) = {:.6f})".format(final))
    except Exception as e:
        print(f"  scipy   ✗  {e}")
        ok = False

    # JAX
    try:
        from numen._check_model import _CheckOsc, _CheckOscSysJax, CheckWorldJax
        from numen.compiler.flatten import compile_spec
        from numen.bridge.jax_backend import JAXBackend

        world2  = CheckWorldJax(components={"o": _CheckOsc()}, systems={"s": _CheckOscSysJax()})
        spec2   = compile_spec(world2)
        jax_b   = JAXBackend(rtol=1e-8, atol=1e-10, solver="Dopri5")
        result2 = jax_b.solve(spec2, (0.0, 1.0))
        final2  = float(result2.x[spec2.state_index_map["o.position"][0], -1])
        assert abs(final2 - 1.0) < 1e-3, f"wrong answer: {final2}"
        print("  JAX     ✓  (Dopri5, oscillator x(1s) = {:.6f})".format(final2))
    except ImportError:
        print("  JAX     –  not installed (pip install 'numen[jax]')")
    except Exception as e:
        print(f"  JAX     ✗  {e}")

    # Julia
    try:
        proc = subprocess.run(["julia", "--version"], capture_output=True, text=True, timeout=10)
        if proc.returncode == 0:
            print(f"  Julia   ✓  {proc.stdout.strip()}")
            print("              (use JuliaBackend to run full timing benchmark)")
        else:
            print("  Julia   –  'julia' binary not found in PATH")
    except FileNotFoundError:
        print("  Julia   –  'julia' binary not found in PATH")
    except Exception as e:
        print(f"  Julia   ✗  {e}")

    print()
    if ok:
        print("Core check passed.  Run 'numen run oscillator' to see a full example.")
    else:
        print("Core check FAILED.  Check your installation.")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

EXAMPLES = {
    "oscillator": {
        "description": "Minimal 1D damped harmonic oscillator — best starting point.",
        "concepts":    ["IntegratedField", "ParameterField", "ScipyBackend", "SnapshotCollector"],
        "domain":      "mechanical",
    },
    "coupled_spring": {
        "description": "Three masses coupled by two springs — multi-entity topology.",
        "concepts":    ["EntityGroup", "entity_groups", "multi-entity System", "energy conservation"],
        "domain":      "mechanical",
    },
    "fluid_poppet": {
        "description": "Pneumatic 4-CV network + spring-mass poppet check valve.",
        "concepts":    ["isentropic orifice flow", "smooth contact", "JAXBackend", "JuliaBackend"],
        "domain":      "fluid/mechanical",
    },
}


def cmd_list(args) -> int:
    print("Built-in examples")
    print("=" * 60)
    for name, meta in EXAMPLES.items():
        print(f"\n  {name}  [{meta['domain']}]")
        print(f"    {meta['description']}")
        print(f"    Concepts: {', '.join(meta['concepts'])}")
    print()
    print("Run:  numen run <name>          — execute with scipy (no plot window)")
    print("      numen new <name>          — scaffold a new model directory")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args) -> int:
    name = args.example
    if name not in EXAMPLES:
        print(f"Unknown example '{name}'.  Available: {', '.join(EXAMPLES)}")
        return 1

    examples_dir = Path(__file__).parent.parent.parent / "examples" / name
    if not examples_dir.exists():
        print(f"Example directory not found: {examples_dir}")
        print("(Examples are only available in the development checkout, not installed package.)")
        return 1

    run_py = examples_dir / "run.py"
    if not run_py.exists():
        print(f"No run.py in {examples_dir}")
        return 1

    print(f"Running example: {name}")
    print("-" * 40)
    env_patch = {"MPLBACKEND": "Agg"}   # suppress plot window
    import os
    env = {**os.environ, **env_patch}
    result = subprocess.run(
        [sys.executable, str(run_py)],
        cwd=str(examples_dir),
        env=env,
    )
    return result.returncode


# ---------------------------------------------------------------------------
# new  (scaffold)
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, dict[str, str]] = {
    "mechanical": {
        "components.py": '''\
from typing import Annotated, Literal
from numen.spec.component import Component
from numen.fields import IntegratedField, ParameterField


class BodyComponent(Component):
    """Rigid body with 1D position/velocity state."""
    kind:     Literal["body"] = "body"
    position: Annotated[float, IntegratedField()] = 0.0   # m
    velocity: Annotated[float, IntegratedField()] = 0.0   # m/s
    mass:     Annotated[float, ParameterField()]  = 1.0   # kg


class SpringComponent(Component):
    """Linear spring — topology declared in SpringSystem.entity_groups."""
    kind:        Literal["spring"] = "spring"
    stiffness:   Annotated[float, ParameterField()] = 100.0  # N/m
    rest_length: Annotated[float, ParameterField()] = 1.0    # m
    damping:     Annotated[float, ParameterField()] = 1.0    # N·s/m
''',
        "dynamics.py": '''\
from typing import ClassVar, Literal
import jax.numpy as jnp
from numen.compiler.flatten import CompiledSpec, CompiledSystem
from numen.fields import EntityGroup
from numen.spec.system import System, DynamicsFn
from components import BodyComponent, SpringComponent


def kinematics_dynamics(dx, x, p, t, spec, system):
    """ẋ = v for all BodyComponent entities."""
    for (eid,) in system.entity_groups:
        c  = spec.view(eid, BodyComponent, x, p)
        dc = spec.dx_view(eid, BodyComponent, dx)
        dc.position += c.velocity


class KinematicsSystem(System):
    component_types: ClassVar[tuple[type, ...]] = (BodyComponent,)
    python_fn:       ClassVar[DynamicsFn]       = staticmethod(kinematics_dynamics)
    kind:            Literal["kinematics"]      = "kinematics"
    dynamics_fn:     str = "{{MODEL_NAME}}Dynamics.kinematics_dynamics!"


def spring_dynamics(dx, x, p, t, spec, system):
    """Spring-damper force between body_a and body_b via a spring entity.
    Entity group: [body_a, spring, body_b]
    """
    for id_a, id_s, id_b in system.entity_groups:
        a = spec.view(id_a, BodyComponent,   x, p)
        b = spec.view(id_b, BodyComponent,   x, p)
        s = spec.view(id_s, SpringComponent, x, p)
        da = spec.dx_view(id_a, BodyComponent, dx)
        db = spec.dx_view(id_b, BodyComponent, dx)

        stretch  = (b.position - a.position) - s.rest_length
        rel_vel  = b.velocity  - a.velocity
        force    = s.stiffness * stretch + s.damping * rel_vel

        da.velocity +=  force / a.mass
        db.velocity += -force / b.mass


class SpringSystem(System):
    component_types: ClassVar[tuple[type, ...]] = ()
    entity_slots:    ClassVar[EntityGroup]      = EntityGroup(
        BodyComponent, SpringComponent, BodyComponent
    )
    python_fn:  ClassVar[DynamicsFn] = staticmethod(spring_dynamics)
    kind:       Literal["spring"]   = "spring"
    dynamics_fn: str = "{{MODEL_NAME}}Dynamics.spring_dynamics!"
''',
        "dynamics.jl": '''\
module {{MODEL_NAME}}Dynamics

import Main: CompiledSpec, CompiledSystemSpec, state_idx, param_idx


function kinematics_dynamics!(
    dx::Vector{Float64}, x::Vector{Float64}, p::Vector{Float64},
    t::Float64, spec::CompiledSpec, sys::CompiledSystemSpec,
)
    for id_body in sys.entity_ids
        i_pos = state_idx(spec, id_body * ".position")
        i_vel = state_idx(spec, id_body * ".velocity")
        dx[i_pos] += x[i_vel]
    end
end


function spring_dynamics!(
    dx::Vector{Float64}, x::Vector{Float64}, p::Vector{Float64},
    t::Float64, spec::CompiledSpec, sys::CompiledSystemSpec,
)
    gs = sys.group_size  # 3
    for i in 1:gs:length(sys.entity_ids)
        id_a = sys.entity_ids[i]
        id_s = sys.entity_ids[i + 1]
        id_b = sys.entity_ids[i + 2]

        pos_a = x[state_idx(spec, id_a * ".position")]
        pos_b = x[state_idx(spec, id_b * ".position")]
        vel_a = x[state_idx(spec, id_a * ".velocity")]
        vel_b = x[state_idx(spec, id_b * ".velocity")]
        mass_a     = p[param_idx(spec, id_a * ".mass")]
        mass_b     = p[param_idx(spec, id_b * ".mass")]
        stiffness  = p[param_idx(spec, id_s * ".stiffness")]
        rest_len   = p[param_idx(spec, id_s * ".rest_length")]
        damping    = p[param_idx(spec, id_s * ".damping")]

        stretch = (pos_b - pos_a) - rest_len
        rel_vel = vel_b - vel_a
        force   = stiffness * stretch + damping * rel_vel

        dx[state_idx(spec, id_a * ".velocity")] +=  force / mass_a
        dx[state_idx(spec, id_b * ".velocity")] += -force / mass_b
    end
end


end  # module {{MODEL_NAME}}Dynamics
''',
        "world.py": '''\
from typing import Annotated, Union
from pydantic import Field
from numen.spec.world import GenericWorld
from components import BodyComponent, SpringComponent
from dynamics import KinematicsSystem, SpringSystem

AnyComponent = Annotated[Union[BodyComponent, SpringComponent], Field(discriminator="kind")]
AnySystem    = Annotated[Union[KinematicsSystem, SpringSystem],  Field(discriminator="kind")]
World        = GenericWorld[AnyComponent, AnySystem, None]


def make_world() -> World:
    """Two bodies connected by a spring-damper.  body_b starts displaced by 0.5 m."""
    return World(
        components={
            "body_a": BodyComponent(position=0.0, velocity=0.0, mass=1.0),
            "spring": SpringComponent(stiffness=100.0, rest_length=1.0, damping=1.0),
            "body_b": BodyComponent(position=1.5, velocity=0.0, mass=1.0),
        },
        systems={
            "kinematics": KinematicsSystem(),
            "spring":     SpringSystem(entity_groups=[["body_a", "spring", "body_b"]]),
        },
    )
''',
        "run.py": '''\
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

from numen.compiler.flatten import compile_spec
from numen.bridge.scipy_backend import ScipyBackend
from numen.reconstruction.collector import SnapshotCollector
from world import make_world


def run():
    world  = make_world()
    spec   = compile_spec(world)

    print("State fields:", list(spec.state_index_map.keys()))
    print("Param fields:", list(spec.param_index_map.keys()))

    tspan  = (0.0, 10.0)
    result = ScipyBackend(rtol=1e-8, atol=1e-10).solve(spec, tspan)
    print(f"Solved: {len(result.t)} steps over {result.t[-1]:.1f} s")

    collector = SnapshotCollector(world, spec, result)
    t, pos_a = collector.field_series("body_a", "position")
    _, pos_b  = collector.field_series("body_b", "position")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, pos_a, label="body_a")
    ax.plot(t, pos_b, label="body_b")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.set_title("Spring-Damper System")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "result.png")
    plt.savefig(out, dpi=150)
    print(f"Plot saved to {out}")
    plt.show()


if __name__ == "__main__":
    run()
''',
    },
    "fluid": {
        "components.py": '''\
from typing import Annotated, Literal
from numen.spec.component import Component
from numen.fields import IntegratedField, ParameterField


class ControlVolumeComponent(Component):
    """Isothermal ideal-gas control volume.  dP/dt = (R·T/V)·ṁ_net"""
    kind:        Literal["control_volume"] = "control_volume"
    pressure:    Annotated[float, IntegratedField()] = 101_325.0   # Pa
    volume:      Annotated[float, ParameterField()]  = 1e-3        # m³
    temperature: Annotated[float, ParameterField()]  = 293.15      # K
    R_specific:  Annotated[float, ParameterField()]  = 287.058     # J/(kg·K)  — air


class OrificeComponent(Component):
    """Fixed-area isentropic orifice."""
    kind:  Literal["orifice"] = "orifice"
    Cd:    Annotated[float, ParameterField()] = 0.7
    area:  Annotated[float, ParameterField()] = 1e-5   # m²
    gamma: Annotated[float, ParameterField()] = 1.4
''',
        "dynamics.py": '''\
from typing import ClassVar, Literal
import jax.numpy as jnp
from numen.compiler.flatten import CompiledSpec, CompiledSystem
from numen.fields import EntityGroup
from numen.spec.system import System, DynamicsFn
from components import ControlVolumeComponent, OrificeComponent


def _orifice_mdot(P_up, P_dn, T_up, R, Cd, A, gamma):
    """Isentropic compressible mass flow (kg/s, always >= 0).  JAX-compatible."""
    safe_P_up = jnp.maximum(P_up, 1e-300)
    beta      = jnp.maximum(0.0, P_dn) / safe_P_up
    beta_crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    choke_exp     = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    mdot_choked   = Cd * A * P_up * jnp.sqrt(gamma / (R * T_up)) * (2.0/(gamma+1.0))**choke_exp
    arg           = beta**(2.0/gamma) - beta**((gamma+1.0)/gamma)
    mdot_unchoked = Cd * A * P_up * jnp.sqrt(jnp.maximum(0.0, 2*gamma/((gamma-1)*R*T_up)*arg))
    mdot = jnp.where(beta <= beta_crit, mdot_choked, mdot_unchoked)
    return jnp.where((P_up <= 0.0) | (A <= 0.0), 0.0, mdot)


def orifice_flow_dynamics(dx, x, p, t, spec, system):
    """Compressible orifice flow between two control volumes.
    Entity group: [cv_a, orifice, cv_b] — flow direction determined by pressure.
    """
    for id_a, id_o, id_b in system.entity_groups:
        cv_a    = spec.view(id_a, ControlVolumeComponent, x, p)
        orifice = spec.view(id_o, OrificeComponent,       x, p)
        cv_b    = spec.view(id_b, ControlVolumeComponent, x, p)

        P_a, P_b = cv_a.pressure, cv_b.pressure
        a_is_up  = P_a >= P_b
        P_up = jnp.where(a_is_up, P_a, P_b)
        P_dn = jnp.where(a_is_up, P_b, P_a)
        T_up = jnp.where(a_is_up, cv_a.temperature, cv_b.temperature)
        R_up = jnp.where(a_is_up, cv_a.R_specific,  cv_b.R_specific)
        sign = jnp.where(a_is_up, 1.0, -1.0)

        mdot = sign * _orifice_mdot(P_up, P_dn, T_up, R_up,
                                    orifice.Cd, orifice.area, orifice.gamma)

        da = spec.dx_view(id_a, ControlVolumeComponent, dx)
        db = spec.dx_view(id_b, ControlVolumeComponent, dx)
        da.pressure += -(cv_a.R_specific * cv_a.temperature / cv_a.volume) * mdot
        db.pressure +=  (cv_b.R_specific * cv_b.temperature / cv_b.volume) * mdot


class OrificeFlowSystem(System):
    component_types: ClassVar[tuple[type, ...]] = ()
    entity_slots:    ClassVar[EntityGroup]      = EntityGroup(
        ControlVolumeComponent, OrificeComponent, ControlVolumeComponent
    )
    python_fn:  ClassVar[DynamicsFn] = staticmethod(orifice_flow_dynamics)
    kind:       Literal["orifice_flow"] = "orifice_flow"
    dynamics_fn: str = "{{MODEL_NAME}}Dynamics.orifice_flow_dynamics!"
''',
        "dynamics.jl": '''\
module {{MODEL_NAME}}Dynamics

import Main: CompiledSpec, CompiledSystemSpec, state_idx, param_idx


function orifice_mdot(
    P_up::Float64, P_dn::Float64, T_up::Float64,
    R::Float64, Cd::Float64, A::Float64, gamma::Float64,
)::Float64
    (P_up <= 0.0 || A <= 0.0) && return 0.0
    beta      = max(0.0, P_dn) / P_up
    beta_crit = (2.0 / (gamma + 1.0))^(gamma / (gamma - 1.0))
    if beta <= beta_crit
        choke_exp = (gamma + 1.0) / (2.0 * (gamma - 1.0))
        return Cd * A * P_up * sqrt(gamma / (R * T_up)) * (2.0/(gamma+1.0))^choke_exp
    else
        arg = beta^(2.0/gamma) - beta^((gamma+1.0)/gamma)
        return Cd * A * P_up * sqrt(max(0.0, 2.0*gamma/((gamma-1.0)*R*T_up)*arg))
    end
end


function orifice_flow_dynamics!(
    dx::Vector{Float64}, x::Vector{Float64}, p::Vector{Float64},
    t::Float64, spec::CompiledSpec, sys::CompiledSystemSpec,
)
    gs = sys.group_size
    for i in 1:gs:length(sys.entity_ids)
        id_a = sys.entity_ids[i]
        id_o = sys.entity_ids[i + 1]
        id_b = sys.entity_ids[i + 2]

        P_a = x[state_idx(spec, id_a * ".pressure")]
        P_b = x[state_idx(spec, id_b * ".pressure")]
        T_a = p[param_idx(spec, id_a * ".temperature")]
        T_b = p[param_idx(spec, id_b * ".temperature")]
        R_a = p[param_idx(spec, id_a * ".R_specific")]
        R_b = p[param_idx(spec, id_b * ".R_specific")]
        V_a = p[param_idx(spec, id_a * ".volume")]
        V_b = p[param_idx(spec, id_b * ".volume")]
        Cd  = p[param_idx(spec, id_o * ".Cd")]
        A   = p[param_idx(spec, id_o * ".area")]
        gam = p[param_idx(spec, id_o * ".gamma")]

        if P_a >= P_b
            mdot = orifice_mdot(P_a, P_b, T_a, R_a, Cd, A, gam)
        else
            mdot = -orifice_mdot(P_b, P_a, T_b, R_b, Cd, A, gam)
        end

        dx[state_idx(spec, id_a * ".pressure")] += -(R_a * T_a / V_a) * mdot
        dx[state_idx(spec, id_b * ".pressure")] +=  (R_b * T_b / V_b) * mdot
    end
end


end  # module {{MODEL_NAME}}Dynamics
''',
        "world.py": '''\
from typing import Annotated, Union
from pydantic import Field
from numen.spec.world import GenericWorld
from components import ControlVolumeComponent, OrificeComponent
from dynamics import OrificeFlowSystem

AnyComponent = Annotated[Union[ControlVolumeComponent, OrificeComponent], Field(discriminator="kind")]
AnySystem    = Annotated[Union[OrificeFlowSystem], Field(discriminator="kind")]
World        = GenericWorld[AnyComponent, AnySystem, None]

P_HIGH = 3e5   # Pa — 3 bar
P_LOW  = 1e5   # Pa — 1 bar  (ambient)
T      = 293.15  # K


def make_world() -> World:
    """Two tanks connected by an orifice.  Inlet at 3 bar, outlet at 1 bar."""
    return World(
        components={
            "inlet":  ControlVolumeComponent(pressure=P_HIGH, volume=1e-2, temperature=T),
            "orifice": OrificeComponent(Cd=0.7, area=1e-5, gamma=1.4),
            "outlet": ControlVolumeComponent(pressure=P_LOW,  volume=1e-2, temperature=T),
        },
        systems={
            "flow": OrificeFlowSystem(entity_groups=[["inlet", "orifice", "outlet"]]),
        },
    )
''',
        "run.py": '''\
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

from numen.compiler.flatten import compile_spec
from numen.bridge.scipy_backend import ScipyBackend
from numen.reconstruction.collector import SnapshotCollector
from world import make_world


def run():
    world  = make_world()
    spec   = compile_spec(world)

    print("State fields:", list(spec.state_index_map.keys()))

    tspan  = (0.0, 1.0)
    result = ScipyBackend(rtol=1e-8, atol=1e-10).solve(spec, tspan)
    print(f"Solved: {len(result.t)} steps over {result.t[-1]:.2f} s")

    collector = SnapshotCollector(world, spec, result)
    t, P_in  = collector.field_series("inlet",  "pressure")
    _, P_out  = collector.field_series("outlet", "pressure")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, P_in  / 1e5, label="inlet (bar)")
    ax.plot(t, P_out / 1e5, label="outlet (bar)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (bar)")
    ax.set_title("Two-Tank Orifice Flow")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "result.png")
    plt.savefig(out, dpi=150)
    print(f"Plot saved to {out}")
    plt.show()


if __name__ == "__main__":
    run()
''',
    },
}

# Generic (blank) template — same structure but minimal content
TEMPLATES["generic"] = {
    "components.py": '''\
from typing import Annotated, Literal
from numen.spec.component import Component
from numen.fields import IntegratedField, ParameterField


class MyComponent(Component):
    kind:  Literal["my"] = "my"
    # TODO: add IntegratedField (state) and ParameterField (params)
    state: Annotated[float, IntegratedField()] = 0.0
''',
    "dynamics.py": '''\
from typing import ClassVar, Literal
import jax.numpy as jnp                    # always jnp, never np, inside dynamics
from numen.compiler.flatten import CompiledSpec, CompiledSystem
from numen.spec.system import System, DynamicsFn
from components import MyComponent


def my_dynamics(dx, x, p, t, spec, system):
    for (eid,) in system.entity_groups:
        c  = spec.view(eid, MyComponent, x, p)
        dc = spec.dx_view(eid, MyComponent, dx)
        # TODO: dc.state += ...


class MySystem(System):
    component_types: ClassVar[tuple[type, ...]] = (MyComponent,)
    python_fn:       ClassVar[DynamicsFn]       = staticmethod(my_dynamics)
    kind:            Literal["my_system"]       = "my_system"
    dynamics_fn:     str = "{{MODEL_NAME}}Dynamics.my_dynamics!"
''',
    "dynamics.jl": '''\
module {{MODEL_NAME}}Dynamics

import Main: CompiledSpec, CompiledSystemSpec, state_idx, param_idx


function my_dynamics!(
    dx::Vector{Float64}, x::Vector{Float64}, p::Vector{Float64},
    t::Float64, spec::CompiledSpec, sys::CompiledSystemSpec,
)
    for id_e in sys.entity_ids
        i_state = state_idx(spec, id_e * ".state")
        # TODO: dx[i_state] += ...
    end
end


end  # module {{MODEL_NAME}}Dynamics
''',
    "world.py": '''\
from typing import Annotated, Union
from pydantic import Field
from numen.spec.world import GenericWorld
from components import MyComponent
from dynamics import MySystem

AnyComponent = Annotated[Union[MyComponent], Field(discriminator="kind")]
AnySystem    = Annotated[Union[MySystem],    Field(discriminator="kind")]
World        = GenericWorld[AnyComponent, AnySystem, None]


def make_world() -> World:
    return World(
        components={"entity": MyComponent()},
        systems={"system": MySystem()},
    )
''',
    "run.py": '''\
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from numen.compiler.flatten import compile_spec
from numen.bridge.scipy_backend import ScipyBackend
from world import make_world


def run():
    world  = make_world()
    spec   = compile_spec(world)
    result = ScipyBackend(rtol=1e-8, atol=1e-10).solve(spec, tspan=(0.0, 1.0))
    print(f"Solved: {len(result.t)} steps")


if __name__ == "__main__":
    run()
''',
}


def cmd_new(args) -> int:
    name   = args.name
    domain = args.domain
    outdir = Path(args.output) / name if args.output else Path(name)

    if domain not in TEMPLATES:
        print(f"Unknown domain '{domain}'.  Choose from: {', '.join(TEMPLATES)}")
        return 1

    if outdir.exists() and not args.force:
        print(f"Directory '{outdir}' already exists.  Use --force to overwrite.")
        return 1

    outdir.mkdir(parents=True, exist_ok=True)
    model_name = name.replace("-", "_").replace(" ", "_").title().replace("_", "")

    tmpl = TEMPLATES[domain]
    for filename, content in tmpl.items():
        text = content.replace("{{MODEL_NAME}}", model_name)
        (outdir / filename).write_text(text)

    print(f"Scaffolded '{name}' ({domain}) in {outdir}/")
    print()
    print("Files created:")
    for f in sorted(outdir.iterdir()):
        print(f"  {f.name}")
    print()
    print("Next steps:")
    print(f"  1. Edit {outdir}/components.py  — define your state and parameter fields")
    print(f"  2. Edit {outdir}/dynamics.py    — write physics (use jnp.*, not np.*)")
    print(f"  3. Edit {outdir}/world.py       — set initial conditions")
    print(f"  4. Run:  cd {outdir} && uv run python run.py")
    print()
    print("Julia backend (optional, ~600x faster):")
    print(f"  5. Edit {outdir}/dynamics.jl    — mirror the Python dynamics")
    print(f"  6. Pass dynamics.jl to JuliaBackend(julia_file=...)")
    return 0


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

def cmd_info(args) -> int:
    text = textwrap.dedent("""
    Numen — Physics Simulation Framework
    =====================================

    CORE PATTERN
      Component  — data bag: IntegratedField (state x) + ParameterField (param p)
      System     — stateless: reads x,p via spec.view(), writes dx via spec.dx_view()
      World      — dict of components + dict of systems
      compile_spec(world) → CompiledSpec  (flat x0, p, index maps)

    BACKENDS
      ScipyBackend(rtol, atol)                  — RK45, good for development
      JAXBackend(solver="Dopri5", max_steps=…)  — ~1500× faster warm, needs jnp.*
      JuliaBackend(julia_file=…, rtol, atol)    — ~600× faster warm, needs .jl file

    JAX RULES  (or get TracerBoolConversionError)
      • import jax.numpy as jnp  — never np.* inside dynamics
      • no if/else on state values — use jnp.where(cond, a, b)
      • guard sqrt/log arguments: jnp.sqrt(jnp.maximum(0, x))
      • use Dopri5, not Tsit5, when atol << rtol × |state|

    SMOOTH CONTACT (hard stops / collision)
      Sharp max(0,-pos) kink → 99% step rejection → use _soft_pen() instead.
      See examples/fluid_poppet/dynamics.py for the 1 µm ramp pattern.

    ACCESS RESULTS
      result.t                          → time array
      result.x[spec.state_index_map["entity.field"][0]]  → field time series
      SnapshotCollector(world,spec,result).field_series("entity","field")

    COMMANDS
      numen check             verify backends
      numen list              list examples
      numen run <example>     run example (no plot window)
      numen new <name> --domain mechanical|fluid|generic
      numen info              this screen

    FILES
      CLAUDE.md               full guide for Claude Code
      DESIGN.md               architecture decisions and open questions
      examples/fluid_poppet/  most complete reference example
    """)
    print(text)
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="numen",
        description="Numen physics simulation framework CLI",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    p_init = sub.add_parser("init", help="Bootstrap a new project with CLAUDE.md")
    p_init.add_argument("directory", nargs="?", default=None,
                        help="Target directory (default: current directory)")
    p_init.add_argument("--model", default=None,
                        help="Also scaffold a first model with this name")
    p_init.add_argument("--domain", choices=list(TEMPLATES), default=None,
                        help="Domain for --model scaffold (default: generic)")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing files")

    sub.add_parser("check", help="Verify scipy / JAX / Julia backends")

    sub.add_parser("list", help="List built-in examples")

    p_run = sub.add_parser("run", help="Run a built-in example")
    p_run.add_argument("example", choices=list(EXAMPLES))

    p_new = sub.add_parser("new", help="Scaffold a new model directory")
    p_new.add_argument("name", help="Model name (becomes directory name)")
    p_new.add_argument(
        "--domain", choices=list(TEMPLATES), default="generic",
        help="Starting template (default: generic)",
    )
    p_new.add_argument("--output", default=None, help="Parent directory (default: cwd)")
    p_new.add_argument("--force", action="store_true", help="Overwrite existing directory")

    sub.add_parser("info", help="Print framework cheat-sheet")

    args = parser.parse_args()

    dispatch = {
        "init":  cmd_init,
        "check": cmd_check,
        "list":  cmd_list,
        "run":   cmd_run,
        "new":   cmd_new,
        "info":  cmd_info,
    }

    if args.command is None:
        parser.print_help()
        return 0

    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
