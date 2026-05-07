"""Numen CLI."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from numen._scaffold import EXAMPLES, TEMPLATES

# Ensure UTF-8 output on Windows (cmd.exe / PowerShell default to CP1252)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# ─── App ──────────────────────────────────────────────────────────────────────
app = typer.Typer(
    name="numen",
    no_args_is_help=False,
    rich_markup_mode="rich",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
console = Console()

try:
    from importlib.metadata import version as _pkg_version
    _VERSION = _pkg_version("numen")
except Exception:
    _VERSION = "0.1.0"

# ─── Logo ─────────────────────────────────────────────────────────────────────
_LOGO_LINES = [
    " ██╗   ██╗██╗   ██╗███╗   ███╗███████╗███╗   ██╗",
    " ████╗  ██║██║   ██║████╗ ████║██╔════╝████╗  ██║",
    " ██╔██╗ ██║██║   ██║██╔████╔██║█████╗  ██╔██╗ ██║",
    " ██║╚██╗██║██║   ██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║",
    " ██║ ╚████║╚██████╔╝██║ ╚═╝ ██║███████╗██║ ╚████║",
    " ╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝",
]
_SHADES = ["#a855f7", "#9333ea", "#7c3aed", "#6d28d9", "#7c3aed", "#9333ea"]


def _logo_panel() -> Panel:
    t = Text(justify="center")
    for line, shade in zip(_LOGO_LINES, _SHADES):
        t.append(line + "\n", style=f"bold {shade}")
    t.append("\n")
    t.append("  physics simulation framework", style="dim white")
    t.append("  ·  ", style="dim #7c3aed")
    t.append(f"v{_VERSION}\n", style="dim")
    t.append("  Python", style="dim #3b82f6")
    t.append("  ·  ", style="dim")
    t.append("JAX", style="dim #10b981")
    t.append("  ·  ", style="dim")
    t.append("Julia", style="dim #f59e0b")
    return Panel(Align.center(t), border_style="#7c3aed", padding=(1, 4))


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _ok(msg: str) -> None:
    console.print(f"  [bold #10b981]✓[/]  {msg}")

def _fail(msg: str) -> None:
    console.print(f"  [bold #ef4444]✗[/]  {msg}")

def _warn(msg: str) -> None:
    console.print(f"  [bold #f59e0b]⚠[/]  {msg}")

def _step(n: int, msg: str) -> None:
    console.print(f"  [dim]{n}.[/]  {msg}")

def _file(name: str, desc: str) -> None:
    console.print(f"  [bold #10b981]✓[/]  [bold]{name:<22}[/] [dim]{desc}[/]")

def _header(title: str) -> None:
    console.print()
    console.print(Rule(f"[bold #06b6d4]{title}[/]", style="dim #7c3aed"))
    console.print()


# ─── Commands ─────────────────────────────────────────────────────────────────

@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        console.print(_logo_panel())
        console.print()
        # print a pretty commands table
        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        t.add_column(style="bold #06b6d4", no_wrap=True)
        t.add_column(style="dim")
        t.add_row("check",  "Verify scipy, JAX, and Julia backends")
        t.add_row("init",   "Bootstrap a new project with CLAUDE.md")
        t.add_row("new",    "Scaffold a model directory")
        t.add_row("list",   "Show built-in examples")
        t.add_row("run",    "Run a built-in example")
        t.add_row("info",   "Quick-reference cheat sheet")
        console.print(t)
        console.print()
        console.print(f"  [dim]Run [bold]numen <command> --help[/bold] for details.[/dim]")
        console.print()


@app.command()
def check() -> None:
    """Verify scipy, JAX, and Julia backends."""
    console.print(_logo_panel())
    _header("Backend Check")
    ok = True

    # scipy
    with console.status("[dim]checking scipy...[/dim]", spinner="dots"):
        try:
            from numen._check_model import _CheckOsc, _CheckOscSys, CheckWorld
            from numen.compiler.flatten import compile_spec
            from numen.bridge.scipy_backend import ScipyBackend
            world = CheckWorld(components={"o": _CheckOsc()}, systems={"s": _CheckOscSys()})
            spec = compile_spec(world)
            result = ScipyBackend(rtol=1e-8, atol=1e-10).solve(spec, (0.0, 1.0))
            final = result.x[spec.state_index_map["o.position"][0], -1]
            assert abs(final - 1.0) < 1e-4, f"wrong: {final}"
            scipy_ok = True
        except Exception as e:
            scipy_ok = False
            scipy_err = str(e).split("\n")[0][:100]
    if scipy_ok:
        _ok(f"[bold]scipy[/bold]   RK45 · oscillator x(1s) = {final:.6f}")
    else:
        _fail(f"[bold]scipy[/bold]   {scipy_err}")
        ok = False

    # JAX
    with console.status("[dim]checking JAX...[/dim]", spinner="dots"):
        try:
            from numen._check_model import _CheckOsc, _CheckOscSysJax, CheckWorldJax
            from numen.compiler.flatten import compile_spec
            from numen.bridge.jax_backend import JAXBackend
            world2 = CheckWorldJax(components={"o": _CheckOsc()}, systems={"s": _CheckOscSysJax()})
            spec2 = compile_spec(world2)
            result2 = JAXBackend(rtol=1e-8, atol=1e-10, solver="Dopri5").solve(spec2, (0.0, 1.0))
            final2 = float(result2.x[spec2.state_index_map["o.position"][0], -1])
            assert abs(final2 - 1.0) < 1e-3, f"wrong: {final2}"
            jax_ok = True
        except ImportError:
            jax_ok = None
        except Exception as e:
            jax_ok = False
            jax_err = str(e).split("\n")[0][:100]
    if jax_ok is True:
        _ok(f"[bold]JAX[/bold]     Dopri5 · oscillator x(1s) = {final2:.6f}")
    elif jax_ok is None:
        _warn(f"[bold]JAX[/bold]     not installed  [dim](pip install 'numen[jax]')[/dim]")
    else:
        _fail(f"[bold]JAX[/bold]     {jax_err}")

    # Julia
    with console.status("[dim]checking Julia...[/dim]", spinner="dots"):
        try:
            proc = subprocess.run(["julia", "--version"], capture_output=True, text=True, timeout=10)
            julia_ok = proc.returncode == 0
            julia_ver = proc.stdout.strip()
        except FileNotFoundError:
            julia_ok = False
            julia_ver = "'julia' not found in PATH"
        except Exception as e:
            julia_ok = False
            julia_ver = str(e)[:80]
    if julia_ok:
        _ok(f"[bold]Julia[/bold]   {julia_ver}")
    else:
        _warn(f"[bold]Julia[/bold]   {julia_ver}  [dim](install from julialang.org)[/dim]")

    console.print()
    console.print(Rule(style="dim #7c3aed"))
    console.print()
    if ok:
        console.print("  [bold #10b981]All checks passed.[/bold #10b981]  Run [bold]numen run oscillator[/bold] to see a full example.")
    else:
        console.print("  [bold #ef4444]Check failed.[/bold #ef4444]  Verify your installation and try again.")
    console.print()
    if not ok:
        raise typer.Exit(code=1)


@app.command()
def init(
    directory: Optional[str] = typer.Argument(None, help="Target directory (default: cwd)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Also scaffold a first model"),
    domain: str = typer.Option("generic", "--domain", "-d", help="Domain: mechanical | fluid | generic"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
) -> None:
    """Bootstrap a new project: writes CLAUDE.md and optionally scaffolds a first model."""
    target = Path(directory).resolve() if directory else Path.cwd()
    target.mkdir(parents=True, exist_ok=True)

    claude_md = target / "CLAUDE.md"
    if claude_md.exists() and not force:
        console.print(f"  [bold #f59e0b]⚠[/]  CLAUDE.md already exists. Use [bold]--force[/bold] to overwrite.")
        raise typer.Exit(code=1)

    model_dirs = ""
    model_path = None
    if model:
        if domain not in TEMPLATES:
            console.print(f"  [bold #ef4444]✗[/]  Unknown domain [bold]{domain!r}[/]. Choose: {', '.join(TEMPLATES)}")
            raise typer.Exit(code=1)
        model_path = target / model
        if model_path.exists() and not force:
            console.print(f"  [bold #f59e0b]⚠[/]  Directory [bold]{model_path}[/bold] already exists. Use [bold]--force[/bold].")
            raise typer.Exit(code=1)
        model_path.mkdir(parents=True, exist_ok=True)
        model_name = model.replace("-", "_").replace(" ", "_").title().replace("_", "")
        tmpl = TEMPLATES.get(domain, TEMPLATES["generic"])
        for filename, content in tmpl.items():
            (model_path / filename).write_text(content.replace("{{MODEL_NAME}}", model_name), encoding="utf-8")
        model_dirs = f"├── {model}/        ({domain} model)\n"

    project_name = target.name
    claude_content = _INIT_CLAUDE_MD.format(project_name=project_name, model_dirs=model_dirs)
    claude_md.write_text(claude_content, encoding="utf-8")

    console.print(_logo_panel())
    _header("Project Initialized")
    console.print(f"  [bold]Project:[/bold]  {project_name}")
    console.print(f"  [bold]Location:[/bold] {target}")
    console.print()
    console.print("  [dim]Created:[/dim]")
    _file("CLAUDE.md", "AI assistant context — loaded automatically by Claude Code")
    if model_path:
        _file(f"{model}/", f"{domain} model scaffold")
    console.print()
    console.print("  [dim]Next steps:[/dim]")
    _step(1, "Open this directory in [bold]Claude Code[/bold]")
    _step(2, "Run: [bold]numen check[/bold]")
    if model:
        _step(3, f"Run: [bold]cd {model} && python run.py[/bold]")
    else:
        _step(3, "Run: [bold]numen new <model_name> --domain mechanical|fluid|generic[/bold]")
    console.print()


@app.command()
def new(
    name: str = typer.Argument(..., help="Model name (becomes directory name)"),
    domain: str = typer.Option("generic", "--domain", "-d", help="Template: mechanical | fluid | generic"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Parent directory (default: cwd)"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite if exists"),
) -> None:
    """Scaffold a new model directory."""
    if domain not in TEMPLATES:
        console.print(f"  [bold #ef4444]✗[/]  Unknown domain [bold]{domain!r}[/]. Choose: {', '.join(TEMPLATES)}")
        raise typer.Exit(code=1)

    outdir = Path(output) / name if output else Path(name)
    if outdir.exists() and not force:
        console.print(f"  [bold #f59e0b]⚠[/]  Directory [bold]{outdir}[/] already exists. Use [bold]--force[/bold].")
        raise typer.Exit(code=1)

    outdir.mkdir(parents=True, exist_ok=True)
    model_name = name.replace("-", "_").replace(" ", "_").title().replace("_", "")
    tmpl = TEMPLATES[domain]
    for filename, content in tmpl.items():
        (outdir / filename).write_text(content.replace("{{MODEL_NAME}}", model_name), encoding="utf-8")

    _header(f"Scaffolded: {name}  [{domain}]")
    _file("components.py", "Component classes (IntegratedField, ParameterField)")
    _file("dynamics.py",   "Physics functions — JAX-compatible, use jnp.*")
    _file("dynamics.jl",   "Julia mirror for JuliaBackend / JuliaServerBackend")
    _file("world.py",      "World assembly and make_world()")
    _file("run.py",        "Solve and plot")
    console.print()
    console.print("  [dim]Next steps:[/dim]")
    _step(1, f"Edit [bold]{outdir}/components.py[/bold]  — define state and parameter fields")
    _step(2, f"Edit [bold]{outdir}/dynamics.py[/bold]    — write physics (use [bold]jnp.*[/bold], not np.*)")
    _step(3, f"Edit [bold]{outdir}/world.py[/bold]       — set initial conditions")
    _step(4, f"Run:  [bold]cd {outdir} && python run.py[/bold]")
    console.print()
    console.print("  [dim]Julia backend (optional, ~300–600× faster):[/dim]")
    _step(5, f"Edit [bold]{outdir}/dynamics.jl[/bold]    — mirror Python dynamics in Julia")
    _step(6, "Use [bold]JuliaServerBackend[/bold] or [bold]JuliaBackend[/bold] in run.py")
    console.print()


@app.command("list")
def list_examples() -> None:
    """Show built-in examples."""
    _header("Built-in Examples")
    t = Table(box=box.ROUNDED, border_style="dim #7c3aed", show_header=True, padding=(0, 2))
    t.add_column("Name",        style="bold #06b6d4",  no_wrap=True, min_width=16)
    t.add_column("Domain",      style="dim",           no_wrap=True, min_width=14)
    t.add_column("Description", style="")
    for name, meta in EXAMPLES.items():
        t.add_row(name, meta["domain"], meta["description"])
    console.print(t)
    console.print()
    console.print("  [bold]numen run [dim]<name>[/dim][/bold]    Run with scipy (no plot window)")
    console.print("  [bold]numen new [dim]<name>[/dim][/bold]    Scaffold a new model")
    console.print()


@app.command()
def run(
    example: str = typer.Argument(..., help=f"Example name ({', '.join(EXAMPLES)})"),
) -> None:
    """Run a built-in example with scipy."""
    if example not in EXAMPLES:
        console.print(f"  [bold #ef4444]✗[/]  Unknown example [bold]{example!r}[/]. Run [bold]numen list[/bold] to see options.")
        raise typer.Exit(code=1)

    examples_dir = Path(__file__).parent.parent.parent / "examples" / example
    if not examples_dir.exists():
        console.print(f"  [bold #f59e0b]⚠[/]  Example directory not found: {examples_dir}")
        console.print("  [dim](Examples are only available in the development checkout, not an installed package.)[/dim]")
        raise typer.Exit(code=1)

    run_py = examples_dir / "run.py"
    if not run_py.exists():
        console.print(f"  [bold #ef4444]✗[/]  No run.py in {examples_dir}")
        raise typer.Exit(code=1)

    console.print()
    console.print(Rule(f"[bold]numen run {example}[/bold]", style="dim #7c3aed"))
    console.print()
    env = {**os.environ, "MPLBACKEND": "Agg"}
    proc = subprocess.run([sys.executable, str(run_py)], cwd=str(examples_dir), env=env)
    console.print()
    if proc.returncode != 0:
        raise typer.Exit(code=proc.returncode)


@app.command()
def info() -> None:
    """Print the framework quick-reference cheat sheet."""
    console.print(_logo_panel())

    def _panel(title: str, content: str, style: str = "#7c3aed") -> Panel:
        return Panel(content.strip(), title=f"[bold #06b6d4]{title}[/]",
                     border_style=f"dim {style}", padding=(1, 3))

    # Core pattern
    console.print(_panel("Core Pattern", textwrap.dedent("""\
        [bold]Component[/]  [dim]IntegratedField[/dim] (state [italic]x[/]) [dim]+[/dim] [dim]ParameterField[/dim] (param [italic]p[/])
        [bold]System[/]     [dim]spec.view() reads[/dim]  ·  [dim]spec.dx_view() writes (+=)[/dim]
        [bold]World[/]      [dim]dict of components  +  dict of systems[/dim]
        [bold]compile_spec[/](world) → [bold]CompiledSpec[/] → [bold]backend.solve()[/]""")))

    # Backends
    t = Table(box=None, show_header=True, padding=(0, 3), show_edge=False)
    t.add_column("Backend",       style="bold #06b6d4", no_wrap=True)
    t.add_column("Speed",         style="#10b981",      no_wrap=True)
    t.add_column("Use when",      style="dim")
    t.add_row("ScipyBackend",     "1×",       "development, debugging")
    t.add_row("JAXBackend",       "~1500×",   "repeated solves, Monte Carlo")
    t.add_row("JuliaBackend",     "~300–600×","long runs, stiff, one-off")
    t.add_row("JuliaServerPool",  "~300–600×","parameter sweeps, parallel workers")
    console.print(Panel(t, title="[bold #06b6d4]Backends[/]", border_style="dim #7c3aed", padding=(1, 3)))
    console.print(Panel(textwrap.dedent("""\
        [bold #f59e0b]Stiff problems[/]  [dim](multi-timescale, high-frequency oscillations):[/dim]
          Julia → [bold]method="Rodas5P"[/bold]  or  [bold]method="FBDF"[/bold]  [dim](implicit — handles stiffness)[/dim]
          JAX   → [bold]solver="Kvaerno5"[/bold]  [dim](SDIRK implicit) or use Julia if very stiff[/dim]
          If JAX hits max_steps, [dim]pip install equinox[/dim] for better error messages."""),
        title="[bold #06b6d4]Stiff Solvers[/]", border_style="dim #f59e0b", padding=(1, 3)))

    # JAX rules
    console.print(_panel("JAX Rules  ⚠", textwrap.dedent("""\
        [bold #f59e0b]⚠[/]  [bold]import jax.numpy as jnp[/bold]  — never [dim]np.*[/dim] inside dynamics
        [bold #f59e0b]⚠[/]  [bold]jnp.where(cond, a, b)[/bold]   — not [dim]if/else[/dim] on state values
        [bold #f59e0b]⚠[/]  [bold]jnp.sqrt(jnp.maximum(0.0, x))[/bold]  — guard sqrt / log
        [bold #f59e0b]⚠[/]  [bold]solver="Dopri5"[/bold]  — not Tsit5 with tight [italic]atol[/italic]
        [bold #f59e0b]⚠[/]  Both branches of [bold]jnp.where[/bold] are always evaluated — guard NaN/Inf""")))

    # Contact
    console.print(_panel("Smooth Contact", textwrap.dedent("""\
        Sharp [bold]max(0,−pos)[/bold] kink → 99% step rejection.
        Use a C¹-smooth 1 µm ramp:
        [bold]_D = 1e-6[/bold]
        [bold]def _soft_pen(x):[/bold]
        [bold]    return jnp.where(x<=0, 0, jnp.where(x>=_D, x-0.5*_D, 0.5*x*x/_D))[/bold]
        See [dim]examples/fluid_poppet/dynamics.py[/dim] for the full pattern.""")))

    # Parameter sweeps
    console.print(_panel("Parameter Sweeps", textwrap.dedent("""\
        [bold]with JuliaServerPool(n_workers=4, julia_file="dynamics.jl", method="Rodas5P") as pool:[/bold]
        [bold]    results = pool.map([/bold]
        [bold]        lambda srv, p: srv.solve(compile_spec(make_world(p)), tspan),[/bold]
        [bold]        param_list, progress=True,[/bold]
        [bold]    )[/bold]
        Pays JIT cost once per worker. Use [bold]ScipyBackend(progress=True)[/bold] for t-tracking bar.""")))

    # Reference
    console.print(_panel("Reference", textwrap.dedent("""\
        [dim]examples/fluid_poppet/[/dim]   most complete reference (6-state pneumatic system)
        [dim]CLAUDE.md[/dim]               AI assistant context — auto-loaded by Claude Code
        [dim]DESIGN.md[/dim]               architecture decisions and open questions""")))

    console.print()


# ─── _INIT_CLAUDE_MD ──────────────────────────────────────────────────────────

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

For stiff problems (multiple timescales, high-frequency oscillations):
- **Julia**: use `method="Rodas5P"` or `method="FBDF"` — Rosenbrock/BDF implicit
  solvers take far larger steps than explicit methods (Tsit5, Vern7)
- **JAX**: try `solver="Kvaerno5"` (implicit SDIRK) before giving up on JAX;
  if the problem is highly stiff, Rodas5P in Julia will outperform anything JAX can do

If JAX hits `max_steps`, install `equinox` (`pip install equinox`) for clearer
error messages, then either increase `max_steps` or switch to an implicit solver.

---

## Parameter sweeps — JuliaServerBackend and JuliaServerPool

`JuliaBackend` spawns a fresh Julia process per call (~6–12 s startup + JIT).
`JuliaServerBackend` keeps one process alive — pay JIT once, warm-solve forever.
`JuliaServerPool` runs N servers in parallel for multi-core sweeps.

### Single server (sequential sweep)

```python
from numen.bridge.server_backend import JuliaServerBackend

with JuliaServerBackend(
    julia_file="dynamics.jl",
    method="Rodas5P",   # implicit solver — best for stiff problems
    rtol=1e-6,
    atol=1e-8,
    eager=True,         # start Julia immediately, not on first solve
) as server:
    results = []
    for params in parameter_grid:
        spec   = compile_spec(make_world(params))
        result = server.solve(spec, tspan=(0.0, 3600.0))
        results.append(result)
```

### Parallel pool (multi-core sweep)

```python
from numen.bridge.server_backend import JuliaServerPool
import numpy as np

params = [{{"spring_k": k}} for k in np.linspace(100, 1000, 50)]

with JuliaServerPool(
    n_workers=4,                  # 4 Julia processes running simultaneously
    julia_file="dynamics.jl",
    method="Rodas5P",
    rtol=1e-6,
    atol=1e-8,
) as pool:
    results = pool.map(
        lambda server, p: server.solve(compile_spec(make_world(p)), (0.0, 3600.0)),
        params,
        progress=True,            # tqdm bar over completed tasks
    )
```

`pool.map` distributes tasks across all workers and returns results in the
same order as the input list.  Each worker is a full Julia process with
compiled dynamics — no JIT overhead after the first solve per worker.

You can also call `pool.solve(spec, tspan)` directly to acquire an idle
worker (blocking if all are busy) without using `map`.

### Progress bars

`ScipyBackend` supports a real integration-progress bar (tracks `t`):

```python
result = ScipyBackend().solve(spec, tspan, progress=True)
```

`JuliaServerBackend` and `JAXBackend` show an elapsed-time spinner:

```python
result = server.solve(spec, tspan, progress=True)
result = jax_backend.solve(spec, tspan, progress=True)
```

All progress display requires `tqdm` (`pip install tqdm`) and is silently
skipped if tqdm is not installed.  `progress=False` is the default.

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


# ─── Entry point ──────────────────────────────────────────────────────────────
main = app

if __name__ == "__main__":
    app()
