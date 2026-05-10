"""Scaffold templates for numen new and numen init commands."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Backend option helpers
# ---------------------------------------------------------------------------

VALID_BACKENDS = ("scipy", "jax", "julia", "julia_server")

_ALL_BACKEND_IMPORTS = (
    ("scipy",        "from numen.bridge.scipy_backend import ScipyBackend"),
    ("jax",          "from numen.bridge.jax_backend import JAXBackend"),
    ("julia",        "from numen.bridge.runtime import JuliaBackend"),
    ("julia_server", "from numen.bridge.runtime import JuliaServerBackend"),
)

_ALL_BACKEND_INITS = (
    ("scipy",        "ScipyBackend(rtol=1e-8, atol=1e-10)"),
    ("jax",          'JAXBackend(rtol=1e-8, atol=1e-10, solver="Dopri5", max_steps=100_000)'),
    ("julia",        'JuliaBackend(julia_file="dynamics.jl", rtol=1e-8, atol=1e-10)'),
    ("julia_server", 'JuliaServerBackend(julia_file="dynamics.jl", rtol=1e-8, atol=1e-10)'),
)


def _backend_import_block(backend: str) -> str:
    lines = []
    for name, imp in _ALL_BACKEND_IMPORTS:
        lines.append(imp if name == backend else f"# {imp}")
    return "\n".join(lines)


def _backend_solve_line(backend: str) -> str:
    """Return the solve line(s) for run.py. First line has no leading spaces
    (the template's own indentation covers it); subsequent lines carry their own
    4-space indent so they land correctly after template substitution."""
    lines = []
    for name, init in _ALL_BACKEND_INITS:
        line = f"result = {init}.solve(spec, tspan)"
        lines.append(line if name == backend else f"# {line}")
    return "\n    ".join(lines)


def _backend_yaml_section(backend: str) -> str:
    lines = [
        "backend:",
        f"  type: {backend}   # options: scipy | jax | julia | julia_server",
        "  rtol: 1.0e-8",
        "  atol: 1.0e-10",
    ]
    if backend in ("julia", "julia_server"):
        lines.append("  julia_file: dynamics.jl")
    else:
        lines.append("  # julia_file: dynamics.jl   # required for julia / julia_server")
    if backend == "jax":
        lines.append("  solver: Dopri5              # also Tsit5 / Vern7 / Rodas5P")
    else:
        lines.append("  # solver: Dopri5            # jax default; also Tsit5 / Vern7 / Rodas5P")
    if backend == "julia_server":
        lines.append("  n_save_points: 2000")
        lines.append("  # n_workers: 1             # parallel sweep workers")
    else:
        lines.append("  # n_save_points: 2000      # cap output density (julia / julia_server)")
        lines.append("  # n_workers: 1             # parallel sweep workers (julia_server)")
    return "\n".join(lines)


def get_substitutions(model_name: str, backend: str) -> dict[str, str]:
    """Return all template substitution pairs for a given model name and backend."""
    return {
        "{{MODEL_NAME}}":           model_name,
        "{{BACKEND_IMPORT_BLOCK}}": _backend_import_block(backend),
        "{{BACKEND_SOLVE_LINE}}":   _backend_solve_line(backend),
        "{{BACKEND_YAML_SECTION}}": _backend_yaml_section(backend),
    }


EXAMPLES: dict[str, dict] = {
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
    "nonlinear_oscillator": {
        "description": "Nonlinear Duffing-type oscillator with ExcitationPort characterization campaign.",
        "concepts":    ["ExcitationPort", "characterization", "FRF", "amplitude sweep", "chirp"],
        "domain":      "mechanical",
    },
    "pneumatic_dashpot": {
        "description": "Piston in a sealed cylinder with orifice vents — frequency-dependent pneumatic damping.",
        "concepts":    ["orifice flow", "gas-spring", "stiff ODE", "parameter sweep", "Rodas5P"],
        "domain":      "fluid/mechanical",
    },
}


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
    dx  :: AbstractVector{T},
    x   :: AbstractVector{S},
    p   :: Vector{Float64},
    t   :: Real,
    spec:: CompiledSpec,
    sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    for id_body in sys.entity_ids
        i_pos = state_idx(spec, id_body * ".body.position")
        i_vel = state_idx(spec, id_body * ".body.velocity")
        dx[i_pos] += x[i_vel]
    end
end


function spring_dynamics!(
    dx  :: AbstractVector{T},
    x   :: AbstractVector{S},
    p   :: Vector{Float64},
    t   :: Real,
    spec:: CompiledSpec,
    sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    gs = sys.group_size  # 3
    for i in 1:gs:length(sys.entity_ids)
        id_a = sys.entity_ids[i]
        id_s = sys.entity_ids[i + 1]
        id_b = sys.entity_ids[i + 2]

        pos_a = x[state_idx(spec, id_a * ".body.position")]
        pos_b = x[state_idx(spec, id_b * ".body.position")]
        vel_a = x[state_idx(spec, id_a * ".body.velocity")]
        vel_b = x[state_idx(spec, id_b * ".body.velocity")]
        mass_a    = p[param_idx(spec, id_a * ".body.mass")]
        mass_b    = p[param_idx(spec, id_b * ".body.mass")]
        stiffness = p[param_idx(spec, id_s * ".spring.stiffness")]
        rest_len  = p[param_idx(spec, id_s * ".spring.rest_length")]
        damping   = p[param_idx(spec, id_s * ".spring.damping")]

        stretch = (pos_b - pos_a) - rest_len
        rel_vel = vel_b - vel_a
        force   = stiffness * stretch + damping * rel_vel

        dx[state_idx(spec, id_a * ".body.velocity")] +=  force / mass_a
        dx[state_idx(spec, id_b * ".body.velocity")] += -force / mass_b
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
            "body_a": {"body": BodyComponent(position=0.0, velocity=0.0, mass=1.0)},
            "spring": {"spring": SpringComponent(stiffness=100.0, rest_length=1.0, damping=1.0)},
            "body_b": {"body": BodyComponent(position=1.5, velocity=0.0, mass=1.0)},
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
{{BACKEND_IMPORT_BLOCK}}
from numen.reconstruction.collector import SnapshotCollector
from world import make_world


def run():
    world  = make_world()
    spec   = compile_spec(world)

    print("State fields:", list(spec.state_index_map.keys()))
    print("Param fields:", list(spec.param_index_map.keys()))

    tspan  = (0.0, 10.0)
    {{BACKEND_SOLVE_LINE}}
    print(f"Solved: {len(result.t)} steps over {result.t[-1]:.1f} s")

    collector = SnapshotCollector(world, spec, result)
    t, pos_a = collector.field_series("body_a", "body", "position")
    _, pos_b  = collector.field_series("body_b", "body", "position")

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
        "test_plan.yaml": '''\
# Characterization campaign — see CHARACTERIZATION.md for the full schema.
# Run with:  uv run numen characterize test_plan.yaml
#            uv run numen characterize test_plan.yaml -c   # compute only
#            uv run numen characterize test_plan.yaml -p   # plot only
world_module: world
tspan: [0.0, 10.0]

{{BACKEND_YAML_SECTION}}

tests: []   # TODO: add characterization tests (discrete_frequency_sweep, continuous_chirp, …)

plots: []   # TODO: add plot panels (bode, chirp_timeseries, amplitude_sweep, …)
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


# Helper: isentropic orifice mass flow.  P_up/P_dn are type T so ForwardDiff
# can differentiate through them for the state Jacobian.
function orifice_mdot(
    P_up::T, P_dn::T, T_up::Float64,
    R::Float64, Cd::Float64, A::Float64, gamma::Float64,
)::T where T <: Real
    (P_up <= 0.0 || A <= 0.0) && return zero(T)
    beta      = max(zero(T), P_dn) / P_up
    beta_crit = (2.0 / (gamma + 1.0))^(gamma / (gamma - 1.0))
    if beta <= beta_crit
        choke_exp = (gamma + 1.0) / (2.0 * (gamma - 1.0))
        return Cd * A * P_up * sqrt(gamma / (R * T_up)) * (2.0/(gamma+1.0))^choke_exp
    else
        arg = beta^(2.0/gamma) - beta^((gamma+1.0)/gamma)
        return Cd * A * P_up * sqrt(max(zero(T), 2.0*gamma/((gamma-1.0)*R*T_up)*arg))
    end
end


function orifice_flow_dynamics!(
    dx  :: AbstractVector{T},
    x   :: AbstractVector{S},
    p   :: Vector{Float64},
    t   :: Real,
    spec:: CompiledSpec,
    sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    gs = sys.group_size
    for i in 1:gs:length(sys.entity_ids)
        id_a = sys.entity_ids[i]
        id_o = sys.entity_ids[i + 1]
        id_b = sys.entity_ids[i + 2]

        P_a = x[state_idx(spec, id_a * ".control_volume.pressure")]
        P_b = x[state_idx(spec, id_b * ".control_volume.pressure")]
        T_a = p[param_idx(spec, id_a * ".control_volume.temperature")]
        T_b = p[param_idx(spec, id_b * ".control_volume.temperature")]
        R_a = p[param_idx(spec, id_a * ".control_volume.R_specific")]
        R_b = p[param_idx(spec, id_b * ".control_volume.R_specific")]
        V_a = p[param_idx(spec, id_a * ".control_volume.volume")]
        V_b = p[param_idx(spec, id_b * ".control_volume.volume")]
        Cd  = p[param_idx(spec, id_o * ".orifice.Cd")]
        A   = p[param_idx(spec, id_o * ".orifice.area")]
        gam = p[param_idx(spec, id_o * ".orifice.gamma")]

        if P_a >= P_b
            mdot = orifice_mdot(P_a, P_b, T_a, R_a, Cd, A, gam)
        else
            mdot = -orifice_mdot(P_b, P_a, T_b, R_b, Cd, A, gam)
        end

        dx[state_idx(spec, id_a * ".control_volume.pressure")] += -(R_a * T_a / V_a) * mdot
        dx[state_idx(spec, id_b * ".control_volume.pressure")] +=  (R_b * T_b / V_b) * mdot
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
            "inlet":   {"control_volume": ControlVolumeComponent(pressure=P_HIGH, volume=1e-2, temperature=T)},
            "orifice": {"orifice": OrificeComponent(Cd=0.7, area=1e-5, gamma=1.4)},
            "outlet":  {"control_volume": ControlVolumeComponent(pressure=P_LOW,  volume=1e-2, temperature=T)},
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
{{BACKEND_IMPORT_BLOCK}}
from numen.reconstruction.collector import SnapshotCollector
from world import make_world


def run():
    world  = make_world()
    spec   = compile_spec(world)

    print("State fields:", list(spec.state_index_map.keys()))

    tspan  = (0.0, 1.0)
    {{BACKEND_SOLVE_LINE}}
    print(f"Solved: {len(result.t)} steps over {result.t[-1]:.2f} s")

    collector = SnapshotCollector(world, spec, result)
    t, P_in  = collector.field_series("inlet",  "control_volume", "pressure")
    _, P_out  = collector.field_series("outlet", "control_volume", "pressure")

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
        "test_plan.yaml": '''\
# Characterization campaign — see CHARACTERIZATION.md for the full schema.
# Run with:  uv run numen characterize test_plan.yaml
#            uv run numen characterize test_plan.yaml -c   # compute only
#            uv run numen characterize test_plan.yaml -p   # plot only
world_module: world
tspan: [0.0, 1.0]

{{BACKEND_YAML_SECTION}}

tests: []   # TODO: add characterization tests (discrete_frequency_sweep, continuous_chirp, …)

plots: []   # TODO: add plot panels (bode, chirp_timeseries, amplitude_sweep, …)
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
    dx  :: AbstractVector{T},
    x   :: AbstractVector{S},
    p   :: Vector{Float64},
    t   :: Real,
    spec:: CompiledSpec,
    sys :: CompiledSystemSpec,
) where {T <: Real, S <: Real}
    for id_e in sys.entity_ids
        i_state = state_idx(spec, id_e * ".my.state")
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
        components={"entity": {"my": MyComponent()}},
        systems={"system": MySystem()},
    )
''',
    "run.py": '''\
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from numen.compiler.flatten import compile_spec
{{BACKEND_IMPORT_BLOCK}}
from world import make_world


def run():
    world  = make_world()
    spec   = compile_spec(world)
    tspan  = (0.0, 1.0)
    {{BACKEND_SOLVE_LINE}}
    print(f"Solved: {len(result.t)} steps")


if __name__ == "__main__":
    run()
''',
    "test_plan.yaml": '''\
# Characterization campaign — see CHARACTERIZATION.md for the full schema.
# Run with:  uv run numen characterize test_plan.yaml
#            uv run numen characterize test_plan.yaml -c   # compute only
#            uv run numen characterize test_plan.yaml -p   # plot only
world_module: world
tspan: [0.0, 1.0]

{{BACKEND_YAML_SECTION}}

tests: []   # TODO: add characterization tests (discrete_frequency_sweep, continuous_chirp, …)

plots: []   # TODO: add plot panels (bode, chirp_timeseries, amplitude_sweep, …)
''',
}
