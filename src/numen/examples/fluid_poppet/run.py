"""
Fluid poppet check valve example.

Network layout:

    [inlet_tank] --orifice_in--> [inlet_pipe] --poppet--> [outlet_pipe] --orifice_out--> [outlet_tank]

Fluid: air (ideal gas, isothermal)
Inlet tank:  3 bar — large reservoir (slow to deplete)
Outlet tank: 1 bar — ambient sink   (slow to fill)

Cracking pressure = spring_preload / seat_area = 5 N / 5e-5 m² = 1 bar
Available ΔP at t=0 is 2 bar, so the valve opens with 1 bar of net pressure force.
The poppet oscillates on its spring as the pipe pressures equalise, then settles.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from pydantic import Field
from typing import Annotated, Union

sys.path.insert(0, os.path.dirname(__file__))

from components import ControlVolumeComponent, OrificeComponent, PoppetComponent
from dynamics import (
    OrificeFlowSystem,
    PoppetFlowSystem,
    PoppetKinematicsSystem,
    PoppetMechanicsSystem,
    _orifice_mdot,
)

from numen.bridge.scipy_backend import ScipyBackend
from numen.compiler.flatten import compile_spec
from numen.spec.world import GenericWorld


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

AnyComponent = Annotated[
    Union[ControlVolumeComponent, OrificeComponent, PoppetComponent],
    Field(discriminator="kind"),
]
AnySystem = Annotated[
    Union[OrificeFlowSystem, PoppetFlowSystem, PoppetKinematicsSystem, PoppetMechanicsSystem],
    Field(discriminator="kind"),
]
World = GenericWorld[AnyComponent, AnySystem, None]

P_HIGH = 3e5    # Pa — 3 bar
P_LOW  = 1e5    # Pa — 1 bar
T      = 293.15  # K


def make_world() -> World:
    return World(
        components={
            "inlet_tank":  {"control_volume": ControlVolumeComponent(pressure=P_HIGH, volume=1e-2, temperature=T)},
            "outlet_tank": {"control_volume": ControlVolumeComponent(pressure=P_LOW,  volume=1e-2, temperature=T)},
            "inlet_pipe":  {"control_volume": ControlVolumeComponent(pressure=P_HIGH, volume=5e-5, temperature=T)},
            "outlet_pipe": {"control_volume": ControlVolumeComponent(pressure=P_LOW,  volume=5e-5, temperature=T)},
            "orifice_in":  {"orifice": OrificeComponent(Cd=0.7, area=2e-5, gamma=1.4)},
            "orifice_out": {"orifice": OrificeComponent(Cd=0.7, area=2e-5, gamma=1.4)},
            # Cracking ΔP = spring_preload / seat_area = 5 N / 5e-5 m² = 1 bar
            "poppet": {"poppet": PoppetComponent(
                position=0.0,    velocity=0.0,
                mass=0.02,       spring_k=5_000.0,  spring_preload=5.0,
                seat_area=5e-5,  max_travel=3e-3,
                stop_stiffness=1e7, stop_damping=100.0,
                max_flow_area=5e-5, Cd=0.7, gamma=1.4,
            )},
        },
        systems={
            "flow_in":    OrificeFlowSystem(entity_groups=[["inlet_tank", "orifice_in",  "inlet_pipe"]]),
            "flow_valve": PoppetFlowSystem( entity_groups=[["inlet_pipe",  "poppet",      "outlet_pipe"]]),
            "flow_out":   OrificeFlowSystem(entity_groups=[["outlet_pipe", "orifice_out", "outlet_tank"]]),
            "poppet_kin": PoppetKinematicsSystem(),
            "poppet_dyn": PoppetMechanicsSystem(entity_groups=[["inlet_pipe", "poppet", "outlet_pipe"]]),
        },
    )


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def run():
    world = make_world()
    spec  = compile_spec(world)

    print("State fields:", list(spec.state_index_map.keys()))
    print(f"State size: {spec.state_size}   Param size: {spec.param_size}")

    tspan  = (0.0, 0.15)
    t_eval = np.linspace(*tspan, 3000)
    result = ScipyBackend(rtol=1e-8, atol=1e-10).solve(spec, tspan, t_eval=t_eval)
    print(f"Solved: {len(result.t)} steps over {result.t[-1]*1e3:.1f} ms")

    plot(world, spec, result)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(world, spec, result):
    t = result.t
    x = result.x

    def sv(key):   # state vector row
        return x[spec.state_index_map[key][0]]

    P_inlet_tank  = sv("inlet_tank.control_volume.pressure")  / 1e5
    P_inlet_pipe  = sv("inlet_pipe.control_volume.pressure")  / 1e5
    P_outlet_pipe = sv("outlet_pipe.control_volume.pressure") / 1e5
    P_outlet_tank = sv("outlet_tank.control_volume.pressure") / 1e5
    pos_mm        = sv("poppet.poppet.position") * 1e3
    vel           = sv("poppet.poppet.velocity")

    # Reconstruct mass flow through valve at each saved time step
    poppet_c = world.components["poppet"]
    pipe_c   = world.components["inlet_pipe"]
    P_ip = sv("inlet_pipe.pressure")
    P_op = sv("outlet_pipe.pressure")
    pos  = sv("poppet.position")

    opening  = np.clip(pos / poppet_c.max_travel, 0.0, 1.0)
    area     = poppet_c.max_flow_area * opening
    P_up     = np.maximum(P_ip, P_op)
    P_dn     = np.minimum(P_ip, P_op)
    mdot     = np.array([
        _orifice_mdot(pu, pd, pipe_c.temperature, pipe_c.R_specific,
                      poppet_c.Cd, a, poppet_c.gamma)
        for pu, pd, a in zip(P_up, P_dn, area)
    ]) * 1e3  # g/s

    cracking_bar = poppet_c.spring_preload / poppet_c.seat_area / 1e5

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    t_ms = t * 1e3

    ax = axes[0]
    ax.plot(t_ms, P_inlet_tank,  label="inlet tank",  lw=1.5, color="steelblue")
    ax.plot(t_ms, P_inlet_pipe,  label="inlet pipe",  lw=1.5, color="royalblue",  ls="--")
    ax.plot(t_ms, P_outlet_pipe, label="outlet pipe", lw=1.5, color="tomato",     ls="--")
    ax.plot(t_ms, P_outlet_tank, label="outlet tank", lw=1.5, color="firebrick")
    ax.axhline(P_HIGH / 1e5 - cracking_bar, color="gray", ls=":", lw=1,
               label=f"cracking ΔP = {cracking_bar:.1f} bar")
    ax.set_ylabel("Pressure (bar)")
    ax.set_title("Control Volume Pressures")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_ms, P_inlet_pipe - P_outlet_pipe, color="purple", lw=1.5)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("ΔP pipe (bar)")
    ax.set_title("Differential Pressure Across Valve")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t_ms, pos_mm, color="darkorange", lw=1.5, label="position")
    ax.axhline(poppet_c.max_travel * 1e3, color="gray", ls=":", lw=1, label="max travel")
    ax.axhline(0.0, color="gray", ls=":", lw=1, label="seat")
    ax.set_ylabel("Position (mm)")
    ax.set_title("Poppet Position")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(t_ms, mdot, color="seagreen", lw=1.5)
    ax.set_ylabel("Mass flow (g/s)")
    ax.set_xlabel("Time (ms)")
    ax.set_title("Mass Flow Through Valve")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "fluid_poppet.png")
    plt.savefig(out, dpi=150)
    print(f"Plot saved to {out}")
    plt.show()


if __name__ == "__main__":
    run()
