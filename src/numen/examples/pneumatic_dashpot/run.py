"""
Pneumatic dashpot — ring-down demo.

A 0.5 kg piston inside a 40 mm bore cylinder, both chambers vented to
atmosphere through small orifices.  The orifice acts as a frequency-dependent
damper: slow displacements allow pressure to equalize (no spring force); fast
oscillations cannot equalize in time, so the compressed/expanded gas acts as a
spring.

This demo shows ring-down from a 20 mm initial displacement for three orifice
areas, spanning nearly-sealed (high-Q resonance) to nearly-open (overdamped).
"""
import math
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from world import make_world

from numen.bridge.scipy_backend import ScipyBackend
from numen.compiler.flatten import compile_spec
from numen.reconstruction.collector import SnapshotCollector


def run_ringdown(orifice_area: float, x0: float = 0.020) -> SnapshotCollector:
    world  = make_world(x0=x0, orifice_area=orifice_area)
    spec   = compile_spec(world)
    result = ScipyBackend(rtol=1e-8, atol=1e-9, dtmax=5e-4).solve(spec, tspan=(0.0, 2.0))
    return SnapshotCollector(world, spec, result)


def main() -> None:
    x0 = 0.020  # 20 mm initial displacement

    orifice_configs = [
        (1e-7, "A_o = 1e-7 m²  (nearly sealed — high Q)"),
        (1e-5, "A_o = 1e-5 m²  (default — moderate damping)"),
        (5e-4, "A_o = 5e-4 m²  (large — overdamped)"),
    ]

    collectors = []
    for A_o, label in orifice_configs:
        print(f"Solving: {label} …", end=" ", flush=True)
        col = run_ringdown(A_o, x0=x0)
        collectors.append((label, col))
        t, pos = col.field_series("piston", "pneumatic_dashpot", "position")
        print(f"{len(t)} steps")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax_pos, ax_pres = axes

    colors = ["#2196f3", "#4caf50", "#f44336"]

    for (label, col), color in zip(collectors, colors):
        t, pos   = col.field_series("piston", "pneumatic_dashpot", "position")
        _, p_L   = col.field_series("piston", "pneumatic_dashpot", "p_left")
        _, p_R   = col.field_series("piston", "pneumatic_dashpot", "p_right")

        ax_pos.plot(t, pos * 1e3, color=color, lw=1.2, label=label)
        ax_pres.plot(t, (p_L - p_R) * 1e-3, color=color, lw=1.2, label=label)

    # Natural frequency reference line
    import numpy as np
    bore_area   = math.pi * 0.02**2
    half_stroke = 0.08
    clearance   = 0.005
    k_pneu = 2 * 101_325 * bore_area / (half_stroke + clearance)
    f0     = math.sqrt(k_pneu / 0.5) / (2 * math.pi)
    t_ref  = np.linspace(0, 2, 2000)
    zeta   = 2.0 / (2 * math.sqrt(k_pneu * 0.5))
    omega  = 2 * math.pi * f0
    env    = x0 * np.exp(-zeta * omega * t_ref) * 1e3
    ax_pos.plot(t_ref,  env, "k--", lw=0.8, alpha=0.4, label=f"linear envelope  f₀={f0:.1f} Hz")
    ax_pos.plot(t_ref, -env, "k--", lw=0.8, alpha=0.4)

    ax_pos.set_ylabel("Position (mm)")
    ax_pos.set_title("Pneumatic Dashpot — Ring-down from 20 mm")
    ax_pos.legend(fontsize=8, loc="upper right")
    ax_pos.axhline(0, color="gray", lw=0.5)
    ax_pos.grid(True, alpha=0.3)

    ax_pres.set_ylabel("ΔP = P_left − P_right (kPa)")
    ax_pres.set_xlabel("Time (s)")
    ax_pres.axhline(0, color="gray", lw=0.5)
    ax_pres.legend(fontsize=8, loc="upper right")
    ax_pres.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "pneumatic_dashpot_ringdown.png")
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
