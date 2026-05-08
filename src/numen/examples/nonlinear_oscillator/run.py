"""
Nonlinear oscillator: ẍ + (c0 + c1·x²)ẋ + ω²x = 0

The damping coefficient grows with displacement squared.
Large-amplitude swings decay quickly; small-amplitude oscillations
near the origin are only lightly damped — producing a two-timescale decay.

Compare against a purely linear oscillator (c1=0) with the same c0
to see the effect of the position-dependent term.
"""
import sys
import math
import matplotlib.pyplot as plt

from numen.compiler.flatten import compile_spec
from numen.bridge.scipy_backend import ScipyBackend
from numen.reconstruction.collector import SnapshotCollector

# Add the example directory to sys.path so relative imports work when run directly
import os
sys.path.insert(0, os.path.dirname(__file__))

from world import make_world
from plot import plot_nl_oscillator


def main():
    omega = 2 * math.pi   # 1 Hz natural frequency
    c0    = 0.1           # light baseline damping
    c1    = 2.0           # strong position-dependent damping
    x0    = 2.0           # start at 2× the "crossover" radius
    tspan = (0.0, 20.0)

    # --- Nonlinear oscillator ---
    world_nl = make_world(x0=x0, v0=0.0, omega=omega, c0=c0, c1=c1)
    spec_nl  = compile_spec(world_nl)

    print("State fields:", list(spec_nl.state_index_map.keys()))
    print("Param fields:", list(spec_nl.param_index_map.keys()))

    result_nl = ScipyBackend(rtol=1e-9, atol=1e-9).solve(spec_nl, tspan=tspan)
    print(f"Nonlinear solve: {len(result_nl.t)} steps")
    collector_nl = SnapshotCollector(world_nl, spec_nl, result_nl)

    # --- Linear reference (c1=0, same c0) ---
    world_lin = make_world(x0=x0, v0=0.0, omega=omega, c0=c0, c1=0.0)
    spec_lin  = compile_spec(world_lin)
    result_lin = ScipyBackend(rtol=1e-9, atol=1e-9).solve(spec_lin, tspan=tspan)
    print(f"Linear solve:    {len(result_lin.t)} steps")
    collector_lin = SnapshotCollector(world_lin, spec_lin, result_lin)

    # --- Snapshot comparison at t=5 s ---
    for label, col in [("Nonlinear", collector_nl), ("Linear   ", collector_lin)]:
        snap = col.at(t=5.0)
        osc  = snap.components["osc"]
        print(f"{label} at t=5 s:  x={osc.position:+.6f}  v={osc.velocity:+.6f}")

    fig = plot_nl_oscillator(
        collector_nl,
        entity_id="osc",
        title="Nonlinear Oscillator  (c(x) = c₀ + c₁x²)",
        linear_collector=collector_lin,
    )
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "nonlinear_oscillator.png")
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
