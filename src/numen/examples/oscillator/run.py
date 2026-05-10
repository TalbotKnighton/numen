"""
1D Harmonic Oscillator — minimal end-to-end numen example.

Runs with the scipy backend (no Julia required).
"""
import math
import matplotlib.pyplot as plt

from numen.compiler.flatten import compile_spec
from numen.bridge.scipy_backend import ScipyBackend
from numen.reconstruction.collector import SnapshotCollector

from world import make_world
from plot import plot_oscillator


def main():
    world = make_world(x0=1.0, v0=0.0, omega=2 * math.pi, damping=0.05)
    spec  = compile_spec(world)

    print("Compiled spec:")
    print(f"  state fields: {list(spec.state_index_map.keys())}")
    print(f"  param fields: {list(spec.param_index_map.keys())}")
    print(f"  systems: {[(s.dynamics_fn, s.entity_ids) for s in spec.systems]}")

    result = ScipyBackend(rtol=1e-9, atol=1e-9).solve(spec, tspan=(0.0, 5.0))
    print(f"\nSolver finished: {len(result.t)} time steps")

    collector = SnapshotCollector(world, spec, result)
    snap = collector.at(0.25)
    osc  = snap.components["osc"]["oscillator"]
    print(f"\nSnapshot at t=0.25s:  position={osc.position:.6f}  velocity={osc.velocity:.6f}")

    fig = plot_oscillator(collector, entity_id="osc", title="1D Harmonic Oscillator")
    plt.tight_layout()
    plt.savefig("oscillator.png", dpi=150)
    print("\nPlot saved to oscillator.png")
    plt.show()


if __name__ == "__main__":
    main()
