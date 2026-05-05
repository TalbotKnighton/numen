"""
Three-mass spring chain — coupled system stress test.

Topology: m1 --- s1 --- m2 --- s2 --- m3
  s1 at rest (rest_length=1.0, initial separation=1.0)
  s2 stretched by 1 m (rest_length=1.0, initial separation=2.0)

SpringForceSystem is registered ONCE with entity_groups declaring the topology.
compile_spec validates each group against EntityGroup(MassComponent, SpringComponent, MassComponent)
and pre-caches the grouped tuples on CompiledSystem.  m2 receives force contributions
from both springs; += accumulation sums them correctly.

Conservation laws (no damping, no external forces):
  - Total momentum p = m1*v1 + m2*v2 + m3*v3 = const
  - Total energy E = ½Σmᵢvᵢ² + ½k(stretch1² + stretch2²) = const
  - Center of mass moves at constant velocity
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt

from numen.compiler.flatten import compile_spec
from numen.bridge.scipy_backend import ScipyBackend
from numen.reconstruction.collector import SnapshotCollector

from world import make_world


def main():
    world = make_world()
    spec  = compile_spec(world)

    print("Compiled spec:")
    print(f"  state fields : {list(spec.state_index_map.keys())}")
    print(f"  param fields : {list(spec.param_index_map.keys())}")
    for s in spec.systems:
        print(f"  system  : {s.dynamics_fn}")
        print(f"    entity_ids = {s.entity_ids}")

    result = ScipyBackend(rtol=1e-10, atol=1e-10).solve(spec, tspan=(0.0, 10.0))
    print(f"\nSolver finished: {len(result.t)} time steps")

    collector = SnapshotCollector(world, spec, result)

    _, x1 = collector.field_series("m1", "position")
    _, v1 = collector.field_series("m1", "velocity")
    _, x2 = collector.field_series("m2", "position")
    _, v2 = collector.field_series("m2", "velocity")
    _, x3 = collector.field_series("m3", "position")
    _, v3 = collector.field_series("m3", "velocity")

    k = 10.0
    stretch1 = x2 - x1 - 1.0
    stretch2 = x3 - x2 - 1.0
    energy   = 0.5 * (v1**2 + v2**2 + v3**2) + 0.5 * k * (stretch1**2 + stretch2**2)
    momentum = v1 + v2 + v3

    print(f"\nConservation over t=[0, 10]:")
    print(f"  energy drift   : {energy[-1] - energy[0]:.2e} J")
    print(f"  momentum drift : {momentum[-1] - momentum[0]:.2e} kg·m/s")

    t = result.t
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("3-Mass Spring Chain  (k=10, m=1,  s2 stretched 1 m)")

    axes[0].plot(t, x1, label="m1")
    axes[0].plot(t, x2, label="m2")
    axes[0].plot(t, x3, label="m3")
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("position [m]")
    axes[0].legend()
    axes[0].set_title("Positions")

    axes[1].plot(t, energy)
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("E [J]")
    axes[1].set_title(f"Total Energy (drift: {energy[-1]-energy[0]:.2e} J)")

    plt.tight_layout()
    plt.savefig("coupled_spring.png", dpi=150)
    print("\nPlot saved to coupled_spring.png")
    plt.show()


if __name__ == "__main__":
    main()
