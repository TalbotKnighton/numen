import math
import numpy as np
import matplotlib.pyplot as plt

from numen.reconstruction.collector import SnapshotCollector


def plot_coupled_spring(
    collector: SnapshotCollector,
    k: float,
    rest_length: float,
    title: str = "Coupled Spring-Mass",
) -> plt.Figure:
    t = collector.result.t
    _, x1 = collector.field_series("m1", "mass", "position")
    _, v1 = collector.field_series("m1", "mass", "velocity")
    _, x2 = collector.field_series("m2", "mass", "position")
    _, v2 = collector.field_series("m2", "mass", "velocity")

    # Conservation quantities
    snap0 = collector.at(t[0])
    m1 = snap0.components["m1"]["mass"].mass
    m2 = snap0.components["m2"]["mass"].mass
    total_mass = m1 + m2

    momentum = m1 * v1 + m2 * v2
    ke       = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2
    stretch  = x2 - x1 - rest_length
    pe       = 0.5 * k * stretch**2
    energy   = ke + pe

    x_cm = (m1 * x1 + m2 * x2) / total_mass

    # Analytical solution for equal masses, zero initial velocity
    omega_rel = math.sqrt(k * total_mass / (m1 * m2))  # reduced-mass frequency
    r0 = x2[0] - x1[0]
    A  = r0 - rest_length
    x_cm0 = (m1 * x1[0] + m2 * x2[0]) / total_mass
    x1_ana = x_cm0 - (rest_length + A * np.cos(omega_rel * t)) / 2
    x2_ana = x_cm0 + (rest_length + A * np.cos(omega_rel * t)) / 2

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)

    # Positions
    ax = axes[0, 0]
    ax.plot(t, x1, label="m1 (solver)")
    ax.plot(t, x2, label="m2 (solver)")
    ax.plot(t, x1_ana, "--", alpha=0.5, label="m1 (analytic)")
    ax.plot(t, x2_ana, "--", alpha=0.5, label="m2 (analytic)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("position [m]")
    ax.legend()
    ax.set_title("Positions")

    # Center of mass (should be constant)
    ax = axes[0, 1]
    ax.plot(t, x_cm)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("x_cm [m]")
    ax.set_title("Center of Mass (const)")

    # Total energy (should be constant)
    ax = axes[1, 0]
    ax.plot(t, energy)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("E [J]")
    ax.set_title(f"Total Energy (drift: {energy[-1] - energy[0]:.2e} J)")

    # Total momentum (should be constant)
    ax = axes[1, 1]
    ax.plot(t, momentum)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("p [kg·m/s]")
    ax.set_title(f"Total Momentum (drift: {momentum[-1] - momentum[0]:.2e})")

    return fig
