import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from numen.bridge.runtime import SolveResult
from numen.reconstruction.collector import SnapshotCollector
from numen.compiler.flatten import CompiledSpec


def plot_oscillator(
    collector: SnapshotCollector,
    entity_id: str = "osc",
    title: str = "1D Oscillator",
) -> plt.Figure:
    t, pos = collector.field_series(entity_id, "oscillator", "position")
    t, vel = collector.field_series(entity_id, "oscillator", "velocity")

    spec = collector.spec
    omega   = collector.spec.p[spec.param_idx(f"{entity_id}.oscillator.omega")]
    damping = collector.spec.p[spec.param_idx(f"{entity_id}.oscillator.damping")]

    x0 = collector.spec.x0[spec.state_idx(f"{entity_id}.oscillator.position")]
    v0 = collector.spec.x0[spec.state_idx(f"{entity_id}.oscillator.velocity")]

    # Analytical solution — undamped or underdamped depending on ζ
    if damping == 0.0:
        pos_exact = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
    elif damping < 1.0:
        omega_d   = omega * np.sqrt(1.0 - damping**2)
        pos_exact = np.exp(-damping * omega * t) * (
            x0 * np.cos(omega_d * t)
            + (v0 + damping * omega * x0) / omega_d * np.sin(omega_d * t)
        )
    else:
        pos_exact = None   # overdamped: skip exact overlay

    energy = 0.5 * vel**2 + 0.5 * omega**2 * pos**2

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"{title}  (ω={omega:.3f} rad/s, ζ={damping:.3f})", fontsize=13)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Position vs time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, pos, label="numerical", lw=2)
    if pos_exact is not None:
        ax1.plot(t, pos_exact, "--", label="exact", lw=1.5, alpha=0.7)
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("position [m]")
    ax1.set_title("Position")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Velocity vs time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, vel, color="tab:orange", lw=2)
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("velocity [m/s]")
    ax2.set_title("Velocity")
    ax2.grid(True, alpha=0.3)

    # Phase diagram
    ax3 = fig.add_subplot(gs[1, 0])
    sc = ax3.scatter(pos, vel, c=t, cmap="viridis", s=8)
    plt.colorbar(sc, ax=ax3, label="t [s]")
    ax3.set_xlabel("position [m]")
    ax3.set_ylabel("velocity [m/s]")
    ax3.set_title("Phase Diagram")
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect("equal", adjustable="datalim")

    # Energy vs time
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, energy, color="tab:green", lw=2)
    ax4.axhline(energy[0], color="gray", linestyle="--", lw=1, label=f"E₀ = {energy[0]:.4f}")
    ax4.set_xlabel("t [s]")
    ax4.set_ylabel("energy [J]")
    energy_title = "Mechanical Energy (conserved)" if damping == 0.0 else f"Mechanical Energy (ζ={damping:.3f} dissipates)"
    ax4.set_title(energy_title)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    return fig
