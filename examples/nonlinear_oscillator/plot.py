import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from numen.reconstruction.collector import SnapshotCollector


def plot_nl_oscillator(
    collector: SnapshotCollector,
    entity_id: str = "osc",
    title: str = "Nonlinear Oscillator",
    linear_collector: SnapshotCollector | None = None,
) -> plt.Figure:
    t, pos = collector.field_series(entity_id, "position")
    t, vel = collector.field_series(entity_id, "velocity")

    spec  = collector.spec
    omega = spec.p[spec.param_idx(f"{entity_id}.omega")]
    c0    = spec.p[spec.param_idx(f"{entity_id}.c0")]
    c1    = spec.p[spec.param_idx(f"{entity_id}.c1")]

    effective_damping = c0 + c1 * pos ** 2
    energy            = 0.5 * vel ** 2 + 0.5 * omega ** 2 * pos ** 2

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"{title}  (ω={omega:.2f} rad/s,  c₀={c0:.3f},  c₁={c1:.3f})",
        fontsize=13,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # --- Position vs time ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, pos, lw=2, label="nonlinear")
    if linear_collector is not None:
        tl, posl = linear_collector.field_series(entity_id, "position")
        ax1.plot(tl, posl, "--", lw=1.5, alpha=0.7, label="linear (c₁=0)")
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("position [m]")
    ax1.set_title("Position")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Phase diagram ---
    ax2 = fig.add_subplot(gs[0, 1])
    sc = ax2.scatter(pos, vel, c=t, cmap="viridis", s=8)
    plt.colorbar(sc, ax=ax2, label="t [s]")
    ax2.set_xlabel("position [m]")
    ax2.set_ylabel("velocity [m/s]")
    ax2.set_title("Phase Portrait  (spirals to origin)")
    ax2.grid(True, alpha=0.3)

    # --- Effective damping vs time ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, effective_damping, color="tab:orange", lw=2)
    ax3.axhline(c0, color="gray", linestyle="--", lw=1, label=f"linear baseline c₀={c0:.3f}")
    ax3.set_xlabel("t [s]")
    ax3.set_ylabel("c(x) = c₀ + c₁x²")
    ax3.set_title("Effective Damping Coefficient")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- Energy vs time ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, energy, color="tab:green", lw=2)
    ax4.axhline(energy[0], color="gray", linestyle="--", lw=1, label=f"E₀ = {energy[0]:.4f} J")
    ax4.set_xlabel("t [s]")
    ax4.set_ylabel("E = ½v² + ½ω²x²  [J]")
    ax4.set_title("Mechanical Energy")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    return fig
