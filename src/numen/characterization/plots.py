"""Visualization for characterization results — Bode, operating-point waterfall."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from numen.characterization.results import (
        AmplitudeSweepResult,
        CampaignResults,
        ChirpResult,
        DCSweptFRFResult,
        FRFResult,
        ParameterFamilyResult,
    )


def plot_bode(
    frf: "FRFResult",
    ax_mag: "plt.Axes | None" = None,
    ax_phase: "plt.Axes | None" = None,
    label: str | None = None,
    title: str | None = None,
    db: bool = True,
) -> tuple["plt.Figure", "plt.Axes", "plt.Axes"]:
    """Plot a Bode diagram (magnitude + phase vs frequency) for a single FRF.

    Args:
        frf:      FRFResult from a discrete frequency sweep.
        ax_mag:   Optional existing Axes for magnitude; created if None.
        ax_phase: Optional existing Axes for phase; created if None.
        label:    Legend label.  Defaults to frf.name.
        title:    Figure suptitle.  Defaults to frf.name.
        db:       If True, plot magnitude in dB (20 log10 |H|).

    Returns:
        (fig, ax_mag, ax_phase)
    """
    import matplotlib.pyplot as mpl_plt
    import matplotlib.gridspec as gridspec

    if ax_mag is None or ax_phase is None:
        fig = mpl_plt.figure(figsize=(10, 7))
        gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.05)
        ax_mag   = fig.add_subplot(gs[0])
        ax_phase = fig.add_subplot(gs[1], sharex=ax_mag)
    else:
        fig = ax_mag.get_figure()

    lbl = label or frf.name
    mag = 20.0 * np.log10(np.maximum(frf.magnitudes, 1e-12)) if db else frf.magnitudes

    phases = np.degrees(np.unwrap(np.radians(frf.phases_deg)))
    ax_mag.semilogx(frf.frequencies, mag, lw=2, label=lbl)
    ax_phase.semilogx(frf.frequencies, phases, lw=2, label=lbl)

    # Mark f0 and -3 dB level
    if frf.f0 is not None:
        peak_val = 20.0 * np.log10(np.max(frf.magnitudes)) if db else np.max(frf.magnitudes)
        three_db = peak_val - 3.0 if db else peak_val / np.sqrt(2.0)
        ax_mag.axvline(frf.f0, color="gray", linestyle="--", lw=1, alpha=0.7)
        ax_mag.axhline(three_db, color="gray", linestyle=":", lw=1, alpha=0.7)
        q_str = f"Q={frf.Q:.2f}" if frf.Q else ""
        ax_mag.annotate(
            f"f₀={frf.f0:.3f} Hz  {q_str}",
            xy=(frf.f0, peak_val),
            xytext=(frf.f0 * 1.5, peak_val),
            fontsize=9, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        )

    mag_ylabel = "|H(f)| [dB]" if db else "|H(f)|"
    ax_mag.set_ylabel(mag_ylabel)
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.legend(fontsize=9)
    mpl_plt.setp(ax_mag.get_xticklabels(), visible=False)

    ax_phase.set_xlabel("Frequency [Hz]")
    ax_phase.set_ylabel("Phase [deg]")
    ax_phase.grid(True, which="both", alpha=0.3)

    suptitle = title or frf.name
    fig.suptitle(suptitle, fontsize=12)

    return fig, ax_mag, ax_phase


def plot_bode_family(
    results: list["FRFResult"],
    param_values: list[float] | None = None,
    param_label: str = "param",
    title: str = "FRF Family",
    db: bool = True,
) -> tuple["plt.Figure", "plt.Axes", "plt.Axes"]:
    """Overlay multiple Bode plots from a parameter sweep, coloured by parameter value."""
    import matplotlib.pyplot as mpl_plt
    import matplotlib.gridspec as gridspec
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig = mpl_plt.figure(figsize=(11, 7))
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.05)
    ax_mag   = fig.add_subplot(gs[0])
    ax_phase = fig.add_subplot(gs[1], sharex=ax_mag)

    vals = param_values or list(range(len(results)))
    norm = Normalize(vmin=min(vals), vmax=max(vals))
    cmap = mpl_plt.get_cmap("viridis")

    for frf, val in zip(results, vals):
        color = cmap(norm(val))
        lbl   = f"{param_label}={val:.3g}"
        mag   = 20.0 * np.log10(np.maximum(frf.magnitudes, 1e-12)) if db else frf.magnitudes
        phases = np.degrees(np.unwrap(np.radians(frf.phases_deg)))
        ax_mag.semilogx(frf.frequencies, mag, lw=1.5, color=color, label=lbl)
        ax_phase.semilogx(frf.frequencies, phases, lw=1.5, color=color)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=[ax_mag, ax_phase], label=param_label, fraction=0.03, pad=0.04)

    ax_mag.set_ylabel("|H(f)| [dB]" if db else "|H(f)|")
    ax_mag.grid(True, which="both", alpha=0.3)
    mpl_plt.setp(ax_mag.get_xticklabels(), visible=False)

    ax_phase.set_xlabel("Frequency [Hz]")
    ax_phase.set_ylabel("Phase [deg]")
    ax_phase.grid(True, which="both", alpha=0.3)

    fig.suptitle(title, fontsize=12)
    return fig, ax_mag, ax_phase


def plot_dc_sweep(
    result: "DCSweptFRFResult",
    title: str | None = None,
) -> "plt.Figure":
    """Plot how small-signal gain and phase vary with DC operating point."""
    import matplotlib.pyplot as mpl_plt
    import matplotlib.gridspec as gridspec

    fig = mpl_plt.figure(figsize=(10, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax_mag   = fig.add_subplot(gs[0])
    ax_phase = fig.add_subplot(gs[1])

    dc_vals  = result.dc_values
    mags     = result.magnitudes
    phases   = result.phases_deg

    ax_mag.plot(dc_vals, mags, "o-", lw=2, ms=6)
    ax_mag.set_xlabel("DC offset (force)")
    ax_mag.set_ylabel(f"|H| at f={result.probe_frequency:.2f} Hz")
    ax_mag.set_title("Small-signal gain vs operating point")
    ax_mag.grid(True, alpha=0.3)

    ax_phase.plot(dc_vals, phases, "s-", lw=2, ms=6, color="tab:orange")
    ax_phase.set_xlabel("DC offset (force)")
    ax_phase.set_ylabel("Phase [deg]")
    ax_phase.set_title("Small-signal phase vs operating point")
    ax_phase.grid(True, alpha=0.3)

    fig.suptitle(title or result.name, fontsize=12)
    return fig


def plot_amplitude_sweep(
    result: "AmplitudeSweepResult",
    title: str | None = None,
    db: bool = False,
) -> "plt.Figure":
    """Plot transfer function magnitude and phase vs drive amplitude.

    For a linear system, |H| is flat.  Slopes reveal softening (|H| decreases
    with amplitude) or hardening (|H| increases) nonlinearities.
    """
    import matplotlib.pyplot as mpl_plt
    import matplotlib.gridspec as gridspec

    fig = mpl_plt.figure(figsize=(10, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax_mag   = fig.add_subplot(gs[0])
    ax_phase = fig.add_subplot(gs[1])

    amps = result.drive_amplitudes
    mag  = 20.0 * np.log10(np.maximum(result.H_magnitudes, 1e-12)) if db else result.H_magnitudes

    ax_mag.plot(amps, mag, "o-", lw=2, ms=6)
    ax_mag.set_xlabel("Drive amplitude")
    ylabel = "|H| [dB]" if db else "|H|"
    ax_mag.set_ylabel(f"{ylabel}  at f={result.frequency:.3g} Hz")
    ax_mag.set_title("Gain vs amplitude")
    ax_mag.grid(True, alpha=0.3)

    ax_phase.plot(amps, result.phases_deg, "s-", lw=2, ms=6, color="tab:orange")
    ax_phase.set_xlabel("Drive amplitude")
    ax_phase.set_ylabel("Phase [deg]")
    ax_phase.set_title("Phase vs amplitude")
    ax_phase.grid(True, alpha=0.3)

    fig.suptitle(title or result.name, fontsize=12)
    return fig


def plot_chirp_frf(
    result: "ChirpResult",
    ax_mag: "plt.Axes | None" = None,
    ax_phase: "plt.Axes | None" = None,
    label: str | None = None,
    title: str | None = None,
    db: bool = True,
) -> tuple["plt.Figure", "plt.Axes", "plt.Axes"]:
    """Plot Bode diagram extracted from a chirp result (cross-spectrum FRF).

    Follows the same interface as plot_bode() so chirp and stepped-sine
    results can be overlaid on the same axes.
    """
    import matplotlib.pyplot as mpl_plt
    import matplotlib.gridspec as gridspec

    if ax_mag is None or ax_phase is None:
        fig = mpl_plt.figure(figsize=(10, 7))
        gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.05)
        ax_mag   = fig.add_subplot(gs[0])
        ax_phase = fig.add_subplot(gs[1], sharex=ax_mag)
    else:
        fig = ax_mag.get_figure()

    lbl = label or result.name
    mag = 20.0 * np.log10(np.maximum(result.H_mag, 1e-12)) if db else result.H_mag

    phases = np.degrees(np.unwrap(np.radians(result.H_phase_deg)))
    ax_mag.semilogx(result.frequencies, mag, lw=2, label=lbl, alpha=0.85)
    ax_phase.semilogx(result.frequencies, phases, lw=2, label=lbl, alpha=0.85)

    mag_ylabel = "|H(f)| [dB]" if db else "|H(f)|"
    ax_mag.set_ylabel(mag_ylabel)
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.legend(fontsize=9)
    mpl_plt.setp(ax_mag.get_xticklabels(), visible=False)

    ax_phase.set_xlabel("Frequency [Hz]")
    ax_phase.set_ylabel("Phase [deg]")
    ax_phase.grid(True, which="both", alpha=0.3)

    suptitle = title or result.name
    fig.suptitle(suptitle, fontsize=12)
    return fig, ax_mag, ax_phase
