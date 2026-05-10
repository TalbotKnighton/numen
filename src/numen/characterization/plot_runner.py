"""CharacterizationPlotter — config-driven figure generation.

Reads the ``plots:`` section of a CharacterizationConfig alongside a loaded
CampaignResults and renders a multi-panel figure.  Each panel type maps to one
renderer function; panels are laid out automatically in a grid.

Usage::

    config  = CharacterizationConfig.from_yaml("test_plan.yaml")
    results = CampaignResults.load("results.json")
    plotter = CharacterizationPlotter(config, results, yaml_dir=Path("."))
    out     = plotter.run()          # returns Path to saved PNG
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure
    import matplotlib.gridspec


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------

def resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else base_dir / p


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _auto_layout(n: int) -> tuple[int, int]:
    """Return (rows, cols) for a roughly-square grid of n panels."""
    if n <= 0:
        return 1, 1
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def _default_size(rows: int, cols: int) -> tuple[float, float]:
    return (cols * 5.5, rows * 4.5)


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

_SUPPORTED_METRICS = ("f0", "Q", "damping_ratio", "peak_magnitude")


def _extract_metric(result: Any, metric: str) -> float | None:
    """Pull a scalar metric from any result type (recurses into family/grid/DOE)."""
    from numen.characterization.results import FRFResult, ParameterFamilyResult

    if isinstance(result, FRFResult):
        if metric == "f0":            return result.f0
        if metric == "Q":             return result.Q
        if metric == "damping_ratio": return result.damping_ratio
        if metric == "peak_magnitude":
            return float(np.max(result.magnitudes)) if len(result.magnitudes) else None

    # For composite results, extract from the first available sub FRF
    sub_results = getattr(result, "sub_results", None)
    if sub_results:
        for sr in sub_results:
            val = _extract_metric(sr, metric)
            if val is not None:
                return val

    return None


# ---------------------------------------------------------------------------
# Individual panel renderers
# ---------------------------------------------------------------------------

def _unwrap_deg(phases: Any) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(np.asarray(phases, dtype=float))))


def _render_bode(fig, subplot_spec, spec, results_by_name):
    """2-stacked-axis Bode: magnitude (top) + phase (bottom)."""
    import matplotlib.gridspec as gridspec
    from numen.characterization.results import FRFResult, ChirpResult

    has_phase = any(s.show_phase for s in spec.series)
    if has_phase:
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec, hspace=0.08)
        ax_mag   = fig.add_subplot(inner[0])
        ax_phase = fig.add_subplot(inner[1], sharex=ax_mag)
    else:
        ax_mag   = fig.add_subplot(subplot_spec)
        ax_phase = None

    for s in spec.series:
        result = results_by_name.get(s.test)
        if result is None:
            continue
        lbl = s.label or s.test

        if isinstance(result, FRFResult):
            mag = 20 * np.log10(np.maximum(result.magnitudes, 1e-12)) if s.db else result.magnitudes
            ax_mag.semilogx(result.frequencies, mag, "o-", lw=2, ms=4, label=lbl)
            if ax_phase is not None and s.show_phase:
                ax_phase.semilogx(result.frequencies, _unwrap_deg(result.phases_deg),
                                  "o-", lw=2, ms=4)
            # f0 annotation
            if result.f0 is not None:
                peak_val = float(np.max(mag))
                ax_mag.axvline(result.f0, color="gray", linestyle="--", lw=1, alpha=0.6)
                ax_mag.annotate(
                    f"f₀={result.f0:.3f} Hz\nQ={result.Q:.1f}" if result.Q else f"f₀={result.f0:.3f} Hz",
                    xy=(result.f0, peak_val),
                    xytext=(result.f0 * 1.6, peak_val - (6 if s.db else peak_val * 0.3)),
                    fontsize=8, color="gray",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                )

        elif isinstance(result, ChirpResult):
            mag = 20 * np.log10(np.maximum(result.H_mag, 1e-12)) if s.db else result.H_mag
            ax_mag.semilogx(result.frequencies, mag, lw=1.5, alpha=0.7, label=lbl)
            # Chirp phase is unreliable for high-Q — skip even if show_phase=True;
            # caller should set show_phase=False for chirp series

    ylabel = "|H(f)| [dB]" if (spec.series and spec.series[0].db) else "|H(f)|"
    ax_mag.set_ylabel(ylabel)
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.legend(fontsize=8)

    if ax_phase is not None:
        import matplotlib.pyplot as plt
        plt.setp(ax_mag.get_xticklabels(), visible=False)
        ax_phase.set_xlabel("Frequency [Hz]")
        ax_phase.set_ylabel("Phase [deg]")
        ax_phase.set_ylim(-200, 20)
        ax_phase.grid(True, which="both", alpha=0.3)
    else:
        ax_mag.set_xlabel("Frequency [Hz]")

    if spec.title:
        ax_mag.set_title(spec.title, fontsize=10)


def _render_chirp_timeseries(fig, subplot_spec, spec, results_by_name):
    from numen.characterization.results import ChirpResult
    ax = fig.add_subplot(subplot_spec)
    result = results_by_name.get(spec.test)
    if result is not None and isinstance(result, ChirpResult):
        ax.plot(result.t, result.output, lw=0.8, color="tab:orange", alpha=0.85)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("output state")
    ax.grid(True, alpha=0.3)
    ax.set_title(spec.title or f"Chirp Time Series — {spec.test}", fontsize=10)


def _render_amplitude_sweep(fig, subplot_spec, spec, results_by_name):
    from numen.characterization.results import AmplitudeSweepResult
    import matplotlib.gridspec as gridspec

    result = results_by_name.get(spec.test)
    if spec.show_phase:
        inner  = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec, hspace=0.08)
        ax_mag = fig.add_subplot(inner[0])
        ax_ph  = fig.add_subplot(inner[1], sharex=ax_mag)
    else:
        ax_mag = fig.add_subplot(subplot_spec)
        ax_ph  = None

    if result is not None and isinstance(result, AmplitudeSweepResult):
        amps  = result.drive_amplitudes
        hmags = result.H_magnitudes
        ax_mag.semilogx(amps, hmags, "o-", lw=2, ms=6, color="tab:green")
        ax_mag.axhline(hmags[0], color="gray", linestyle="--", lw=1,
                       label=f"linear limit  |H|={hmags[0]:.3f}")
        ax_mag.legend(fontsize=8)
        if ax_ph is not None:
            ax_ph.semilogx(amps, _unwrap_deg(result.phases_deg), "s-", lw=2, ms=5,
                           color="tab:orange")
            ax_ph.set_xlabel("Drive amplitude")
            ax_ph.set_ylabel("Phase [deg]")
            ax_ph.grid(True, which="both", alpha=0.3)
            import matplotlib.pyplot as plt
            plt.setp(ax_mag.get_xticklabels(), visible=False)
        else:
            ax_mag.set_xlabel("Drive amplitude")

    ax_mag.set_ylabel(f"|H| at f={getattr(result, 'frequency', '?')} Hz"
                      if result else "|H|")
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.set_title(spec.title or f"Amplitude Sweep — {spec.test}", fontsize=10)


def _render_dc_sweep(fig, subplot_spec, spec, results_by_name):
    from numen.characterization.results import DCSweptFRFResult
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    inner  = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec, hspace=0.08)
    ax_mag = fig.add_subplot(inner[0])
    ax_ph  = fig.add_subplot(inner[1], sharex=ax_mag)

    result = results_by_name.get(spec.test)
    if result is not None and isinstance(result, DCSweptFRFResult):
        dc  = result.dc_values
        mag = result.magnitudes
        ph  = result.phases_deg
        ax_mag.plot(dc, mag, "o-", lw=2, ms=6, color="tab:purple")
        ax_ph.plot(dc, ph, "s-", lw=2, ms=6, color="tab:red")
        ax_mag.set_ylabel(f"|H| at f={result.probe_frequency:.2f} Hz")
    else:
        ax_mag.set_ylabel("|H|")

    plt.setp(ax_mag.get_xticklabels(), visible=False)
    ax_ph.set_xlabel("DC offset")
    ax_ph.set_ylabel("Phase [deg]")
    ax_mag.grid(True, alpha=0.3)
    ax_ph.grid(True, alpha=0.3)
    ax_mag.set_title(spec.title or f"DC Sweep — {spec.test}", fontsize=10)


def _render_parameter_family(fig, subplot_spec, spec, results_by_name):
    from numen.characterization.results import (
        ParameterFamilyResult, FRFResult, ChirpResult, AmplitudeSweepResult,
    )
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    result = results_by_name.get(spec.test)
    if result is None or not isinstance(result, ParameterFamilyResult):
        ax = fig.add_subplot(subplot_spec)
        ax.text(0.5, 0.5, f"No data for '{spec.test}'", transform=ax.transAxes,
                ha="center", va="center", color="gray")
        return

    subs = result.sub_results
    vals = result.param_values[:len(subs)]
    if not subs:
        ax = fig.add_subplot(subplot_spec)
        ax.text(0.5, 0.5, "No sub-results", transform=ax.transAxes,
                ha="center", va="center", color="gray")
        return

    norm = Normalize(vmin=min(vals), vmax=max(vals))
    cmap = plt.get_cmap("viridis")

    # ── FRF family: Bode curves coloured by parameter ──────────────────────
    if isinstance(subs[0], FRFResult):
        inner  = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec, hspace=0.08)
        ax_mag = fig.add_subplot(inner[0])
        ax_ph  = fig.add_subplot(inner[1], sharex=ax_mag) if spec.show_phase else None

        for frf, val in zip(subs, vals):
            color = cmap(norm(val))
            mag   = 20 * np.log10(np.maximum(frf.magnitudes, 1e-12)) if spec.db else frf.magnitudes
            ax_mag.semilogx(frf.frequencies, mag, lw=1.5, color=color)
            if ax_ph is not None:
                ax_ph.semilogx(frf.frequencies, _unwrap_deg(frf.phases_deg), lw=1.5, color=color)

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        axes = [ax_mag, ax_ph] if ax_ph else [ax_mag]
        fig.colorbar(sm, ax=axes, label=result.sweep_param, fraction=0.03, pad=0.04)

        ax_mag.set_ylabel("|H(f)| [dB]" if spec.db else "|H(f)|")
        ax_mag.grid(True, which="both", alpha=0.3)
        if ax_ph is not None:
            plt.setp(ax_mag.get_xticklabels(), visible=False)
            ax_ph.set_xlabel("Frequency [Hz]")
            ax_ph.set_ylabel("Phase [deg]")
            ax_ph.set_ylim(-200, 20)
            ax_ph.grid(True, which="both", alpha=0.3)
        else:
            ax_mag.set_xlabel("Frequency [Hz]")
        ax_mag.set_title(spec.title or f"FRF Family — {result.sweep_param}", fontsize=10)

    # ── Chirp family: magnitude FRFs coloured by parameter ─────────────────
    elif isinstance(subs[0], ChirpResult):
        ax = fig.add_subplot(subplot_spec)
        for chirp, val in zip(subs, vals):
            color = cmap(norm(val))
            mag   = 20 * np.log10(np.maximum(chirp.H_mag, 1e-12)) if spec.db else chirp.H_mag
            ax.semilogx(chirp.frequencies, mag, lw=1.5, alpha=0.8, color=color)

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=result.sweep_param, fraction=0.05, pad=0.04)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("|H(f)| [dB]" if spec.db else "|H(f)|")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(spec.title or f"Chirp Family — {result.sweep_param}", fontsize=10)

    # ── Amplitude sweep family: |H| vs drive amplitude coloured by parameter
    elif isinstance(subs[0], AmplitudeSweepResult):
        ax = fig.add_subplot(subplot_spec)
        for amp_result, val in zip(subs, vals):
            color = cmap(norm(val))
            ax.semilogx(amp_result.drive_amplitudes, amp_result.H_magnitudes,
                        lw=1.5, color=color)

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=result.sweep_param, fraction=0.05, pad=0.04)
        ref_freq = subs[0].frequency if subs else "?"
        ax.set_xlabel("Drive amplitude")
        ax.set_ylabel(f"|H| at f={ref_freq} Hz")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(spec.title or f"Amplitude Family — {result.sweep_param}", fontsize=10)

    else:
        ax = fig.add_subplot(subplot_spec)
        ax.text(0.5, 0.5, f"Unsupported sub-result type:\n{type(subs[0]).__name__}",
                transform=ax.transAxes, ha="center", va="center", color="gray", fontsize=9)


def _render_doe_scatter(fig, subplot_spec, spec, results_by_name):
    from numen.characterization.results import DOESweepResult, ParameterFamilyResult, ParameterGridResult

    result = results_by_name.get(spec.test)
    ax     = fig.add_subplot(subplot_spec)

    if result is None:
        ax.text(0.5, 0.5, f"No data for '{spec.test}'", transform=ax.transAxes,
                ha="center", va="center", color="gray")
        return

    # Collect (x_val, y_val, color_val?) for each design point
    combos      = getattr(result, "combinations", [])
    sub_results = getattr(result, "sub_results", [])

    x_vals = [c.get(spec.x_param) for c in combos]
    y_vals = [_extract_metric(r, spec.y_metric) for r in sub_results]
    valid  = [(x, y) for x, y in zip(x_vals, y_vals) if x is not None and y is not None]

    if not valid:
        ax.text(0.5, 0.5, f"No valid data\n(metric={spec.y_metric}, param={spec.x_param})",
                transform=ax.transAxes, ha="center", va="center", color="gray", fontsize=9)
        return

    xs, ys = zip(*valid)

    if spec.color_param:
        c_vals = [c.get(spec.color_param) for c in combos
                  if c.get(spec.x_param) is not None]
        sc = ax.scatter(xs, ys, c=c_vals, cmap="viridis", s=40, alpha=0.8)
        fig.colorbar(sc, ax=ax, label=spec.color_param, fraction=0.05, pad=0.04)
    else:
        ax.scatter(xs, ys, s=40, alpha=0.8, color="tab:blue")

    ax.set_xlabel(spec.x_param)
    ax.set_ylabel(spec.y_metric)
    ax.grid(True, alpha=0.3)
    ax.set_title(spec.title or f"DOE Scatter — {spec.y_metric} vs {spec.x_param}", fontsize=10)


def _render_parameter_grid_heatmap(fig, subplot_spec, spec, results_by_name):
    from numen.characterization.results import ParameterGridResult

    result = results_by_name.get(spec.test)
    ax     = fig.add_subplot(subplot_spec)

    if result is None or not isinstance(result, ParameterGridResult):
        ax.text(0.5, 0.5, f"No data for '{spec.test}'", transform=ax.transAxes,
                ha="center", va="center", color="gray")
        return

    if len(result.param_keys) != 2:
        ax.text(0.5, 0.5, "Heatmap requires exactly\n2 parameters",
                transform=ax.transAxes, ha="center", va="center", color="gray", fontsize=9)
        return

    key0, key1 = result.param_keys
    vals0 = sorted({c[key0] for c in result.combinations})
    vals1 = sorted({c[key1] for c in result.combinations})

    grid = np.full((len(vals1), len(vals0)), np.nan)
    for combo, sub_r in zip(result.combinations, result.sub_results):
        i = vals1.index(combo[key1])
        j = vals0.index(combo[key0])
        val = _extract_metric(sub_r, spec.metric)
        if val is not None:
            grid[i, j] = val

    im = ax.imshow(grid, aspect="auto", origin="lower",
                   extent=[vals0[0], vals0[-1], vals1[0], vals1[-1]],
                   cmap="viridis")
    fig.colorbar(im, ax=ax, label=spec.metric, fraction=0.05, pad=0.04)
    ax.set_xlabel(key0)
    ax.set_ylabel(key1)
    ax.set_title(spec.title or f"Grid Heatmap — {spec.metric}", fontsize=10)


def _render_two_tone_spectrum(fig, subplot_spec, spec, results_by_name):
    """Full FFT with labelled harmonic and IM product components."""
    from numen.characterization.results import TwoToneResult

    ax     = fig.add_subplot(subplot_spec)
    result = results_by_name.get(spec.test)

    if result is None or not isinstance(result, TwoToneResult):
        ax.text(0.5, 0.5, f"No data for '{spec.test}'",
                transform=ax.transAxes, ha="center", va="center", color="gray")
        return

    # Background spectrum
    mag_db = 20.0 * np.log10(np.maximum(result.spectrum_mag, 1e-30))
    mag_db = np.maximum(mag_db, spec.db_floor)
    ax.plot(result.spectrum_freq, mag_db, lw=0.7, color="tab:blue", alpha=0.6)

    if spec.annotate_products:
        # Color scheme: fundamentals red, key IM3 orange, harmonics green, rest gray
        def _color(name: str) -> str:
            if name in ("f1", "f2"):
                return "red"
            if name in ("2f1-f2", "2f2-f1", "-f1+2f2", "2f2-1f1"):
                return "darkorange"
            if name in ("2f1", "2f2", "3f1", "3f2"):
                return "tab:green"
            return "gray"

        for name, amp in result.components.items():
            if amp <= 0:
                continue
            # Reconstruct the frequency from f1, f2 and the name via a lookup
            f_comp = _component_freq_from_name(name, result.f1, result.f2)
            if f_comp is None or f_comp <= 0:
                continue
            db_val = max(20.0 * np.log10(max(amp, 1e-30)), spec.db_floor)
            color  = _color(name)
            ax.axvline(f_comp, color=color, lw=0.9, alpha=0.55, linestyle="--")
            ax.annotate(
                name, xy=(f_comp, db_val),
                xytext=(0, 6), textcoords="offset points",
                fontsize=7, color=color, rotation=60, ha="left",
            )

    # Info box
    lines = [f"IMD3 = {result.imd3:.1f} dB", f"THD = {result.thd:.2f}%"]
    if result.ip3_estimate is not None:
        lines.append(f"IP3 ≈ {result.ip3_estimate:.3g}")
    ax.text(0.98, 0.02, "\n".join(lines), transform=ax.transAxes, fontsize=8,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|output| [dB]")
    ax.set_ylim(bottom=spec.db_floor)
    ax.grid(True, alpha=0.3)
    ax.set_title(spec.title or f"Two-Tone Spectrum — {spec.test}", fontsize=10)


def _component_freq_from_name(name: str, f1: float, f2: float) -> float | None:
    """Reconstruct the frequency of an IM component from its descriptive name.

    Names follow the pattern produced by extract_intermod_components:
    "f1" → f1, "f2" → f2, "2f1" → 2f1, "2f1-f2" → 2f1-f2, etc.
    Returns None if the name cannot be parsed.
    """
    try:
        import re
        # Replace f1→A and f2→B to evaluate as an expression
        expr = name.replace("f1", f"({f1})").replace("f2", f"({f2})")
        # Insert * between coefficient and opening paren: "2(" → "2*("
        expr = re.sub(r"(\d)\(", r"\1*(", expr)
        val  = float(eval(expr))  # noqa: S307 — controlled substitution, no user input
        return val if val > 0 else None
    except Exception:
        return None


def _render_thd_spectrum(fig, subplot_spec, spec, results_by_name):
    """THD(f) + harmonic magnitudes from a harmonic_distortion_sweep."""
    from numen.characterization.results import HarmonicDistortionResult
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    result = results_by_name.get(spec.test)
    inner  = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec, hspace=0.12)
    ax_thd  = fig.add_subplot(inner[0])
    ax_harm = fig.add_subplot(inner[1], sharex=ax_thd)

    if result is None or not isinstance(result, HarmonicDistortionResult):
        ax_thd.text(0.5, 0.5, f"No data for '{spec.test}'",
                    transform=ax_thd.transAxes, ha="center", va="center", color="gray")
        return

    ax_thd.semilogx(result.freqs, result.thd, lw=2, color="tab:red")
    ax_thd.set_ylabel("THD [%]")
    ax_thd.grid(True, which="both", alpha=0.3)
    ax_thd.set_title(spec.title or f"Harmonic Distortion — {spec.test}", fontsize=10)
    plt.setp(ax_thd.get_xticklabels(), visible=False)

    harm_colors = ["tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink"]
    ax_harm.semilogx(result.freqs, result.H1, lw=2, color="tab:blue", label="H1 (fund)")
    for idx, h_num in enumerate(spec.show_harmonics):
        row = h_num - 2   # Hn row index: H2 → row 0, H3 → row 1, …
        if 0 <= row < len(result.Hn):
            ax_harm.semilogx(
                result.freqs, result.Hn[row],
                lw=1.5, color=harm_colors[idx % len(harm_colors)],
                label=f"H{h_num}",
            )
    ax_harm.set_xlabel("Frequency [Hz]")
    ax_harm.set_ylabel("|H|")
    ax_harm.legend(fontsize=8)
    ax_harm.grid(True, which="both", alpha=0.3)


def _render_backbone_curve(fig, subplot_spec, spec, results_by_name):
    """Backbone curve (frequency vs amplitude) and damping vs amplitude."""
    from numen.characterization.results import FreeDecayResult
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    result = results_by_name.get(spec.decay_test)
    inner  = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=subplot_spec, wspace=0.35)
    ax_bb  = fig.add_subplot(inner[0])
    ax_dmp = fig.add_subplot(inner[1])

    if result is None or not isinstance(result, FreeDecayResult):
        ax_bb.text(0.5, 0.5, f"No data for '{spec.decay_test}'",
                   transform=ax_bb.transAxes, ha="center", va="center", color="gray")
        return

    # Backbone: frequency (x) vs amplitude (y) — "how natural frequency shifts with amplitude"
    ax_bb.plot(result.backbone_frequency, result.backbone_amplitude,
               lw=2, color="tab:blue")
    ax_bb.set_xlabel("Instantaneous frequency [Hz]")
    ax_bb.set_ylabel("Amplitude")
    ax_bb.set_title(spec.title or f"Backbone Curve — {spec.decay_test}", fontsize=10)
    ax_bb.grid(True, alpha=0.3)

    # Time-varying damping over the decay
    ax_dmp.plot(result.t, result.inst_damping, lw=1.2, color="tab:orange", alpha=0.8)
    ax_dmp.set_xlabel("t [s]")
    ax_dmp.set_ylabel("Damping ratio ζ")
    ax_dmp.set_title("Instantaneous Damping", fontsize=10)
    ax_dmp.grid(True, alpha=0.3)


def _render_phase_portrait(fig, subplot_spec, spec, results_by_name):
    """Limit cycle trajectories in (position, velocity) space with Poincaré dots."""
    from numen.characterization.results import PhasePortraitResult
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    ax     = fig.add_subplot(subplot_spec)
    tests  = list(spec.tests)
    if not tests:
        ax.text(0.5, 0.5, "No tests listed", transform=ax.transAxes,
                ha="center", va="center", color="gray")
        return

    valid_results = [
        (name, results_by_name[name])
        for name in tests
        if name in results_by_name and isinstance(results_by_name[name], PhasePortraitResult)
    ]

    if not valid_results:
        ax.text(0.5, 0.5, f"No PhasePortraitResult for:\n{tests}",
                transform=ax.transAxes, ha="center", va="center", color="gray", fontsize=8)
        return

    n      = len(valid_results)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n))

    for (name, result), color in zip(valid_results, colors):
        lbl = f"A={result.amplitude:.3g}, f={result.frequency:.3g} Hz"
        ax.plot(result.x, result.xdot, lw=1.0, color=color, alpha=0.75, label=lbl)
        if spec.show_poincare and result.poincare_x is not None:
            ax.scatter(
                result.poincare_x, result.poincare_xdot,
                s=18, color=color, zorder=5,
                edgecolors="black", linewidths=0.4,
            )

    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(spec.title or "Phase Portrait", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _render_stochastic_response(fig, subplot_spec, spec, results_by_name):
    """Random vibration response: input PSD vs response PSD + BLA + coherence."""
    from numen.characterization.results import StochasticExcitationResult
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as mgs

    result = results_by_name.get(spec.test)
    if result is None or not isinstance(result, StochasticExcitationResult):
        ax = fig.add_subplot(subplot_spec)
        ax.text(0.5, 0.5, f"No StochasticExcitationResult for '{spec.test}'",
                transform=ax.transAxes, ha="center", va="center", color="gray", fontsize=8)
        return

    # Determine number of sub-rows
    show_bla  = spec.show_bla
    show_coh  = spec.show_coherence
    n_rows    = 2 + int(show_bla) + int(show_coh)
    heights   = [1.0, 2.0] + ([1.5] if show_bla else []) + ([1.0] if show_coh else [])

    gs  = mgs.GridSpecFromSubplotSpec(n_rows, 1, subplot_spec=subplot_spec,
                                      hspace=0.55, height_ratios=heights)
    row = 0

    # Row 0 — input signal preview
    ax0 = fig.add_subplot(gs[row]); row += 1
    dt  = result.dt_sig
    ax0.plot(result.t_input, result.input_signal, lw=0.6, color="#64748b", alpha=0.85)
    ax0.set_xlabel("Time [s]", fontsize=7)
    ax0.set_ylabel("Input [m/s²]", fontsize=7)
    ax0.tick_params(labelsize=6)
    ax0.set_title(spec.title or "Stochastic Response", fontsize=9, pad=3)
    ax0.grid(True, alpha=0.25)

    # Scalar info box
    info = (f"input RMS={result.input_rms:.3g} m/s²  "
            f"resp RMS={result.response_rms:.3g}  "
            f"crest={result.crest_factor:.2f}  "
            f"seed={result.seed_used}")
    ax0.text(0.01, 0.97, info, transform=ax0.transAxes, fontsize=5.5,
             va="top", ha="left", color="#374151",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    # Row 1 — PSD comparison
    ax1 = fig.add_subplot(gs[row]); row += 1
    psd_x = result.input_psd
    psd_y = result.response_psd
    if spec.psd_db:
        with np.errstate(divide="ignore"):
            psd_x = 10.0 * np.log10(np.maximum(psd_x, 1e-300))
            psd_y = 10.0 * np.log10(np.maximum(psd_y, 1e-300))
        psd_label = "PSD [dB re 1 m²/s⁴/Hz]"
    else:
        psd_label = "PSD [m²/s⁴/Hz]"

    ax1.semilogy(result.input_psd_freq,    np.maximum(result.input_psd, 1e-300),
                 lw=1.0, color="#94a3b8", label="Input", alpha=0.85)
    ax1.semilogy(result.response_psd_freq, np.maximum(result.response_psd, 1e-300),
                 lw=1.2, color="#0ea5e9", label="Response")
    ax1.set_xlabel("Frequency [Hz]", fontsize=7)
    ax1.set_ylabel("PSD [m²/s⁴/Hz]", fontsize=7)
    ax1.tick_params(labelsize=6)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.25, which="both")

    # Row 2 (optional) — BLA gain
    if show_bla:
        ax2 = fig.add_subplot(gs[row]); row += 1
        ax2.semilogy(result.input_psd_freq, np.maximum(result.bla_gain, 1e-300),
                     lw=1.2, color="#f59e0b")
        ax2.set_ylabel("BLA gain", fontsize=7)
        ax2.set_xlabel("Frequency [Hz]", fontsize=7)
        ax2.tick_params(labelsize=6)
        ax2.grid(True, alpha=0.25, which="both")

    # Row 3 (optional) — Coherence
    if show_coh:
        ax3 = fig.add_subplot(gs[row]); row += 1
        ax3.plot(result.input_psd_freq, result.bla_coherence,
                 lw=1.0, color="#10b981")
        ax3.axhline(1.0, lw=0.5, color="gray", ls="--")
        ax3.set_ylim(-0.05, 1.1)
        ax3.set_ylabel("Coherence γ²", fontsize=7)
        ax3.set_xlabel("Frequency [Hz]", fontsize=7)
        ax3.tick_params(labelsize=6)
        ax3.grid(True, alpha=0.25)


_PANEL_RENDERERS = {
    "bode":                   _render_bode,
    "chirp_timeseries":       _render_chirp_timeseries,
    "amplitude_sweep":        _render_amplitude_sweep,
    "dc_sweep":               _render_dc_sweep,
    "parameter_family":       _render_parameter_family,
    "doe_scatter":            _render_doe_scatter,
    "parameter_grid_heatmap": _render_parameter_grid_heatmap,
    "two_tone_spectrum":      _render_two_tone_spectrum,
    "thd_spectrum":           _render_thd_spectrum,
    "backbone_curve":         _render_backbone_curve,
    "phase_portrait_panel":   _render_phase_portrait,
    "stochastic_response":    _render_stochastic_response,
}


# ---------------------------------------------------------------------------
# CharacterizationPlotter
# ---------------------------------------------------------------------------

class CharacterizationPlotter:
    """Render a multi-panel figure from a CharacterizationConfig + CampaignResults."""

    def __init__(
        self,
        config:   "CharacterizationConfig",
        results:  "CampaignResults",
        yaml_dir: Path,
    ) -> None:
        from numen.characterization.schema import CharacterizationConfig
        from numen.characterization.results import CampaignResults
        self.config   = config
        self.results  = results
        self.yaml_dir = yaml_dir

    def run(self) -> Path:
        """Build and save the figure; return the output path."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        plots_spec = self.config.plots
        panels = [p for p in plots_spec.panels if p.enabled]

        if not panels:
            raise ValueError("No enabled panels in plots: section — nothing to draw.")

        rows, cols = _auto_layout(len(panels))
        fig_size   = plots_spec.figure.size or _default_size(rows, cols)
        fig        = plt.figure(figsize=fig_size)

        # Figure-level title
        fig_title = plots_spec.figure.title
        if fig_title:
            sub = plots_spec.figure.subtitle
            full = f"{fig_title}\n{sub}" if sub else fig_title
            fig.suptitle(full, fontsize=12)

        outer_gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.5, wspace=0.4)

        results_by_name = {r.name: r for r in self.results.all_results()}

        for idx, panel_spec in enumerate(panels):
            row = idx // cols
            col = idx % cols
            cell = outer_gs[row, col]
            renderer = _PANEL_RENDERERS.get(panel_spec.type)
            if renderer is None:
                ax = fig.add_subplot(cell)
                ax.text(0.5, 0.5, f"Unknown panel type:\n{panel_spec.type}",
                        transform=ax.transAxes, ha="center", va="center", color="red")
            else:
                try:
                    renderer(fig, cell, panel_spec, results_by_name)
                except Exception as exc:
                    ax = fig.add_subplot(cell)
                    ax.text(0.5, 0.5, f"Render error:\n{exc}",
                            transform=ax.transAxes, ha="center", va="center",
                            color="red", fontsize=8, wrap=True)

        out = resolve_path(plots_spec.output, self.yaml_dir)
        fig.savefig(out, dpi=plots_spec.dpi, bbox_inches="tight")
        plt.close(fig)
        return out
