"""Load results.json from a numen characterize run and plot everything."""
import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.json")


def load():
    with open(RESULTS_FILE) as f:
        return json.load(f)["results"]


def main():
    results = load()
    by_type = {r["type"]: r for r in results}

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Nonlinear Oscillator — Characterization Campaign\n"
        r"$\ddot{x} + (c_0 + c_1 x^2)\dot{x} + \omega^2 x = F(t)$"
        "      ω=2π rad/s  c₀=0.1  c₁=2.0",
        fontsize=13,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── 1. Bode (stepped sine + chirp overlay) ────────────────────────────────
    ax_mag   = fig.add_subplot(gs[0, 0])
    ax_phase = fig.add_subplot(gs[1, 0], sharex=ax_mag)

    def _unwrap_deg(phases):
        return np.degrees(np.unwrap(np.radians(np.array(phases))))

    if "frf" in by_type:
        r = by_type["frf"]
        freqs = np.array(r["frequencies"])
        mags  = 20 * np.log10(np.maximum(r["magnitudes"], 1e-12))
        ax_mag.semilogx(freqs, mags, "o-", lw=2, ms=4, label="stepped sine", color="tab:blue")
        ax_phase.semilogx(freqs, _unwrap_deg(r["phases_deg"]), "o-", lw=2, ms=4, color="tab:blue")
        ax_mag.annotate(
            f"f₀={r['f0']:.3f} Hz\nQ={r['Q']:.1f}",
            xy=(r["f0"], max(mags)),
            xytext=(r["f0"] * 1.6, max(mags) - 6),
            fontsize=8, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        )

    if "chirp" in by_type:
        r = by_type["chirp"]
        freqs = np.array(r["frequencies"])
        H_db  = 20 * np.log10(np.maximum(r["H_mag"], 1e-12))
        ax_mag.semilogx(freqs, H_db, lw=1.5, alpha=0.7, label="chirp (cross-spectrum)", color="tab:orange")
        # Chirp phase is noise for high-Q systems: bandwidth ≪ FFT resolution → don't plot

    ax_mag.set_ylabel("|H(f)| [dB]")
    ax_mag.set_title("Bode — Stepped Sine vs Chirp")
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.legend(fontsize=8)
    plt.setp(ax_mag.get_xticklabels(), visible=False)

    ax_phase.set_xlabel("Frequency [Hz]")
    ax_phase.set_ylabel("Phase [deg]")
    ax_phase.grid(True, which="both", alpha=0.3)
    ax_phase.set_ylim(-200, 20)
    ax_phase.text(
        0.97, 0.95, "Chirp phase omitted\n(high Q → poor SNR)",
        transform=ax_phase.transAxes, ha="right", va="top",
        fontsize=7, color="tab:orange", style="italic",
    )

    # ── 2. Chirp time series ───────────────────────────────────────────────────
    ax_chirp = fig.add_subplot(gs[0, 1])
    if "chirp" in by_type:
        r = by_type["chirp"]
        t = np.array(r["t"])
        y = np.array(r["output"])
        ax_chirp.plot(t, y, lw=0.8, color="tab:orange", alpha=0.85)
        ax_chirp.set_xlabel("t [s]")
        ax_chirp.set_ylabel("position [m]")
        ax_chirp.set_title("Chirp Time Series (log-sweep 0.2→3 Hz)")
        ax_chirp.grid(True, alpha=0.3)

    # ── 3. Amplitude sweep — nonlinearity signature ───────────────────────────
    ax_amp = fig.add_subplot(gs[0, 2])
    if "amplitude_sweep" in by_type:
        r = by_type["amplitude_sweep"]
        amps  = np.array(r["drive_amplitudes"])
        Hmags = np.array(r["H_magnitudes"])
        ax_amp.semilogx(amps, Hmags, "o-", lw=2, ms=6, color="tab:green")
        ax_amp.axhline(Hmags[0], color="gray", linestyle="--", lw=1,
                       label=f"linear limit  |H|={Hmags[0]:.3f}")
        ax_amp.set_xlabel("Drive amplitude [N]")
        ax_amp.set_ylabel(f"|H| at f={r['frequency']:.2f} Hz")
        ax_amp.set_title("Amplitude Sweep — Nonlinearity Signature")
        ax_amp.legend(fontsize=8)
        ax_amp.grid(True, which="both", alpha=0.3)

    # ── 4. DC bias — small-signal gain vs operating point ─────────────────────
    ax_dc_mag = fig.add_subplot(gs[1, 1])
    ax_dc_ph  = fig.add_subplot(gs[1, 2])
    if "dc_swept_frf" in by_type:
        r    = by_type["dc_swept_frf"]
        meas = r["measurements"]
        dc_v = [m["dc_value"]  for m in meas]
        mags = [m["magnitude"] for m in meas]
        phs  = [m["phase_deg"] for m in meas]
        ax_dc_mag.plot(dc_v, mags, "o-", lw=2, ms=6, color="tab:purple")
        ax_dc_mag.set_xlabel("DC force offset [N]")
        ax_dc_mag.set_ylabel(f"|H| at f={r['probe_frequency']:.2f} Hz")
        ax_dc_mag.set_title("DC Sweep — Small-signal Gain vs Bias")
        ax_dc_mag.grid(True, alpha=0.3)

        ax_dc_ph.plot(dc_v, phs, "s-", lw=2, ms=6, color="tab:red")
        ax_dc_ph.set_xlabel("DC force offset [N]")
        ax_dc_ph.set_ylabel("Phase [deg]")
        ax_dc_ph.set_title("DC Sweep — Phase vs Bias")
        ax_dc_ph.grid(True, alpha=0.3)

    out = os.path.join(os.path.dirname(__file__), "characterization_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
