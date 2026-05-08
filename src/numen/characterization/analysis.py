"""Signal analysis utilities for characterization campaigns.

All functions operate on plain numpy arrays and are backend-agnostic.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Lock-in detection
# ---------------------------------------------------------------------------

def lock_in(
    t: np.ndarray,
    y: np.ndarray,
    f: float,
    t_start: float,
) -> tuple[float, float]:
    """Extract amplitude and phase at frequency f using lock-in (synchronous) detection.

    Computes the in-phase (I) and quadrature (Q) projections of y onto a
    reference sinusoid at frequency f, averaged over the measurement window
    [t_start, t[-1]].  More accurate than FFT for single-frequency extraction
    because it handles non-integer cycle counts correctly.

    Args:
        t:       Time array.
        y:       Signal array (same length as t).
        f:       Reference frequency [Hz].
        t_start: Start of the measurement window; earlier samples are discarded
                 as settling transient.

    Returns:
        (amplitude, phase_rad) — amplitude is the peak value of the sinusoidal
        component at f; phase is in radians relative to sin(2πft).
    """
    mask = t >= t_start
    t_m, y_m = t[mask], y[mask]
    if len(t_m) < 2:
        return 0.0, 0.0
    T = t_m[-1] - t_m[0]
    if T <= 0.0:
        return 0.0, 0.0
    ref_sin = np.sin(2.0 * np.pi * f * t_m)
    ref_cos = np.cos(2.0 * np.pi * f * t_m)
    I = 2.0 * np.trapezoid(y_m * ref_sin, t_m) / T
    Q = 2.0 * np.trapezoid(y_m * ref_cos, t_m) / T
    amplitude = np.sqrt(I**2 + Q**2)
    phase     = np.arctan2(Q, I)
    return float(amplitude), float(phase)


# ---------------------------------------------------------------------------
# Resonance characterization
# ---------------------------------------------------------------------------

def extract_resonance(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
) -> tuple[float, float | None]:
    """Extract resonant frequency f0 and quality factor Q from a magnitude FRF.

    Uses the -3 dB bandwidth method:
        Q = f0 / (f_high - f_low)
    where f_low and f_high are the frequencies at which |H| = |H_peak| / sqrt(2).

    Linear interpolation is used to locate the -3 dB crossings between
    discrete frequency points.  If only one crossing is found (e.g. the
    sweep doesn't cover the full bandwidth), Q is estimated from that side alone.

    Args:
        frequencies: Frequency array [Hz], must be monotonically increasing.
        magnitudes:  Magnitude array |H(f)|, same length as frequencies.

    Returns:
        (f0, Q) — Q is None if no -3 dB crossing is found at all (flat response).
    """
    peak_idx = int(np.argmax(magnitudes))
    f0       = float(frequencies[peak_idx])
    H_peak   = float(magnitudes[peak_idx])
    H_3db    = H_peak / np.sqrt(2.0)

    # Lower -3 dB crossing (search left of peak)
    f_low: float | None = None
    for i in range(peak_idx, 0, -1):
        if magnitudes[i - 1] < H_3db <= magnitudes[i]:
            t = (H_3db - magnitudes[i]) / (magnitudes[i - 1] - magnitudes[i])
            f_low = float(frequencies[i] + t * (frequencies[i - 1] - frequencies[i]))
            break

    # Upper -3 dB crossing (search right of peak)
    f_high: float | None = None
    for i in range(peak_idx, len(magnitudes) - 1):
        if magnitudes[i] >= H_3db > magnitudes[i + 1]:
            t = (magnitudes[i] - H_3db) / (magnitudes[i] - magnitudes[i + 1])
            f_high = float(frequencies[i] + t * (frequencies[i + 1] - frequencies[i]))
            break

    Q: float | None = None
    if f_low is not None and f_high is not None:
        bw = f_high - f_low
        Q  = f0 / bw if bw > 0 else None
    elif f_low is not None:
        Q = f0 / (2.0 * (f0 - f_low)) if f0 > f_low else None
    elif f_high is not None:
        Q = f0 / (2.0 * (f_high - f0)) if f_high > f0 else None

    return f0, Q


def build_frequency_grid(
    f_start: float,
    f_end: float,
    n_points: int,
    spacing: str = "log",
) -> np.ndarray:
    """Return a frequency array with log or linear spacing."""
    if spacing == "log":
        return np.logspace(np.log10(f_start), np.log10(f_end), n_points)
    return np.linspace(f_start, f_end, n_points)


# ---------------------------------------------------------------------------
# Settling helpers
# ---------------------------------------------------------------------------

def settle_tspan(f: float, settle_periods: int, measure_periods: int) -> tuple[float, float, float]:
    """Return (t_settle, t_end, tspan) for a single-frequency solve.

    t_settle is the time at which the measurement window begins.
    tspan = (0.0, t_end).
    """
    t_settle = settle_periods / f
    t_end    = t_settle + measure_periods / f
    return t_settle, t_end, (0.0, t_end)


# ---------------------------------------------------------------------------
# Chirp FRF extraction
# ---------------------------------------------------------------------------

def chirp_phase(
    t: np.ndarray,
    f_start: float,
    f_end: float,
    duration: float,
    chirp_type: str = "log",
) -> np.ndarray:
    """Return the instantaneous phase φ(t) for a chirp signal.

    The chirp signal is amplitude * sin(φ(t)).

    Args:
        t:          Time array.
        f_start:    Start frequency [Hz].
        f_end:      End frequency [Hz].
        duration:   Total chirp duration [s].
        chirp_type: "log" (geometric rate) or "linear" (constant rate).

    Returns:
        Phase array φ(t) in radians.
    """
    if chirp_type == "log":
        k     = np.log(f_end / f_start)
        phase = 2.0 * np.pi * f_start * duration / k * (np.exp(k * t / duration) - 1.0)
    else:
        phase = 2.0 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2.0 * duration))
    return phase


def analyze_chirp_frf(
    t: np.ndarray,
    output: np.ndarray,
    f_start: float,
    f_end: float,
    duration: float,
    amplitude: float,
    chirp_type: str = "log",
    margin: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract FRF from a chirp-forced simulation via the cross-spectrum method.

    Reconstructs the known input signal from the chirp parameters, then
    estimates H(f) = S_xy(f) / S_xx(f) via FFT.

    Args:
        t:          Time array from the solve.
        output:     Output state time series (same length as t).
        f_start:    Chirp start frequency [Hz].
        f_end:      Chirp end frequency [Hz].
        duration:   Chirp duration [s] (used for phase reconstruction).
        amplitude:  Chirp amplitude.
        chirp_type: "log" or "linear".
        margin:     Fractional guard band — frequencies below f_start*(1-margin)
                    or above f_end*(1+margin) are excluded from the result.

    Returns:
        (frequencies, H_magnitudes, H_phases_deg) — trimmed to the swept band.
    """
    if len(t) < 4:
        empty = np.empty(0)
        return empty, empty, empty

    # Reconstruct the known input signal at the solve's time points
    phase        = chirp_phase(t, f_start, f_end, duration, chirp_type)
    input_signal = amplitude * np.sin(phase)

    # Uniform grid for FFT (resample if needed — solve output may be adaptive)
    n      = len(t)
    t_span = t[-1] - t[0]
    if t_span <= 0.0:
        empty = np.empty(0)
        return empty, empty, empty

    dt    = t_span / (n - 1)
    freqs = np.fft.rfftfreq(n, dt)

    # Cross-spectrum estimate: H(f) = S_xy(f) / S_xx(f)
    X   = np.fft.rfft(input_signal)
    Y   = np.fft.rfft(output)
    Sxx = np.real(X * np.conj(X))
    Sxy = np.conj(X) * Y
    H   = Sxy / (Sxx + 1e-300)

    # Trim to the swept frequency band
    f_lo = f_start * (1.0 - margin)
    f_hi = f_end   * (1.0 + margin)
    mask = (freqs >= f_lo) & (freqs <= f_hi) & (freqs > 0.0)

    return freqs[mask], np.abs(H)[mask], np.degrees(np.angle(H)[mask])
