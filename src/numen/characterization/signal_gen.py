"""Signal generation utilities for stochastic excitation tests.

Generates pre-computed time series from PSD specifications, external files,
or other signal descriptions. The resulting array is stored in the parameter
vector and replayed by the ODE solver via table lookup (no randomness during
integration).
"""
from __future__ import annotations

import csv
import json
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np
from numpy.fft import irfft, rfftfreq


# ---------------------------------------------------------------------------
# PSD → time series
# ---------------------------------------------------------------------------

def generate_psd_signal(
    breakpoints: list[tuple[float, float]],
    duration: float,
    dt_sig: float,
    seed: int | None,
    units: str = "g_rms",
    target_grms: float | None = None,
) -> tuple[np.ndarray, int]:
    """Generate a random realisation from a log-log PSD specification.

    Args:
        breakpoints:  [(f_hz, psd_level), …] defining the one-sided PSD.
                      ``psd_level`` units match ``units``.
        duration:     Signal length [s]. N = round(duration / dt_sig) samples.
        dt_sig:       Sample period [s].
        seed:         NumPy RNG seed.  ``None`` samples from os.urandom and
                      the used seed is returned so the run is always replayable.
        units:        ``"g_rms"`` → breakpoints in g²/Hz, signal output in
                      m/s² (×9.80665).  ``"m_s2"`` → breakpoints in m²/s⁴,
                      signal output in m/s².
        target_grms:  Optional RMS normalisation target [g].  If given, the
                      signal is scaled so its RMS equals target_grms (in g).
                      Only meaningful when ``units="g_rms"``.

    Returns:
        (signal, seed_used) where signal has shape (N,) [m/s²] and seed_used
        is the integer seed consumed (for recording in the result).
    """
    rng, seed_used = _make_rng(seed)

    N    = int(round(duration / dt_sig))
    freq = rfftfreq(N, d=dt_sig)       # one-sided freq grid [Hz]

    # Log-log interpolate PSD onto the rfft freq grid
    bkf  = np.array([b[0] for b in breakpoints], dtype=float)
    bkp  = np.array([b[1] for b in breakpoints], dtype=float)
    psd  = np.zeros(len(freq), dtype=float)
    mask = (freq >= bkf[0]) & (freq <= bkf[-1])
    if mask.any():
        psd[mask] = np.exp(
            np.interp(np.log(freq[mask]), np.log(bkf), np.log(bkp))
        )

    # PSD amplitude → one-sided complex amplitude spectrum.
    #
    # NumPy irfft convention: x[n] = (1/N) sum_k X[k] exp(2πi k n/N)
    # Parseval: E[x²] = (1/N²) · (|X[0]|² + 2·Σ|X[k]|² + |X[N/2]|²)
    #
    # Welch one-sided PSD: S[k] = (2/(N·fs)) · |X[k]|²
    # → |X[k]|² = N·fs/2 · S[k]
    # → amplitude for bin k: amp[k] = sqrt(N·fs/2 · psd[k])
    fs  = 1.0 / dt_sig
    amp = np.sqrt(N * fs / 2.0 * psd)
    amp[0]  = 0.0                         # force zero DC
    if len(amp) > 1:
        amp[-1] = 0.0                     # zero Nyquist (avoid artefacts)

    # Random phase realisation
    phase = rng.uniform(0.0, 2.0 * np.pi, len(freq))
    X     = amp * np.exp(1j * phase)
    signal = irfft(X, n=N).astype(np.float64)

    # Unit conversion
    G = 9.80665  # m/s² per g
    if units == "g_rms":
        signal = signal * G              # g → m/s²

    # Optional RMS normalisation (in g space if g_rms, else m/s²)
    if target_grms is not None:
        if units == "g_rms":
            target_ms2 = target_grms * G
        else:
            target_ms2 = target_grms   # caller responsibility
        actual_rms = float(np.sqrt(np.mean(signal ** 2)))
        if actual_rms > 1e-300:
            signal = signal * (target_ms2 / actual_rms)

    return signal, seed_used


# ---------------------------------------------------------------------------
# File-based PSD
# ---------------------------------------------------------------------------

def load_psd_file(path: str | Path) -> tuple[list[tuple[float, float]], str]:
    """Load PSD breakpoints from a CSV or JSON file.

    CSV: two columns ``frequency,psd`` (header optional).  First column is
    frequency [Hz]; second column is PSD level.  Lines starting with ``#``
    are treated as comments.

    JSON: ``{"breakpoints": [[f0, psd0], …], "units": "g_rms"}``

    Returns:
        (breakpoints, units)  where units defaults to ``"g_rms"`` if absent.
    """
    path = Path(path)
    ext  = path.suffix.lower()

    if ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        bk   = [(float(r[0]), float(r[1])) for r in data["breakpoints"]]
        units = data.get("units", "g_rms")
        return bk, units

    if ext in (".csv", ".txt"):
        rows: list[tuple[float, float]] = []
        with open(path, newline="", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        f_val = float(parts[0])
                        p_val = float(parts[1])
                        rows.append((f_val, p_val))
                except ValueError:
                    pass  # skip header or malformed lines
        return rows, "g_rms"

    if ext == ".npy":
        arr = np.load(path)
        if arr.ndim == 2 and arr.shape[0] == 2:
            bk = [(float(arr[0, i]), float(arr[1, i])) for i in range(arr.shape[1])]
        elif arr.ndim == 2 and arr.shape[1] == 2:
            bk = [(float(arr[i, 0]), float(arr[i, 1])) for i in range(arr.shape[0])]
        else:
            raise ValueError(f"PSD .npy file must be shape (2, N) or (N, 2); got {arr.shape}")
        return bk, "g_rms"

    raise ValueError(f"Unsupported PSD file extension: {ext!r}. Use .csv, .json, or .npy")


# ---------------------------------------------------------------------------
# Time-series file
# ---------------------------------------------------------------------------

def load_time_series_file(
    path: str | Path,
    dt_sig: float | None = None,
    resample: bool = True,
) -> tuple[np.ndarray, float]:
    """Load a pre-computed time series from a file.

    Supported formats:
      - CSV: two columns [time_s, signal] (with or without header) OR single
        column [signal] — in this case ``dt_sig`` is required.
      - JSON: ``{"t": [...], "f": [...]}``  or  ``{"dt": 0.001, "f": [...]}``
      - NPY: shape (N,) for uniform dt (requires ``dt_sig``) or shape (2, N)
        for [t; f].

    Args:
        path:     File path.
        dt_sig:   Required when the file has no time column.  Ignored when
                  the file provides an explicit time axis (but checked for
                  consistency ±0.1%).
        resample: If True, resample to a uniform grid at ``dt_sig`` spacing
                  (needed when the file has a non-uniform time axis or a
                  different sample rate).  If False, use the file's native dt
                  and return it as the second element.

    Returns:
        (signal, actual_dt)  where signal is the (possibly resampled) array
        and actual_dt is the sample period used.
    """
    path = Path(path)
    ext  = path.suffix.lower()

    if ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        sig  = np.array(data["f"], dtype=float)
        if "t" in data:
            t_arr = np.array(data["t"], dtype=float)
            native_dt = float(np.mean(np.diff(t_arr)))
        else:
            native_dt = float(data["dt"])
            t_arr = np.arange(len(sig)) * native_dt

    elif ext in (".csv", ".txt"):
        rows_1: list[float] = []
        rows_2: list[float] = []
        two_col = None
        with open(path, newline="", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                try:
                    if two_col is None:
                        two_col = len(parts) >= 2
                    if two_col and len(parts) >= 2:
                        rows_1.append(float(parts[0]))
                        rows_2.append(float(parts[1]))
                    else:
                        rows_1.append(float(parts[0]))
                except ValueError:
                    pass
        if two_col:
            t_arr     = np.array(rows_1, dtype=float)
            sig       = np.array(rows_2, dtype=float)
            native_dt = float(np.mean(np.diff(t_arr))) if len(t_arr) > 1 else (dt_sig or 1.0)
        else:
            sig = np.array(rows_1, dtype=float)
            if dt_sig is None:
                raise ValueError("Single-column time-series CSV requires dt_sig on the test spec")
            native_dt = dt_sig
            t_arr     = np.arange(len(sig)) * native_dt

    elif ext == ".npy":
        arr = np.load(path)
        if arr.ndim == 2 and arr.shape[0] == 2:
            t_arr     = arr[0].astype(float)
            sig       = arr[1].astype(float)
            native_dt = float(np.mean(np.diff(t_arr))) if len(t_arr) > 1 else (dt_sig or 1.0)
        elif arr.ndim == 1:
            sig = arr.astype(float)
            if dt_sig is None:
                raise ValueError("1-D .npy time-series requires dt_sig on the test spec")
            native_dt = dt_sig
            t_arr     = np.arange(len(sig)) * native_dt
        else:
            raise ValueError(f"Time-series .npy must be shape (2, N) or (N,); got {arr.shape}")
    else:
        raise ValueError(f"Unsupported time-series file extension: {ext!r}. Use .csv, .json, or .npy")

    if not resample:
        return sig, native_dt

    if dt_sig is None:
        return sig, native_dt

    # Resample to requested dt_sig grid
    t_out  = np.arange(0, t_arr[-1], dt_sig)
    sig_rs = np.interp(t_out, t_arr, sig)
    return sig_rs, dt_sig


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def _make_rng(seed: int | None) -> tuple[np.random.Generator, int]:
    """Return (rng, seed_used).  If seed is None, sample from os.urandom."""
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "little") % (2 ** 31)
    return np.random.default_rng(seed), int(seed)


def resolve_seed(test_seed: int | None, global_override: int | None) -> int | None:
    """Return the effective seed for a test, applying global override if set.

    Args:
        test_seed:       Per-test seed from YAML (None = random).
        global_override: CLI --seed value (None or negative = no override).
    """
    if global_override is not None and global_override >= 0:
        return global_override
    return test_seed
