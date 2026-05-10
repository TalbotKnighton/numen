"""Random vibration (stochastic excitation) test runner.

Generates a pre-computed forcing signal from a PSD spec or external file,
injects it into the model via table-lookup excitation, solves, and extracts:
  - Response PSD (Welch estimate)
  - Best Linear Approximation (BLA gain and coherence)
  - Scalar metrics: RMS, crest factor

See docs/plan_random_vibe_testing.md for the full design rationale.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import csd, welch

from numen.characterization.excitation import inject_table_excitation
from numen.characterization.results import StochasticExcitationResult
from numen.characterization.schema import (
    MultisineSignalSpec,
    PSDFileSignalSpec,
    PSDProfileSignalSpec,
    StochasticExcitationSpec,
    TimeSeriesFileSignalSpec,
)
from numen.characterization.signal_gen import (
    _make_rng,
    generate_psd_signal,
    load_psd_file,
    load_time_series_file,
    resolve_seed,
)

_log = logging.getLogger("numen.characterization.tests.stochastic_excitation")

# Maximum preview duration stored in the result [s] — prevents huge JSON payloads
_PREVIEW_DURATION_S = 5.0


def run_stochastic_excitation(
    test: StochasticExcitationSpec,
    base_spec: Any,
    exc_entity_id: str,
    exc_port_name: str,
    exc_component_kind: str,
    exc_target_field: str,
    output_state_key: str,
    backend: Any,
    plan_dir: Path | None = None,
    global_seed: int | None = None,
) -> StochasticExcitationResult:
    """Run a random vibration characterization test.

    Args:
        test:               Validated StochasticExcitationSpec.
        base_spec:          CompiledSpec with no excitation injected yet.
        exc_entity_id:      Entity that owns the ExcitationPort.
        exc_port_name:      Name of the ExcitationPort field.
        exc_component_kind: Component kind that owns the ExcitationPort, e.g. ``"nl_oscillator"``.
        exc_target_field:   IntegratedField driven by excitation (e.g. ``"velocity"``).
        output_state_key:   3-part state key for the measured response, e.g. ``"osc.nl_oscillator.position"``.
        backend:            Open solver backend (scipy or julia_server).
        plan_dir:           Directory of the test_plan.yaml (used to resolve relative
                            paths in psd_file / time_series_file signal specs).
        global_seed:        CLI --seed override (negative or None = use per-test seed).

    Returns:
        StochasticExcitationResult.
    """
    effective_seed = resolve_seed(test.seed, global_seed)
    signal_arr, seed_used = _build_signal(test, effective_seed, plan_dir)

    N = len(signal_arr)
    _log.info(
        "stochastic '%s': %d samples, dt_sig=%.4g s, seed=%d, duration=%.1f s",
        test.name, N, test.dt_sig, seed_used, test.duration,
    )

    spec = inject_table_excitation(
        base_spec,
        entity_id      = exc_entity_id,
        component_kind = exc_component_kind,
        port_name      = exc_port_name,
        target_field   = exc_target_field,
        signal         = signal_arr,
        dt_sig         = test.dt_sig,
        t_start        = 0.0,
        dc             = test.dc_offset,
    )

    tspan = (0.0, test.duration)
    result = backend.solve(spec, tspan=tspan)

    out_idx = spec.state_idx(output_state_key)
    t_raw   = result.t
    y_raw   = result.x[out_idx]

    # Strip transient
    t_settle = test.transient_fraction * test.duration
    mask     = t_raw >= t_settle
    t_resp   = t_raw[mask]
    y_resp   = y_raw[mask]

    # Resample response to uniform grid matching dt_sig (Welch needs uniform spacing)
    t_u = np.arange(t_settle, test.duration, test.dt_sig)
    y_u = np.interp(t_u, t_raw, y_raw)

    # Corresponding input signal slice (settled portion, resampled)
    n_settle = int(round(t_settle / test.dt_sig))
    x_u      = signal_arr[n_settle : n_settle + len(y_u)]
    # Guard against length mismatch at boundaries
    min_len  = min(len(x_u), len(y_u))
    x_u      = x_u[:min_len]
    y_u      = y_u[:min_len]

    # Welch PSD estimates
    fs      = 1.0 / test.dt_sig
    n_seg   = max(64, len(x_u) // test.n_welch_segments)
    f_psd, Sxx = welch(x_u, fs=fs, nperseg=n_seg)
    _,     Syy = welch(y_u, fs=fs, nperseg=n_seg)
    _,     Sxy = csd(x_u, y_u, fs=fs, nperseg=n_seg)

    bla_gain      = np.abs(Sxy) / np.maximum(Sxx, 1e-300)
    bla_coherence = np.abs(Sxy) ** 2 / np.maximum(Sxx * Syy, 1e-300)
    bla_coherence = np.clip(bla_coherence, 0.0, 1.0)

    # Scalar metrics
    input_rms    = float(np.sqrt(np.mean(x_u ** 2)))
    response_rms = float(np.sqrt(np.mean(y_u ** 2)))
    peak_resp    = float(np.max(np.abs(y_u))) if len(y_u) > 0 else 0.0
    crest_factor = peak_resp / max(response_rms, 1e-300)

    # Input signal preview (capped at _PREVIEW_DURATION_S)
    n_preview  = min(int(_PREVIEW_DURATION_S / test.dt_sig), N)
    t_input_prev = np.arange(n_preview) * test.dt_sig
    x_input_prev = signal_arr[:n_preview]

    _log.info(
        "stochastic '%s' done: input_rms=%.4g, response_rms=%.4g, crest_factor=%.2f",
        test.name, input_rms, response_rms, crest_factor,
    )

    return StochasticExcitationResult(
        name              = test.name,
        seed_used         = seed_used,
        t                 = t_resp,
        response          = y_resp,
        t_input           = t_input_prev,
        input_signal      = x_input_prev,
        input_psd_freq    = f_psd,
        input_psd         = Sxx,
        response_psd_freq = f_psd,
        response_psd      = Syy,
        bla_gain          = bla_gain,
        bla_coherence     = bla_coherence,
        input_rms         = input_rms,
        response_rms      = response_rms,
        crest_factor      = crest_factor,
        duration          = test.duration,
        dt_sig            = test.dt_sig,
        dc_offset         = test.dc_offset,
        signal_type       = test.signal.type,
    )


# ---------------------------------------------------------------------------
# Signal dispatch
# ---------------------------------------------------------------------------

def _build_signal(
    test: StochasticExcitationSpec,
    seed: int | None,
    plan_dir: Path | None,
) -> tuple[np.ndarray, int]:
    """Generate the pre-computed forcing array from the signal spec.

    Returns (signal_array, seed_used).
    """
    sig = test.signal

    if isinstance(sig, PSDProfileSignalSpec):
        return generate_psd_signal(
            breakpoints = sig.breakpoints,
            duration    = test.duration,
            dt_sig      = test.dt_sig,
            seed        = seed,
            units       = sig.units,
            target_grms = sig.target_grms,
        )

    if isinstance(sig, PSDFileSignalSpec):
        path = _resolve_path(sig.path, plan_dir)
        breakpoints, units_from_file = load_psd_file(path)
        effective_units = sig.units if sig.units != "g_rms" else units_from_file
        return generate_psd_signal(
            breakpoints = breakpoints,
            duration    = test.duration,
            dt_sig      = test.dt_sig,
            seed        = seed,
            units       = effective_units,
            target_grms = sig.target_grms,
        )

    if isinstance(sig, TimeSeriesFileSignalSpec):
        path = _resolve_path(sig.path, plan_dir)
        signal_arr, actual_dt = load_time_series_file(
            path     = path,
            dt_sig   = test.dt_sig,
            resample = sig.resample,
        )
        # Time-series files are deterministic — seed is irrelevant but we
        # still return one so the result field is always populated.
        _, seed_used = _make_rng(seed)
        return signal_arr, seed_used

    if isinstance(sig, MultisineSignalSpec):
        raise NotImplementedError(
            "multisine signal type is Phase 2 — not yet implemented. "
            "Use psd_profile, psd_file, or time_series_file instead."
        )

    raise ValueError(f"Unknown signal type: {type(sig)}")


def _resolve_path(path: str, plan_dir: Path | None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if plan_dir is not None:
        candidate = plan_dir / p
        if candidate.exists():
            return candidate
    return p
