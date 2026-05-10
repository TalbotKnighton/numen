"""Harmonic distortion sweep test runner — THD and Hn(f) across a frequency grid."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from numen.characterization.analysis import (
    build_frequency_grid,
    lock_in,
    settle_tspan,
)
from numen.characterization.excitation import set_excitation_params
from numen.characterization.results import HarmonicDistortionResult
from numen.characterization.schema import HarmonicDistortionSweepSpec

_log = logging.getLogger("numen.characterization.tests.harmonic_distortion")


def run_harmonic_distortion_sweep(
    test: HarmonicDistortionSweepSpec,
    exc_spec: Any,
    exc_entity_id: str,
    exc_port_name: str,
    output_state_key: str,
    backend: Any,
) -> HarmonicDistortionResult:
    """Run a stepped-sine sweep and extract harmonic distortion at each frequency.

    For each frequency f in the grid, applies the drive and measures the
    response at 1f, 2f, …, max_harmonic·f via lock-in detection.  Returns
    H_n(f) = |output_nth_harmonic| / |drive| and THD(f).

    The output is normalised by drive amplitude so H1 is directly comparable
    to a standard FRF magnitude.  THD is in percent.

    Args:
        test:             Validated HarmonicDistortionSweepSpec.
        exc_spec:         CompiledSpec with excitation injected (placeholder params).
        exc_entity_id:    Entity that owns the ExcitationPort.
        exc_port_name:    Name of the ExcitationPort field.
        output_state_key: Dot-key for the measured state, e.g. "osc.position".
        backend:          Open solver backend.

    Returns:
        HarmonicDistortionResult with freqs, H1, Hn (shape max_harmonic-1, n_freqs),
        and thd arrays.
    """
    freqs = build_frequency_grid(
        test.frequencies.f_start,
        test.frequencies.f_end,
        test.frequencies.n_points,
        test.frequencies.spacing,
    )

    H1_list: list[float] = []
    Hn_rows: list[list[float]] = []   # each element: [H2, H3, …, H_max_harmonic] at one freq

    for i, f in enumerate(freqs):
        t_settle, _, tspan = settle_tspan(f, test.settle_periods, test.measure_periods)

        spec_f = set_excitation_params(
            exc_spec, exc_entity_id, exc_port_name,
            amp=test.amplitude, freq=f,
        )
        result  = backend.solve(spec_f, tspan=tspan)
        out_idx = spec_f.state_idx(output_state_key)
        t       = result.t
        y       = result.x[out_idx]

        # Fundamental
        amp1, _ = lock_in(t, y, f, t_settle)
        H1_list.append(amp1 / test.amplitude if test.amplitude != 0.0 else 0.0)

        # Higher harmonics
        row: list[float] = []
        for n in range(2, test.max_harmonic + 1):
            amp_n, _ = lock_in(t, y, n * f, t_settle)
            row.append(amp_n / test.amplitude if test.amplitude != 0.0 else 0.0)
        Hn_rows.append(row)

        _log.debug("f=%.4f Hz  H1=%.4f  (%d/%d)", f, H1_list[-1], i + 1, len(freqs))

    H1  = np.array(H1_list)
    Hn  = np.array(Hn_rows).T   # shape (max_harmonic-1, n_freqs)

    # THD = sqrt(sum H2..Hn²) / H1, in percent
    with np.errstate(invalid="ignore", divide="ignore"):
        thd = np.where(
            H1 > 0,
            100.0 * np.sqrt(np.sum(Hn ** 2, axis=0)) / H1,
            0.0,
        )

    _log.info(
        "harmonic_distortion_sweep '%s' done: max_THD=%.2f%% at f=%.4f Hz",
        test.name,
        float(np.max(thd)),
        float(freqs[int(np.argmax(thd))]),
    )

    return HarmonicDistortionResult(
        name      = test.name,
        freqs     = freqs,
        H1        = H1,
        Hn        = Hn,
        thd       = thd,
        amplitude = test.amplitude,
        dc_offset = test.dc_offset,
    )
