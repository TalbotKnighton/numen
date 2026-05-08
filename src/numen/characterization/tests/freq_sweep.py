"""Discrete frequency sweep (stepped sine) test runner."""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import numpy as np

from numen.characterization.analysis import (
    build_frequency_grid,
    extract_resonance,
    lock_in,
    settle_tspan,
)
from numen.characterization.excitation import set_excitation_params
from numen.characterization.results import FRFResult
from numen.characterization.schema import DiscreteFrequencySweepSpec

_log = logging.getLogger("numen.characterization.tests.freq_sweep")


def run_discrete_frequency_sweep(
    test: DiscreteFrequencySweepSpec,
    exc_spec: Any,
    exc_entity_id: str,
    exc_port_name: str,
    output_state_key: str,
    backend: Any,
) -> FRFResult:
    """Run a stepped-sine frequency sweep and return a full FRF result.

    For each frequency in the grid:
      1. Set excitation params (amp, freq, dc) on exc_spec.
      2. Solve for settle_periods + measure_periods cycles.
      3. Apply lock-in detection on the output state.
      4. Normalise by drive amplitude to get |H(f)|.

    After the sweep, extract f0 and Q via the -3 dB bandwidth method.

    Args:
        test:             Validated DiscreteFrequencySweepSpec from the test plan.
        exc_spec:         CompiledSpec with excitation already injected (placeholder params).
        exc_entity_id:    Entity that owns the ExcitationPort.
        exc_port_name:    Name of the ExcitationPort field.
        output_state_key: Dot-key for the state to measure, e.g. "osc.position".
        backend:          Open solver backend.

    Returns:
        FRFResult with frequencies, magnitudes, phases, f0, Q.
    """
    freqs = build_frequency_grid(
        test.frequencies.f_start,
        test.frequencies.f_end,
        test.frequencies.n_points,
        test.frequencies.spacing,
    )

    magnitudes: list[float] = []
    phases_deg: list[float] = []

    for i, f in enumerate(freqs):
        t_settle, t_end, tspan = settle_tspan(f, test.settle_periods, test.measure_periods)

        spec_f = set_excitation_params(
            exc_spec, exc_entity_id, exc_port_name,
            amp=test.amplitude, freq=f,
            # dc is NOT set here — it is already in exc_spec (set by outer sweep
            # or pre-applied by CharacterizationRunner before this call)
        )

        result = backend.solve(spec_f, tspan=tspan)

        out_idx = spec_f.state_idx(output_state_key)
        output  = result.x[out_idx]

        amp_out, phase_out = lock_in(result.t, output, f, t_start=t_settle)

        H_mag = amp_out / test.amplitude if test.amplitude != 0.0 else 0.0
        magnitudes.append(H_mag)
        phases_deg.append(float(np.degrees(phase_out)))

        _log.debug(
            "f=%.4f Hz  |H|=%.4f  phase=%.1f deg  (%d/%d)",
            f, H_mag, phases_deg[-1], i + 1, len(freqs),
        )

    mag_arr   = np.array(magnitudes)
    phase_arr = np.array(phases_deg)

    f0, Q = extract_resonance(freqs, mag_arr)
    damping_ratio = 1.0 / (2.0 * Q) if Q else None

    _log.info(
        "Sweep '%s' done: f0=%.4f Hz  Q=%s",
        test.name, f0, f"{Q:.2f}" if Q else "N/A",
    )

    return FRFResult(
        name          = test.name,
        frequencies   = freqs,
        magnitudes    = mag_arr,
        phases_deg    = phase_arr,
        f0            = f0,
        Q             = Q,
        damping_ratio = damping_ratio,
        amplitude     = test.amplitude,
        dc_offset     = test.dc_offset,
    )
