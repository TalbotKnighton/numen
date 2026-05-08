"""Amplitude sweep test runner — fixed frequency, varying drive amplitude."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from numen.characterization.analysis import lock_in, settle_tspan
from numen.characterization.excitation import set_excitation_params
from numen.characterization.results import AmplitudeSweepResult
from numen.characterization.schema import AmplitudeSweepSpec

_log = logging.getLogger("numen.characterization.tests.amplitude_sweep")


def run_amplitude_sweep(
    test: AmplitudeSweepSpec,
    exc_spec: Any,
    exc_entity_id: str,
    exc_port_name: str,
    output_state_key: str,
    backend: Any,
) -> AmplitudeSweepResult:
    """Run an amplitude sweep at a fixed frequency and return normalised FRF vs amplitude.

    For each drive amplitude:
      1. Set excitation params (amp, freq=fixed, dc).
      2. Solve for settle_periods + measure_periods cycles.
      3. Apply lock-in detection on the output state.
      4. Normalise by drive amplitude to get |H|.

    For a linear system, |H| is constant across amplitudes.
    Amplitude-dependent variation reveals softening/hardening nonlinearities.

    Args:
        test:             Validated AmplitudeSweepSpec.
        exc_spec:         CompiledSpec with excitation injected.
        exc_entity_id:    Entity owning the ExcitationPort.
        exc_port_name:    ExcitationPort field name.
        output_state_key: Dot-key for the state to measure.
        backend:          Open solver backend.

    Returns:
        AmplitudeSweepResult with drive_amplitudes, response_amplitudes, H_magnitudes.
    """
    t_settle, t_end, tspan = settle_tspan(
        test.frequency, test.settle_periods, test.measure_periods,
    )

    response_amps: list[float] = []
    phases_deg:    list[float] = []
    H_mags:        list[float] = []

    for i, amp in enumerate(test.amplitudes):
        spec_a = set_excitation_params(
            exc_spec, exc_entity_id, exc_port_name,
            amp=amp, freq=test.frequency, dc=test.dc_offset,
        )
        result = backend.solve(spec_a, tspan=tspan)

        out_idx = spec_a.state_idx(output_state_key)
        output  = result.x[out_idx]

        amp_out, phase_out = lock_in(result.t, output, test.frequency, t_start=t_settle)
        H_mag = amp_out / amp if amp != 0.0 else 0.0

        response_amps.append(amp_out)
        phases_deg.append(float(np.degrees(phase_out)))
        H_mags.append(H_mag)

        _log.debug(
            "amp=%.4g  response=%.4g  |H|=%.4f  phase=%.1f deg  (%d/%d)",
            amp, amp_out, H_mag, phases_deg[-1], i + 1, len(test.amplitudes),
        )

    _log.info(
        "Amplitude sweep '%s' done: f=%.4f Hz  %d amplitudes",
        test.name, test.frequency, len(test.amplitudes),
    )

    return AmplitudeSweepResult(
        name                = test.name,
        frequency           = test.frequency,
        drive_amplitudes    = np.array(test.amplitudes),
        response_amplitudes = np.array(response_amps),
        phases_deg          = np.array(phases_deg),
        H_magnitudes        = np.array(H_mags),
        dc_offset           = test.dc_offset,
    )
