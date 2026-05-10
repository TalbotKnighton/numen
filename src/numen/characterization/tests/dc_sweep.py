"""DC operating-point sweep test runner.

For each DC bias value:
  1. Settle the system under DC-only forcing (no sine probe).
  2. Apply a small-amplitude sine probe on top of the DC offset.
  3. Extract the small-signal magnitude and phase at the probe frequency.

This maps out how the linearised FRF changes with operating point —
the first-order fingerprint of a nonlinearity.  For a linear system,
the result is flat across all DC values.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import numpy as np

from numen.characterization.analysis import lock_in, settle_tspan
from numen.characterization.excitation import set_excitation_params
from numen.characterization.results import DCSweptFRFResult, OperatingPointMeasurement
from numen.characterization.schema import DCOperatingPointSweepSpec

_log = logging.getLogger("numen.characterization.tests.dc_sweep")


def run_dc_operating_point_sweep(
    test: DCOperatingPointSweepSpec,
    exc_spec: Any,
    exc_entity_id: str,
    exc_port_name: str,
    output_state_key: str,
    backend: Any,
) -> DCSweptFRFResult:
    """Run a DC operating-point sweep and return small-signal FRF vs bias.

    Args:
        test:             Validated DCOperatingPointSweepSpec.
        exc_spec:         CompiledSpec with excitation injected.
        exc_entity_id:    Entity owning the ExcitationPort.
        exc_port_name:    ExcitationPort field name.
        output_state_key: Dot-key for the state to measure.
        backend:          Open solver backend.

    Returns:
        DCSweptFRFResult with one OperatingPointMeasurement per DC value.
    """
    result_obj = DCSweptFRFResult(
        name=test.name,
        probe_frequency=test.probe_frequency,
    )

    t_settle, t_end, tspan_settle = settle_tspan(
        test.probe_frequency, test.settle_periods, 0,
    )
    _, _, tspan_probe = settle_tspan(
        test.probe_frequency, test.settle_periods, test.measure_periods,
    )

    for dc_val in test.dc_values:
        # --- Step 1: settle to operating point under DC-only forcing ---
        spec_settle = set_excitation_params(
            exc_spec, exc_entity_id, exc_port_name,
            amp=0.0, freq=test.probe_frequency, dc=dc_val,
        )
        res_settle = backend.solve(spec_settle, tspan=(0.0, t_settle))
        x_settled  = list(res_settle.x[:, -1])

        # --- Step 2: probe from settled state ---
        spec_probe = replace(spec_settle, x0=x_settled)
        spec_probe = set_excitation_params(
            spec_probe, exc_entity_id, exc_port_name,
            amp=test.probe_amplitude, freq=test.probe_frequency, dc=dc_val,
        )
        _, t_meas_end, _ = settle_tspan(
            test.probe_frequency, test.settle_periods, test.measure_periods,
        )
        res_probe = backend.solve(spec_probe, tspan=(0.0, t_meas_end))

        out_idx   = spec_probe.state_idx(output_state_key)
        output    = res_probe.x[out_idx]
        amp_out, phase_out = lock_in(
            res_probe.t, output, test.probe_frequency, t_start=t_settle,
        )
        H_mag = amp_out / test.probe_amplitude if test.probe_amplitude != 0.0 else 0.0

        result_obj.measurements.append(OperatingPointMeasurement(
            dc_value  = dc_val,
            magnitude = H_mag,
            phase_deg = float(np.degrees(phase_out)),
            f0        = None,
            Q         = None,
        ))

        _log.debug("dc=%.4f  |H|=%.4f  phase=%.1f deg", dc_val, H_mag, np.degrees(phase_out))

    _log.info("DC sweep '%s' done: %d operating points", test.name, len(test.dc_values))
    return result_obj
