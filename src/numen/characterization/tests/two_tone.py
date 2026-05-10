"""Two-tone intermodulation distortion test runner."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from numen.characterization.analysis import (
    extract_intermod_components,
    lock_in,
    spectrum_fft,
)
from numen.characterization.excitation import inject_excitation, set_excitation_params
from numen.characterization.results import TwoToneResult
from numen.characterization.schema import TwoToneSpec

_log = logging.getLogger("numen.characterization.tests.two_tone")


def run_two_tone(
    test: TwoToneSpec,
    exc_spec: Any,
    exc_entity_id: str,
    exc_port_name: str,
    exc_component_kind: str,
    exc_target_field: str,
    output_state_key: str,
    backend: Any,
) -> TwoToneResult:
    """Run a two-tone test and return intermodulation distortion metrics.

    Injects two independent sinusoids (f1, f2) into the same target slot by
    calling inject_excitation twice with different synthetic port names.  Both
    systems accumulate via += so the drive is A1·sin(2πf1·t) + A2·sin(2πf2·t).

    After the simulation reaches steady state (first 70% of the run discarded),
    the output is analysed via:
      - Full FFT for the spectrum plot
      - Lock-in detection at all IM product frequencies up to max_order
      - THD, IMD3, and IP3 estimate from the extracted component amplitudes

    Args:
        test:             Validated TwoToneSpec from the test plan.
        exc_spec:         CompiledSpec with the first excitation system injected
                          (placeholder parameters; will be updated to amp=A1, freq=f1).
        exc_entity_id:      Entity that owns the ExcitationPort.
        exc_port_name:      Name of the primary ExcitationPort field.
        exc_component_kind: Component kind that owns the port, e.g. "nl_oscillator".
        exc_target_field:   IntegratedField whose derivative is driven (e.g. "velocity").
        output_state_key:   3-part dot-key for the measured state, e.g. "osc.nl_oscillator.position".
        backend:          Open solver backend.

    Returns:
        TwoToneResult with spectrum, component amplitudes, THD, IMD3, and IP3.
    """
    # First tone uses the pre-injected excitation port
    spec = set_excitation_params(
        exc_spec, exc_entity_id, exc_port_name,
        amp=test.amplitude1, freq=test.f1, dc=test.dc_offset,
    )
    # Second tone is a fresh inject with a unique synthetic port name
    spec = inject_excitation(
        spec, exc_entity_id, exc_component_kind, exc_port_name + "_tone2", exc_target_field,
        amp=test.amplitude2, freq=test.f2, dc=0.0,
    )

    t_end    = test.n_cycles / test.f1
    t_settle = 0.7 * t_end
    tspan    = (0.0, t_end)

    _log.debug(
        "two_tone '%s': f1=%.4f Hz, f2=%.4f Hz, A1=%.4g, A2=%.4g, t_end=%.3f s",
        test.name, test.f1, test.f2, test.amplitude1, test.amplitude2, t_end,
    )

    result  = backend.solve(spec, tspan=tspan)
    out_idx = spec.state_idx(output_state_key)
    t       = result.t
    y       = result.x[out_idx]

    # Full FFT (settled portion)
    freqs, mags = spectrum_fft(t, y, t_start=t_settle)

    # Extract harmonic and IM product amplitudes
    components = extract_intermod_components(
        t, y, test.f1, test.f2, test.max_order, t_settle,
    )

    # THD: all components except the fundamentals relative to their mean amplitude
    fund1 = components.get("f1", 0.0)
    fund2 = components.get("f2", 0.0)
    fund_amp = (fund1 + fund2) / 2.0
    harm_power = sum(
        v ** 2
        for k, v in components.items()
        if k not in ("f1", "f2")
    )
    thd = 100.0 * np.sqrt(harm_power) / fund_amp if fund_amp > 0 else 0.0

    # IMD3: strongest lower-sideband 3rd-order product relative to fundamentals
    # Lower sidebands 2f1-f2 and 2f2-f1 are the diagnostically important in-band products
    im3_candidates = [
        components.get(k, 0.0)
        for k in components
        if ("2f1" in k and "f2" in k and "-" in k) or ("2f2" in k and "f1" in k and "-" in k)
    ]
    im3 = max(im3_candidates) if im3_candidates else 0.0
    imd3_db = 20.0 * np.log10(im3 / fund_amp) if (im3 > 0 and fund_amp > 0) else float("-inf")

    # Input-referred IP3: A_IP3 = A · sqrt(fund / im3)
    # Derivation: for a cubic nonlinearity, fund ~ c1·A and im3 ~ (3/4)c3·A³
    # At IP3: c1·A_IP3 = (3/4)c3·A_IP3³ → A_IP3 = sqrt(4c1/(3c3))
    # From measurement at amplitude A: im3/fund = (3/4)(c3/c1)·A²
    # → A_IP3 = A · sqrt(fund/im3)   (exact for a pure cubic, approximate otherwise)
    A_avg    = (test.amplitude1 + test.amplitude2) / 2.0
    ip3      = float(A_avg * np.sqrt(fund_amp / im3)) if im3 > 0 else None

    _log.info(
        "two_tone '%s' done: IMD3=%.1f dB  IP3=%s  THD=%.2f%%",
        test.name, imd3_db,
        f"{ip3:.3g}" if ip3 is not None else "N/A",
        thd,
    )

    return TwoToneResult(
        name         = test.name,
        f1           = test.f1,
        f2           = test.f2,
        amplitude1   = test.amplitude1,
        amplitude2   = test.amplitude2,
        spectrum_freq = freqs,
        spectrum_mag  = mags,
        components   = components,
        thd          = float(thd),
        imd3         = float(imd3_db),
        ip3_estimate = ip3,
    )
