"""Continuous chirp sweep runner — single solve with frequency-swept forcing."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from numen.characterization.analysis import analyze_chirp_frf
from numen.characterization.excitation import inject_chirp_excitation
from numen.characterization.results import ChirpResult
from numen.characterization.schema import ContinuousChirpSpec

_log = logging.getLogger("numen.characterization.tests.chirp_sweep")


def run_continuous_chirp(
    test: ContinuousChirpSpec,
    exc_spec: Any,
    exc_entity_id: str,
    exc_port_name: str,
    output_state_key: str,
    backend: Any,
) -> ChirpResult:
    """Run a single chirp solve and extract the FRF via cross-spectrum method.

    One solve covers the entire frequency band — much faster than a stepped sine
    but with lower frequency resolution and SNR.  Use as a quick survey before
    running a targeted DiscreteFrequencySweep around the resonance.

    The FRF is extracted by:
      1. Reconstructing the known input chirp signal from the test parameters.
      2. Computing H(f) = S_xy(f) / S_xx(f) via FFT cross-spectrum.

    Args:
        test:             Validated ContinuousChirpSpec.
        exc_spec:         CompiledSpec with excitation injected (sinusoidal).
                          The chirp system will be injected on top of this spec
                          (the placeholder sinusoidal excitation remains but its
                          amp defaults to 0, so it contributes nothing).
        exc_entity_id:    Entity owning the ExcitationPort.
        exc_port_name:    ExcitationPort field name.
        output_state_key: Dot-key for the state to measure.
        backend:          Open solver backend.

    Returns:
        ChirpResult with raw time series and cross-spectrum FRF estimate.
    """
    # Look up the ExcitationPort target field from the param map
    chirp_prefix = f"_exc_{exc_entity_id}_{exc_port_name}"
    target_field  = _find_target_field(exc_spec, exc_entity_id, chirp_prefix)

    # Read effective DC from the sinusoidal excitation parameters already in exc_spec.
    # This honours whatever the outer sweep set (or what the runner pre-applied for
    # standalone tests) rather than unconditionally using test.dc_offset.
    _dc_key = f"{chirp_prefix}.dc"
    if _dc_key in exc_spec.param_index_map:
        effective_dc = float(exc_spec.p[exc_spec.param_index_map[_dc_key][0]])
    else:
        effective_dc = test.dc_offset

    # Inject chirp excitation on top of the existing (zero-amplitude) sinusoidal system
    spec_chirp = inject_chirp_excitation(
        exc_spec,
        entity_id  = exc_entity_id,
        port_name  = exc_port_name,
        target_field = target_field,
        amp        = test.amplitude,
        f_start    = test.f_start,
        f_end      = test.f_end,
        duration   = test.duration,
        dc         = effective_dc,
        chirp_type = test.chirp_type,
    )

    result = backend.solve(spec_chirp, tspan=(0.0, test.duration))

    out_idx = spec_chirp.state_idx(output_state_key)
    output  = np.asarray(result.x[out_idx])
    t       = np.asarray(result.t)

    freqs, H_mag, H_phase_deg = analyze_chirp_frf(
        t, output,
        f_start    = test.f_start,
        f_end      = test.f_end,
        duration   = test.duration,
        amplitude  = test.amplitude,
        chirp_type = test.chirp_type,
    )

    _log.info(
        "Chirp '%s' done: f=[%.2f, %.2f] Hz  n_freq=%d  peak|H|=%.4f",
        test.name, test.f_start, test.f_end,
        len(freqs), float(np.max(H_mag)) if len(H_mag) else 0.0,
    )

    return ChirpResult(
        name        = test.name,
        t           = t,
        output      = output,
        frequencies = freqs,
        H_mag       = H_mag,
        H_phase_deg = H_phase_deg,
        f_start     = test.f_start,
        f_end       = test.f_end,
        amplitude   = test.amplitude,
        dc_offset   = test.dc_offset,
        chirp_type  = test.chirp_type,
    )


def _find_target_field(spec: Any, entity_id: str, exc_prefix: str) -> str:
    """Infer the target state field name from the injected excitation system.

    The chirp system needs to write to the same target field as the sinusoidal
    excitation that was already injected.  We recover the field by inspecting
    the dynamics closure of the existing excitation system.

    Falls back to looking for ``{entity_id}.velocity`` then ``{entity_id}.position``
    if the closure inspection fails.
    """
    # Try to find the target key from the excitation system's closure
    for sys in spec.systems:
        if (
            hasattr(sys, "entity_ids")
            and sys.entity_ids
            and sys.entity_ids[0] == exc_prefix
            and sys.python_fn is not None
        ):
            fn = sys.python_fn
            if hasattr(fn, "__code__"):
                for const in fn.__code__.co_consts:
                    if isinstance(const, str) and const.startswith(entity_id + "."):
                        return const.split(".", 1)[1]

    # Heuristic fallback — try common field names
    for candidate in ("velocity", "position", "angle", "flow"):
        key = f"{entity_id}.{candidate}"
        if key in spec.state_index_map:
            return candidate

    raise RuntimeError(
        f"Could not determine chirp target field for entity '{entity_id}'. "
        f"Pass target_field explicitly to inject_chirp_excitation()."
    )
