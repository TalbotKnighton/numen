"""Free-decay ring-down test runner with Hilbert-transform backbone extraction."""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import numpy as np

from numen.characterization.analysis import (
    instantaneous_frequency_amplitude,
    resample_uniform,
)
from numen.characterization.excitation import inject_excitation
from numen.characterization.results import FreeDecayResult
from numen.characterization.schema import FreeDecaySpec

_log = logging.getLogger("numen.characterization.tests.free_decay")


def run_free_decay(
    test: FreeDecaySpec,
    base_spec: Any,
    exc_entity_id: str,
    exc_port_name: str,
    exc_target_field: str,
    output_state_key: str,
    backend: Any,
) -> FreeDecayResult:
    """Ring the system down from a large initial condition and extract the backbone curve.

    Builds a zero-forcing spec from ``base_spec`` (injecting the excitation system
    with amp=0 so the Julia backend loads the excitation module), then overrides
    x0 to set the initial displacement and velocity.  The Hilbert transform of the
    output time series gives instantaneous amplitude and frequency, which together
    trace the backbone curve.

    Args:
        test:             Validated FreeDecaySpec.
        base_spec:        Compiled spec WITHOUT excitation injected (from compile_spec).
        exc_entity_id:    Entity that owns the ExcitationPort.
        exc_port_name:    Name of the ExcitationPort field.
        exc_target_field: IntegratedField whose derivative is driven (also the velocity
                          field used for initial conditions), e.g. "velocity".
        output_state_key: Dot-key for the position state, e.g. "osc.position".
        backend:          Open solver backend.

    Returns:
        FreeDecayResult with time series, envelope, instantaneous frequency and
        damping, and backbone curve arrays.
    """
    # Inject excitation with zero amplitude so the Julia side loads NumenCharacterization
    spec = inject_excitation(
        base_spec, exc_entity_id, exc_port_name, exc_target_field,
        amp=0.0, freq=1.0, dc=0.0,
    )

    # Override initial conditions in x0 (list[float])
    pos_idx = spec.state_idx(output_state_key)
    vel_key = f"{exc_entity_id}.{exc_target_field}"
    vel_idx = spec.state_idx(vel_key)

    x0 = list(spec.x0)
    x0[pos_idx] = test.initial_displacement
    x0[vel_idx] = test.initial_velocity
    spec = replace(spec, x0=x0)

    tspan = (0.0, test.t_end)
    _log.debug(
        "free_decay '%s': x0=%.4g  v0=%.4g  t_end=%.3f s",
        test.name, test.initial_displacement, test.initial_velocity, test.t_end,
    )

    result  = backend.solve(spec, tspan=tspan)
    out_idx = spec.state_idx(output_state_key)
    t_raw   = result.t
    y_raw   = result.x[out_idx]

    # Hilbert transform on a uniform grid (with optional bandpass filter)
    envelope, inst_freq, _ = instantaneous_frequency_amplitude(
        t_raw, y_raw,
        f_bandpass_low  = test.bandpass_low,
        f_bandpass_high = test.bandpass_high,
    )

    # Resample time and raw signal to the same uniform grid used internally
    t_u, y_u = resample_uniform(t_raw, y_raw)

    # Instantaneous damping: log-decrement estimate
    # ζ ≈ -d(ln A)/dt / (2π·f_inst)
    # Use finite differences on log-envelope, smoothed by the uniform grid
    log_env = np.log(np.maximum(envelope, 1e-30))
    d_log_env = np.gradient(log_env, t_u)
    inst_damping = np.where(
        inst_freq > 0,
        -d_log_env / (2.0 * np.pi * np.maximum(inst_freq, 1e-6)),
        0.0,
    )
    # Clip to physically meaningful range [0, 1)
    inst_damping = np.clip(inst_damping, 0.0, 0.99)

    # Backbone curve: sort by decreasing envelope, keep points above 1% of peak
    amp_floor = 0.01 * float(envelope[0]) if len(envelope) > 0 else 0.0
    valid     = envelope > amp_floor
    env_valid = envelope[valid]
    freq_valid = inst_freq[valid]
    damp_valid = inst_damping[valid]

    # Sort by decreasing amplitude for a clean backbone trace
    sort_idx           = np.argsort(env_valid)[::-1]
    backbone_amplitude = env_valid[sort_idx]
    backbone_frequency = freq_valid[sort_idx]

    _log.info(
        "free_decay '%s' done: f0≈%.3f Hz at peak, ζ≈%.4f at peak",
        test.name,
        float(backbone_frequency[0]) if len(backbone_frequency) > 0 else float("nan"),
        float(damp_valid[sort_idx[0]]) if len(sort_idx) > 0 else float("nan"),
    )

    return FreeDecayResult(
        name                 = test.name,
        t                    = t_u,
        x                    = y_u,
        envelope             = envelope,
        inst_freq_hz         = inst_freq,
        inst_damping         = inst_damping,
        backbone_amplitude   = backbone_amplitude,
        backbone_frequency   = backbone_frequency,
        initial_displacement = test.initial_displacement,
        initial_velocity     = test.initial_velocity,
    )
