"""Phase portrait (limit cycle) test runner with optional Poincaré section."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from numen.characterization.excitation import set_excitation_params
from numen.characterization.results import PhasePortraitResult
from numen.characterization.schema import PhasePortraitSpec

_log = logging.getLogger("numen.characterization.tests.phase_portrait")


def run_phase_portrait(
    test: PhasePortraitSpec,
    exc_spec: Any,
    exc_entity_id: str,
    exc_port_name: str,
    exc_component_kind: str,
    exc_target_field: str,
    output_state_key: str,
    backend: Any,
) -> PhasePortraitResult:
    """Run to steady state and record the limit cycle in (position, velocity) space.

    Simulates n_transient_cycles + n_record_cycles forcing periods.  The first
    n_transient_cycles are discarded as transient.  The settled portion is
    returned as the phase portrait.

    If ``poincare=True``, the state is interpolated at every multiple of the
    forcing period T within the settled window, giving a stroboscopic Poincaré
    section.  A periodic response produces a single cluster of points; a
    period-doubled response produces two clusters; a chaotic response fills an
    area.

    Args:
        test:               Validated PhasePortraitSpec.
        exc_spec:           CompiledSpec with excitation injected (placeholder params).
        exc_entity_id:      Entity that owns the ExcitationPort.
        exc_port_name:      Name of the ExcitationPort field.
        exc_component_kind: Component kind that owns the ExcitationPort, e.g. "oscillator".
        exc_target_field:   IntegratedField driven by excitation (also the velocity
                            state used as the y-axis), e.g. "velocity".
        output_state_key:   3-part dot-key for the x-axis state,
                            e.g. "osc.oscillator.position".
        backend:            Open solver backend.

    Returns:
        PhasePortraitResult with limit cycle arrays and optional Poincaré dots.
    """
    T         = 1.0 / test.frequency
    t_settle  = test.n_transient_cycles * T
    t_end     = t_settle + test.n_record_cycles * T
    tspan     = (0.0, t_end)

    spec = set_excitation_params(
        exc_spec, exc_entity_id, exc_port_name,
        amp=test.amplitude, freq=test.frequency, dc=test.dc_offset,
    )

    _log.debug(
        "phase_portrait '%s': f=%.4f Hz, A=%.4g, t_settle=%.3f s, t_end=%.3f s",
        test.name, test.frequency, test.amplitude, t_settle, t_end,
    )

    result  = backend.solve(spec, tspan=tspan)
    out_idx = spec.state_idx(output_state_key)
    vel_key = f"{exc_entity_id}.{exc_component_kind}.{exc_target_field}"
    vel_idx = spec.state_idx(vel_key)

    t    = result.t
    x    = result.x[out_idx]
    xdot = result.x[vel_idx]

    # Keep only the settled portion
    mask  = t >= t_settle
    t_rec = t[mask]
    x_rec = x[mask]
    xd_rec = xdot[mask]

    # Poincaré section: interpolate state at integer multiples of T
    poincare_x: np.ndarray | None    = None
    poincare_xdot: np.ndarray | None = None

    if test.poincare and len(t_rec) > 2:
        poincare_times = t_settle + np.arange(1, test.n_record_cycles + 1) * T
        # Clip to the available time range
        poincare_times = poincare_times[poincare_times <= t_rec[-1]]
        if len(poincare_times) > 0:
            poincare_x    = np.interp(poincare_times, t, x)
            poincare_xdot = np.interp(poincare_times, t, xdot)

    _log.info(
        "phase_portrait '%s' done: %d settled points, %d Poincaré samples",
        test.name,
        len(t_rec),
        len(poincare_x) if poincare_x is not None else 0,
    )

    return PhasePortraitResult(
        name          = test.name,
        t             = t_rec,
        x             = x_rec,
        xdot          = xd_rec,
        frequency     = test.frequency,
        amplitude     = test.amplitude,
        poincare_x    = poincare_x,
        poincare_xdot = poincare_xdot,
        dc_offset     = test.dc_offset,
    )
