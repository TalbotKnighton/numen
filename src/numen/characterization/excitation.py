"""Excitation injection — post-compilation forcing for characterization campaigns.

Usage::

    spec  = compile_spec(world)
    ports = find_excitation_ports(world, "osc")
    # ports == {"force": ExcitationPortInfo(component_kind="nl_oscillator",
    #                                        port=ExcitationPort(targets="velocity", ...))}

    port_info = ports["force"]
    spec = inject_excitation(spec, entity_id="osc",
                             component_kind=port_info.component_kind,
                             port_name="force",
                             target_field=port_info.targets,
                             amp=0.01, freq=1.0, dc=0.0)
    result = backend.solve(spec, tspan=(0.0, 30.0))
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, get_args, get_origin, get_type_hints

try:
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    import numpy as jnp   # type: ignore[no-redef]
    _HAS_JAX = False

import numpy as np

from numen.compiler.flatten import CompiledSpec, CompiledSystem
from numen.fields import ExcitationPort


@dataclass(frozen=True)
class ExcitationPortInfo:
    """Port metadata returned by find_excitation_ports."""
    component_kind: str
    port: ExcitationPort

    @property
    def targets(self) -> str:
        return self.port.targets

    @property
    def port_type(self) -> str:
        return self.port.port_type

    @property
    def units(self) -> str:
        return self.port.units


def find_excitation_ports(
    world: Any,
    entity_id: str,
    component_kind: str | None = None,
) -> dict[str, ExcitationPortInfo]:
    """Return {port_name: ExcitationPortInfo} for all ExcitationPort fields on entity's components.

    Args:
        component_kind: If given, only scan the component with this kind.  Use this to
                        disambiguate entities that have multiple components with the same
                        port field name.

    Raises:
        KeyError:   entity_id not in world.components.
        ValueError: component_kind given but not present on the entity, or the same port
                    name appears on multiple components and component_kind is not specified.
    """
    import typing
    comps: dict[str, Any] = world.components.get(entity_id)
    if comps is None:
        raise KeyError(f"Entity '{entity_id}' not found in world")

    if component_kind is not None:
        if component_kind not in comps:
            raise ValueError(
                f"find_excitation_ports: entity '{entity_id}' has no component with kind "
                f"'{component_kind}'. Available kinds: {list(comps)}"
            )
        components_to_scan: list[tuple[str, Any]] = [(component_kind, comps[component_kind])]
    else:
        components_to_scan = list(comps.items())

    ports: dict[str, ExcitationPortInfo] = {}
    for kind, component in components_to_scan:
        hints = get_type_hints(type(component), include_extras=True)
        for field_name, hint in hints.items():
            if get_origin(hint) is not typing.Annotated:
                continue
            for meta in get_args(hint)[1:]:
                if isinstance(meta, ExcitationPort):
                    if field_name in ports:
                        raise ValueError(
                            f"find_excitation_ports: port '{field_name}' is ambiguous on "
                            f"entity '{entity_id}' — found on both "
                            f"'{ports[field_name].component_kind}' and '{kind}'. "
                            f"Add 'component: <kind>' to the excitation: block."
                        )
                    ports[field_name] = ExcitationPortInfo(component_kind=kind, port=meta)
    return ports


def inject_excitation(
    spec: CompiledSpec,
    entity_id: str,
    component_kind: str,
    port_name: str,
    target_field: str,
    amp: float = 0.0,
    freq: float = 1.0,
    dc: float = 0.0,
) -> CompiledSpec:
    """Return a new CompiledSpec with a sinusoidal forcing system added.

    Injects F(t) = amp·sin(2π·freq·t) + dc into d(target_field)/dt for the
    given entity.  The three scalars (amp, freq, dc) are appended to the
    parameter vector under a synthetic entity key ``_exc_{entity_id}_{port_name}``.

    This is a pure function — the original spec is not modified.

    Args:
        spec:           Compiled spec to extend.
        entity_id:      Entity that owns the ExcitationPort, e.g. "osc".
        component_kind: Component kind that owns the port, e.g. "nl_oscillator".
        port_name:      Name of the ExcitationPort field, e.g. "force".
        target_field:   IntegratedField whose derivative receives F(t), e.g. "velocity".
        amp:            Sine amplitude.
        freq:           Sine frequency [Hz].
        dc:             DC offset added to the force.

    Returns:
        New CompiledSpec with the excitation system appended.
    """
    exc_eid    = f"_exc_{entity_id}_{port_name}"
    target_key = f"{entity_id}.{component_kind}.{target_field}"

    if target_key not in spec.state_index_map:
        available = [k for k in spec.state_index_map if k.startswith(f"{entity_id}.")]
        raise KeyError(
            f"inject_excitation: '{target_key}' not found in state_index_map. "
            f"Available fields for '{entity_id}': {available}"
        )

    # Extend the parameter map with amp / freq / dc / target_idx.
    # target_idx stores the 0-based state index of the target field as a float so
    # the Julia NumenCharacterization.excitation_dynamics! can look it up without
    # needing a closure (Julia named functions can't capture runtime state).
    target_idx_0 = float(spec.state_index_map[target_key][0])

    new_param_map = dict(spec.param_index_map)
    new_p         = list(spec.p)
    for name, val in [("amp", amp), ("freq", freq), ("dc", dc), ("target_idx", target_idx_0)]:
        key   = f"{exc_eid}.{name}"
        start = len(new_p)
        new_param_map[key] = (start, start + 1)
        new_p.append(val)

    _exc_eid    = exc_eid
    _target_key = target_key

    def _excitation_dynamics(dx, x, p, t, spec_inner, system):
        amp_i  = spec_inner.param_idx(f"{_exc_eid}.amp")
        freq_i = spec_inner.param_idx(f"{_exc_eid}.freq")
        dc_i   = spec_inner.param_idx(f"{_exc_eid}.dc")
        F = p[amp_i] * jnp.sin(2.0 * jnp.pi * p[freq_i] * t) + p[dc_i]
        tgt_i = spec_inner.state_idx(_target_key)
        dx[tgt_i] = dx[tgt_i] + F

    exc_system = CompiledSystem(
        dynamics_fn   = "NumenCharacterization.excitation_dynamics!",
        entity_ids    = [exc_eid],
        group_size    = 1,
        entity_groups = ((exc_eid,),),
        python_fn     = _excitation_dynamics,
    )

    return replace(
        spec,
        param_size      = len(new_p),
        param_index_map = new_param_map,
        p               = new_p,
        systems         = spec.systems + [exc_system],
    )


def inject_chirp_excitation(
    spec: CompiledSpec,
    entity_id: str,
    component_kind: str,
    port_name: str,
    target_field: str,
    amp: float = 1.0,
    f_start: float = 1.0,
    f_end: float = 10.0,
    duration: float = 10.0,
    dc: float = 0.0,
    chirp_type: str = "log",
) -> CompiledSpec:
    """Return a new CompiledSpec with a frequency-swept (chirp) forcing system added.

    Injects F(t) = amp·sin(φ(t)) + dc into d(target_field)/dt, where φ(t) is the
    chirp phase accumulated from f_start to f_end over [0, duration].

    Parameters are stored in the param vector under the synthetic key
    ``_chirp_{entity_id}_{port_name}.{amp, f_start, f_end, duration, dc}``.

    Args:
        chirp_type: "log" (geometric sweep) or "linear" (constant rate sweep).
    """
    exc_eid    = f"_chirp_{entity_id}_{port_name}"
    target_key = f"{entity_id}.{component_kind}.{target_field}"

    if target_key not in spec.state_index_map:
        available = [k for k in spec.state_index_map if k.startswith(f"{entity_id}.")]
        raise KeyError(
            f"inject_chirp_excitation: '{target_key}' not found in state_index_map. "
            f"Available fields for '{entity_id}': {available}"
        )

    # target_idx and chirp_type_flag let Julia's chirp_dynamics! find the right
    # state slot and select the sweep formula without needing a closure.
    # chirp_type_flag: 0.0 = log sweep, 1.0 = linear sweep.
    target_idx_0    = float(spec.state_index_map[target_key][0])
    chirp_type_flag = 1.0 if chirp_type == "linear" else 0.0

    new_param_map = dict(spec.param_index_map)
    new_p         = list(spec.p)
    for name, val in [("amp", amp), ("f_start", f_start), ("f_end", f_end),
                      ("duration", duration), ("dc", dc),
                      ("target_idx", target_idx_0), ("chirp_type_flag", chirp_type_flag)]:
        key   = f"{exc_eid}.{name}"
        start = len(new_p)
        new_param_map[key] = (start, start + 1)
        new_p.append(val)

    _exc_eid    = exc_eid
    _target_key = target_key

    if chirp_type == "log":
        def _chirp_dynamics(dx, x, p, t, spec_inner, system):
            amp_i  = spec_inner.param_idx(f"{_exc_eid}.amp")
            fs_i   = spec_inner.param_idx(f"{_exc_eid}.f_start")
            fe_i   = spec_inner.param_idx(f"{_exc_eid}.f_end")
            dur_i  = spec_inner.param_idx(f"{_exc_eid}.duration")
            dc_i   = spec_inner.param_idx(f"{_exc_eid}.dc")
            f_s = p[fs_i]; f_e = p[fe_i]; dur = p[dur_i]
            k     = jnp.log(f_e / f_s)
            phase = 2.0 * jnp.pi * f_s * dur / k * (jnp.exp(k * t / dur) - 1.0)
            F     = p[amp_i] * jnp.sin(phase) + p[dc_i]
            tgt_i = spec_inner.state_idx(_target_key)
            dx[tgt_i] = dx[tgt_i] + F
    else:  # linear
        def _chirp_dynamics(dx, x, p, t, spec_inner, system):
            amp_i  = spec_inner.param_idx(f"{_exc_eid}.amp")
            fs_i   = spec_inner.param_idx(f"{_exc_eid}.f_start")
            fe_i   = spec_inner.param_idx(f"{_exc_eid}.f_end")
            dur_i  = spec_inner.param_idx(f"{_exc_eid}.duration")
            dc_i   = spec_inner.param_idx(f"{_exc_eid}.dc")
            f_s = p[fs_i]; f_e = p[fe_i]; dur = p[dur_i]
            phase = 2.0 * jnp.pi * (f_s * t + (f_e - f_s) * t**2 / (2.0 * dur))
            F     = p[amp_i] * jnp.sin(phase) + p[dc_i]
            tgt_i = spec_inner.state_idx(_target_key)
            dx[tgt_i] = dx[tgt_i] + F

    chirp_system = CompiledSystem(
        dynamics_fn   = "NumenCharacterization.chirp_dynamics!",
        entity_ids    = [exc_eid],
        group_size    = 1,
        entity_groups = ((exc_eid,),),
        python_fn     = _chirp_dynamics,
    )

    return replace(
        spec,
        param_size      = len(new_p),
        param_index_map = new_param_map,
        p               = new_p,
        systems         = spec.systems + [chirp_system],
    )


def set_chirp_params(
    spec: CompiledSpec,
    entity_id: str,
    port_name: str,
    amp: float | None = None,
    f_start: float | None = None,
    f_end: float | None = None,
    duration: float | None = None,
    dc: float | None = None,
) -> CompiledSpec:
    """Return a new spec with updated chirp excitation parameters.

    The chirp system must already have been injected via inject_chirp_excitation().
    """
    exc_eid = f"_chirp_{entity_id}_{port_name}"
    new_p   = list(spec.p)
    for name, val in [("amp", amp), ("f_start", f_start), ("f_end", f_end),
                      ("duration", duration), ("dc", dc)]:
        if val is None:
            continue
        key = f"{exc_eid}.{name}"
        if key not in spec.param_index_map:
            raise KeyError(
                f"set_chirp_params: '{key}' not in param_index_map. "
                f"Did you call inject_chirp_excitation() first?"
            )
        idx = spec.param_index_map[key][0]
        new_p[idx] = val
    return replace(spec, p=new_p)


def set_excitation_params(
    spec: CompiledSpec,
    entity_id: str,
    port_name: str,
    amp: float | None = None,
    freq: float | None = None,
    dc: float | None = None,
) -> CompiledSpec:
    """Return a new spec with updated excitation amplitude / frequency / DC.

    Useful in the inner loop of a frequency sweep — cheaper than calling
    inject_excitation() from scratch because it only touches p.

    The excitation system must already have been injected via inject_excitation().
    """
    exc_eid = f"_exc_{entity_id}_{port_name}"
    new_p = list(spec.p)
    for name, val in [("amp", amp), ("freq", freq), ("dc", dc)]:
        if val is None:
            continue
        key = f"{exc_eid}.{name}"
        if key not in spec.param_index_map:
            raise KeyError(
                f"set_excitation_params: '{key}' not in param_index_map. "
                f"Did you call inject_excitation() first?"
            )
        idx = spec.param_index_map[key][0]
        new_p[idx] = val
    return replace(spec, p=new_p)


def inject_table_excitation(
    spec: CompiledSpec,
    entity_id: str,
    component_kind: str,
    port_name: str,
    target_field: str,
    signal: "np.ndarray",
    dt_sig: float,
    t_start: float = 0.0,
    dc: float = 0.0,
) -> CompiledSpec:
    """Return a new CompiledSpec with a pre-computed table-lookup forcing system added.

    The signal array is stored in the parameter vector.  The ODE right-hand
    side performs linear interpolation at each evaluation — no randomness
    during integration.

    Parameter layout appended to ``p``::

        _table_{entity_id}_{port_name}.n_samples   → float(N)
        _table_{entity_id}_{port_name}.t_start     → t_start [s]
        _table_{entity_id}_{port_name}.dt_sig      → dt_sig [s]
        _table_{entity_id}_{port_name}.dc          → DC offset added to table values
        _table_{entity_id}_{port_name}.target_idx  → state index (float, 0-based)
        _table_{entity_id}_{port_name}.signal      → signal[0..N-1]  (slice in p)

    Note:
        JAX backends are not supported for table excitation because the
        variable-length parameter layout requires retracing for each signal
        length.  Use scipy or julia_server backends.

    Args:
        spec:           Compiled spec to extend.
        entity_id:      Entity that owns the ExcitationPort, e.g. ``"osc"``.
        component_kind: Component kind that owns the port, e.g. ``"nl_oscillator"``.
        port_name:      Name of the ExcitationPort field, e.g. ``"force"``.
        target_field:   IntegratedField whose derivative receives F(t), e.g. ``"velocity"``.
        signal:         Pre-computed forcing time series, shape ``(N,)``.
        dt_sig:         Uniform sample period of ``signal`` [s].
        t_start:        Time at which the first sample applies.  Clamped at boundaries.
        dc:             Constant offset added to the interpolated table value.

    Returns:
        New CompiledSpec with the table excitation system appended.
    """
    exc_eid    = f"_table_{entity_id}_{port_name}"
    target_key = f"{entity_id}.{component_kind}.{target_field}"

    if target_key not in spec.state_index_map:
        available = [k for k in spec.state_index_map if k.startswith(f"{entity_id}.")]
        raise KeyError(
            f"inject_table_excitation: '{target_key}' not found in state_index_map. "
            f"Available fields for '{entity_id}': {available}"
        )

    signal_arr   = np.asarray(signal, dtype=np.float64).ravel()
    N            = len(signal_arr)
    target_idx_0 = float(spec.state_index_map[target_key][0])

    new_param_map = dict(spec.param_index_map)
    new_p         = list(spec.p)

    for name, val in [
        ("n_samples",  float(N)),
        ("t_start",    float(t_start)),
        ("dt_sig",     float(dt_sig)),
        ("dc",         float(dc)),
        ("target_idx", target_idx_0),
    ]:
        key   = f"{exc_eid}.{name}"
        start = len(new_p)
        new_param_map[key] = (start, start + 1)
        new_p.append(val)

    # Signal slice in p
    sig_start = len(new_p)
    sig_end   = sig_start + N
    sig_key   = f"{exc_eid}.signal"
    new_param_map[sig_key] = (sig_start, sig_end)
    new_p.extend(signal_arr.tolist())

    # Python dynamics closure — np.interp for scipy backend
    _exc_eid    = exc_eid
    _target_key = target_key

    def _table_dynamics(dx, x, p, t, spec_inner, system):
        n     = int(p[spec_inner.param_idx(f"{_exc_eid}.n_samples")])
        t0    = p[spec_inner.param_idx(f"{_exc_eid}.t_start")]
        dt    = p[spec_inner.param_idx(f"{_exc_eid}.dt_sig")]
        dc_v  = p[spec_inner.param_idx(f"{_exc_eid}.dc")]
        tgt_i = spec_inner.state_idx(_target_key)
        s_start, s_end = spec_inner.param_index_map[f"{_exc_eid}.signal"]
        sig_p = np.asarray(p[s_start:s_end])
        t_arr = t0 + np.arange(n) * dt
        F     = float(np.interp(float(t), t_arr, sig_p)) + dc_v
        dx[tgt_i] = dx[tgt_i] + F

    table_system = CompiledSystem(
        dynamics_fn   = "NumenCharacterization.table_excitation_dynamics!",
        entity_ids    = [exc_eid],
        group_size    = 1,
        entity_groups = ((exc_eid,),),
        python_fn     = _table_dynamics,
    )

    return replace(
        spec,
        param_size      = len(new_p),
        param_index_map = new_param_map,
        p               = new_p,
        systems         = spec.systems + [table_system],
    )
