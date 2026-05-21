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

import inspect
import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, get_args, get_origin, get_type_hints

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


def _resolve_scale_by_idx(spec: CompiledSpec, scale_by: str | None, helper_name: str) -> float:
    """Resolve scale_by key to a 0-based param index, or -1.0 sentinel if None.

    Returned as a Float64 so it can be packed into the parameter vector alongside
    other excitation parameters.  The Julia/Python dynamics use the sentinel to
    skip the division when no scaling is requested.
    """
    if scale_by is None:
        return -1.0
    if scale_by not in spec.param_index_map:
        available = list(spec.param_index_map)[:10]
        raise KeyError(
            f"{helper_name}: scale_by='{scale_by}' not found in param_index_map. "
            f"Expected full path 'entity.component.field'. "
            f"First few available params: {available}"
        )
    return float(spec.param_index_map[scale_by][0])


def inject_excitation(
    spec: CompiledSpec,
    entity_id: str,
    component_kind: str,
    port_name: str,
    target_field: str,
    amp: float = 0.0,
    freq: float = 1.0,
    dc: float = 0.0,
    scale_by: str | None = None,
) -> CompiledSpec:
    """Return a new CompiledSpec with a sinusoidal forcing system added.

    Injects F(t) = amp·sin(2π·freq·t) + dc into d(target_field)/dt for the
    given entity.  The three scalars (amp, freq, dc) are appended to the
    parameter vector under a synthetic entity key ``_exc_{entity_id}_{port_name}``.

    The framework is unit-agnostic: ``ExcitationPort.port_type`` and ``units``
    are metadata for plot labels only — they do NOT affect injection math.
    F(t) is added directly to d(target_field)/dt with no automatic conversion.
    If you need to convert a "force" amplitude into an acceleration
    (i.e. divide by mass before adding to dv/dt), set ``scale_by`` to the
    full path of the divisor parameter.

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
        scale_by:       Optional full path "entity.component.field" of a
                        ``ParameterField`` to **divide** the computed excitation
                        by before adding to the state derivative.  Use this to
                        convert force [N] into acceleration [m/s²] when the
                        target is a velocity-like state, by setting
                        ``scale_by="osc.mass_body.mass"`` (or similar).
                        When ``None``, no division is applied.

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

    # Extend the parameter map with amp / freq / dc / target_idx / scale_by_idx.
    # target_idx and scale_by_idx store 0-based parameter/state indices as floats so
    # the Julia NumenCharacterization.excitation_dynamics! can look them up without
    # needing a closure.  scale_by_idx = -1.0 is the "no scaling" sentinel.
    target_idx_0   = float(spec.state_index_map[target_key][0])
    scale_by_idx_0 = _resolve_scale_by_idx(spec, scale_by, "inject_excitation")

    new_param_map = dict(spec.param_index_map)
    new_p         = list(spec.p)
    for name, val in [("amp", amp), ("freq", freq), ("dc", dc),
                      ("target_idx", target_idx_0),
                      ("scale_by_idx", scale_by_idx_0)]:
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
        scale_i = spec_inner.param_idx(f"{_exc_eid}.scale_by_idx")
        F = p[amp_i] * jnp.sin(2.0 * jnp.pi * p[freq_i] * t) + p[dc_i]
        scale_idx = int(round(float(p[scale_i])))
        if scale_idx >= 0:
            F = F / p[scale_idx]
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
    scale_by: str | None = None,
) -> CompiledSpec:
    """Return a new CompiledSpec with a frequency-swept (chirp) forcing system added.

    Injects F(t) = amp·sin(φ(t)) + dc into d(target_field)/dt, where φ(t) is the
    chirp phase accumulated from f_start to f_end over [0, duration].

    Parameters are stored in the param vector under the synthetic key
    ``_chirp_{entity_id}_{port_name}.{amp, f_start, f_end, duration, dc}``.

    See ``inject_excitation`` for a description of the unit-agnostic injection
    semantics and the ``scale_by`` divisor.

    Args:
        chirp_type: "log" (geometric sweep) or "linear" (constant rate sweep).
        scale_by:   Optional full path "entity.component.field" of a parameter
                    to **divide** the computed chirp value by before adding to
                    the state derivative.  See ``inject_excitation`` docstring.
    """
    exc_eid    = f"_chirp_{entity_id}_{port_name}"
    target_key = f"{entity_id}.{component_kind}.{target_field}"

    if target_key not in spec.state_index_map:
        available = [k for k in spec.state_index_map if k.startswith(f"{entity_id}.")]
        raise KeyError(
            f"inject_chirp_excitation: '{target_key}' not found in state_index_map. "
            f"Available fields for '{entity_id}': {available}"
        )

    # target_idx, chirp_type_flag, and scale_by_idx let Julia's chirp_dynamics!
    # find the right state slot, select the sweep formula, and decide whether to
    # divide — all without needing a closure.
    # chirp_type_flag: 0.0 = log sweep, 1.0 = linear sweep.
    # scale_by_idx:    -1.0 = no scaling, otherwise 0-based param index of the divisor.
    target_idx_0    = float(spec.state_index_map[target_key][0])
    chirp_type_flag = 1.0 if chirp_type == "linear" else 0.0
    scale_by_idx_0  = _resolve_scale_by_idx(spec, scale_by, "inject_chirp_excitation")

    new_param_map = dict(spec.param_index_map)
    new_p         = list(spec.p)
    for name, val in [("amp", amp), ("f_start", f_start), ("f_end", f_end),
                      ("duration", duration), ("dc", dc),
                      ("target_idx", target_idx_0), ("chirp_type_flag", chirp_type_flag),
                      ("scale_by_idx", scale_by_idx_0)]:
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
            scale_i = spec_inner.param_idx(f"{_exc_eid}.scale_by_idx")
            f_s = p[fs_i]; f_e = p[fe_i]; dur = p[dur_i]
            k     = jnp.log(f_e / f_s)
            phase = 2.0 * jnp.pi * f_s * dur / k * (jnp.exp(k * t / dur) - 1.0)
            F     = p[amp_i] * jnp.sin(phase) + p[dc_i]
            scale_idx = int(round(float(p[scale_i])))
            if scale_idx >= 0:
                F = F / p[scale_idx]
            tgt_i = spec_inner.state_idx(_target_key)
            dx[tgt_i] = dx[tgt_i] + F
    else:  # linear
        def _chirp_dynamics(dx, x, p, t, spec_inner, system):
            amp_i  = spec_inner.param_idx(f"{_exc_eid}.amp")
            fs_i   = spec_inner.param_idx(f"{_exc_eid}.f_start")
            fe_i   = spec_inner.param_idx(f"{_exc_eid}.f_end")
            dur_i  = spec_inner.param_idx(f"{_exc_eid}.duration")
            dc_i   = spec_inner.param_idx(f"{_exc_eid}.dc")
            scale_i = spec_inner.param_idx(f"{_exc_eid}.scale_by_idx")
            f_s = p[fs_i]; f_e = p[fe_i]; dur = p[dur_i]
            phase = 2.0 * jnp.pi * (f_s * t + (f_e - f_s) * t**2 / (2.0 * dur))
            F     = p[amp_i] * jnp.sin(phase) + p[dc_i]
            scale_idx = int(round(float(p[scale_i])))
            if scale_idx >= 0:
                F = F / p[scale_idx]
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
    scale_by: str | None = None,
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

    signal_arr     = np.asarray(signal, dtype=np.float64).ravel()
    N              = len(signal_arr)
    target_idx_0   = float(spec.state_index_map[target_key][0])
    scale_by_idx_0 = _resolve_scale_by_idx(spec, scale_by, "inject_table_excitation")

    new_param_map = dict(spec.param_index_map)
    new_p         = list(spec.p)

    for name, val in [
        ("n_samples",    float(N)),
        ("t_start",      float(t_start)),
        ("dt_sig",       float(dt_sig)),
        ("dc",           float(dc)),
        ("target_idx",   target_idx_0),
        ("scale_by_idx", scale_by_idx_0),
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
        scale_i = spec_inner.param_idx(f"{_exc_eid}.scale_by_idx")
        tgt_i = spec_inner.state_idx(_target_key)
        s_start, s_end = spec_inner.param_index_map[f"{_exc_eid}.signal"]
        sig_p = np.asarray(p[s_start:s_end])
        t_arr = t0 + np.arange(n) * dt
        F     = float(np.interp(float(t), t_arr, sig_p)) + dc_v
        scale_idx = int(round(float(p[scale_i])))
        if scale_idx >= 0:
            F = F / p[scale_idx]
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


def inject_custom_excitation(
    spec: CompiledSpec,
    entity_id: str,
    component_kind: str,
    port_name: str,
    target_field: str,
    params: dict[str, float],
    python_fn: Callable[..., float],
    julia_fn: str,
    scale_by: str | None = None,
) -> CompiledSpec:
    """Return a new CompiledSpec with a user-defined excitation system added.

    Lets you inject an arbitrary forcing F(t, *params) without having to
    hand-construct a synthetic entity + CompiledSystem.  Sweep parameters
    work for free because all params live in the parameter vector ``p``
    (sweepable via ``excitation.<param_name>`` paths).

    **Function contract.**
    ``python_fn`` must take ``t`` as its first positional argument followed
    by one positional argument per entry in ``params``, in the order the
    dict was provided.  Example::

        def my_gate(t, amp, freq, t_on, t_off):
            if t_on <= t < t_off:
                return amp * math.sin(2 * math.pi * freq * t)
            return 0.0

        params = {"amp": 4.9, "freq": 1250.0, "t_on": 0.1, "t_off": 0.4}

    The helper inspects ``python_fn``'s signature at injection time and
    raises ``ValueError`` if the argument names don't match ``("t",) +
    tuple(params.keys())``.

    **Julia side.**
    For ``julia_file``-backed solves you must define ``julia_fn`` in your
    ``dynamics.jl`` file.  The simplest pattern uses the helper
    ``NumenCharacterization.make_custom_excitation_dyn``::

        module MyDyn
        # Human-readable function — same shape as python_fn
        function my_gate(t, amp, freq, t_on, t_off)
            (t < t_on || t >= t_off) && return 0.0
            return amp * sin(2π * freq * t)
        end

        # One-line wrapper — this is what `julia_fn` points to
        const my_gate_dyn! = Main.NumenCharacterization.make_custom_excitation_dyn(
            my_gate, ("amp", "freq", "t_on", "t_off"),
        )
        end

    Then ``julia_fn = "MyDyn.my_gate_dyn!"``.

    **Unit-agnostic.**  Like all ``inject_*`` helpers, the value returned by
    ``python_fn`` / ``julia_fn`` is added directly to ``d(target_field)/dt``.
    Pass ``scale_by="entity.component.field"`` to **divide** the returned
    value by a parameter before adding (e.g. convert force [N] into
    acceleration [m/s²] by dividing by mass).

    Args:
        spec:           Compiled spec to extend.
        entity_id:      Entity that owns the ExcitationPort.
        component_kind: Component kind that owns the port.
        port_name:      ExcitationPort field name.
        target_field:   IntegratedField whose derivative receives F.
        params:         Initial values for the user function's params.  Dict
                        insertion order determines the positional argument
                        order both ``python_fn`` and ``julia_fn`` are called with.
        python_fn:      Python callable ``(t, *params) -> float``.  Used by
                        scipy backend.  Function signature must match
                        ``("t",) + tuple(params.keys())``.
        julia_fn:       Fully-qualified name ``"Module.function!"`` of the Julia
                        dynamics function in the user's ``dynamics.jl``.  Used by
                        Julia backends.  Required even for scipy-only campaigns
                        (will be validated lazily by the Julia backend).
        scale_by:       Optional ``"entity.component.field"`` path of a
                        ParameterField to **divide** F by before adding to dx.

    Returns:
        New CompiledSpec with the custom excitation system appended.
    """
    if not callable(python_fn):
        raise TypeError(
            f"inject_custom_excitation: python_fn must be callable, got {type(python_fn).__name__}"
        )

    # Validate the Python signature matches ("t",) + params.keys()
    sig = inspect.signature(python_fn)
    sig_names = [
        name for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.POSITIONAL_ONLY)
    ]
    expected = ["t"] + list(params.keys())
    if sig_names != expected:
        raise ValueError(
            f"inject_custom_excitation: python_fn signature {sig_names} does not match "
            f"expected {expected}.  The function must accept ``t`` followed by one "
            f"positional argument per params dict entry, in order."
        )

    exc_eid    = f"_custom_{entity_id}_{port_name}"
    target_key = f"{entity_id}.{component_kind}.{target_field}"

    if target_key not in spec.state_index_map:
        available = [k for k in spec.state_index_map if k.startswith(f"{entity_id}.")]
        raise KeyError(
            f"inject_custom_excitation: '{target_key}' not found in state_index_map. "
            f"Available fields for '{entity_id}': {available}"
        )

    target_idx_0   = float(spec.state_index_map[target_key][0])
    scale_by_idx_0 = _resolve_scale_by_idx(spec, scale_by, "inject_custom_excitation")
    user_param_order = list(params.keys())

    new_param_map = dict(spec.param_index_map)
    new_p         = list(spec.p)

    # Pack user params first (in the dict's insertion order)
    for name in user_param_order:
        key   = f"{exc_eid}.{name}"
        start = len(new_p)
        new_param_map[key] = (start, start + 1)
        new_p.append(float(params[name]))

    # Then meta params
    for name, val in [("target_idx", target_idx_0),
                      ("scale_by_idx", scale_by_idx_0)]:
        key   = f"{exc_eid}.{name}"
        start = len(new_p)
        new_param_map[key] = (start, start + 1)
        new_p.append(val)

    _exc_eid     = exc_eid
    _target_key  = target_key
    _user_fn     = python_fn
    _param_names = tuple(user_param_order)

    def _custom_dynamics(dx, x, p, t, spec_inner, system):
        scale_i = spec_inner.param_idx(f"{_exc_eid}.scale_by_idx")
        tgt_i   = spec_inner.state_idx(_target_key)
        param_vals = tuple(
            float(p[spec_inner.param_idx(f"{_exc_eid}.{name}")])
            for name in _param_names
        )
        F = _user_fn(t, *param_vals)
        scale_idx = int(round(float(p[scale_i])))
        if scale_idx >= 0:
            F = F / p[scale_idx]
        dx[tgt_i] = dx[tgt_i] + F

    custom_system = CompiledSystem(
        dynamics_fn   = julia_fn,
        entity_ids    = [exc_eid],
        group_size    = 1,
        entity_groups = ((exc_eid,),),
        python_fn     = _custom_dynamics,
    )

    return replace(
        spec,
        param_size      = len(new_p),
        param_index_map = new_param_map,
        p               = new_p,
        systems         = spec.systems + [custom_system],
    )
