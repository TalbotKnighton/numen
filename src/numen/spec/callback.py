from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from pydantic import BaseModel

if TYPE_CHECKING:
    import numpy as np
    from numen.compiler.flatten import CompiledSpec


class ControlFn:
    """Protocol for Python-side controller callbacks (scipy backend).

    Signature: (t, x, p, spec) -> dict[str, Any]

    Return a flat dict mapping "entity_id.field_name" to new values.
    Only DiscreteField slots should be updated; writing to IntegratedField
    slots is legal but unusual (it is a hard state reset, not a derivative).

    Example::

        def my_controller(t, x, p, spec):
            err = x[spec.state_idx("sensor.angle")] - p[spec.param_idx("ctrl.setpoint")]
            return {"actuator.force": p[spec.param_idx("ctrl.kp")] * err}
    """

    def __call__(
        self,
        t: float,
        x: "np.ndarray",
        p: "np.ndarray",
        spec: "CompiledSpec",
    ) -> "dict[str, Any]": ...


class Callback(BaseModel):
    """Base class for controller / event callbacks.

    Callbacks fire at fixed intervals (``dt``) and can read and write state.
    Scipy fires them between solver segments; Julia fires them inside the
    integrator via ``PeriodicCallback``.

    Class variables (statically declared, not serialized):
        python_fn:  Python callable for the scipy backend.
                    Signature: ``(t, x, p, spec) -> dict[str, Any]``
                    The dict maps ``"entity.field"`` keys to new values written
                    into ``x`` before the next solver segment.

    Pydantic fields (serialized):
        dt:        Controller period in seconds.  Must be > 0.
        julia_fn:  Julia function reference: ``"Module.function_name!"``.
                   Resolved from the already-loaded module scope in runner.jl /
                   server.jl — symbol lookup only, no eval of arbitrary code.
        params:    Scalar parameters passed to both the Python and Julia
                   callbacks.  Accessible in Julia as a ``Dict{String,Float64}``.

    Julia callback signature::

        function my_controller!(integrator, spec, params)
            # integrator.u  — current state x (writable)
            # integrator.p  — parameters p (read-only)
            # integrator.t  — current time
            # params        — Dict{String,Float64}
            i = state_idx(spec, "actuator.force")
            err = integrator.u[state_idx(spec, "sensor.angle")] - params["setpoint"]
            integrator.u[i] = params["kp"] * err
        end

    Example (Python side)::

        def pid_controller(t, x, p, spec):
            err = x[spec.state_idx("sensor.angle")] - 0.5
            return {"actuator.force": 1.0 * err}

        class PIDCallback(Callback):
            kind:      Literal["pid"] = "pid"
            dt:        float = 0.01
            julia_fn:  str   = "MyDynamics.pid_controller!"
            params:    dict[str, float] = {"kp": 1.0, "setpoint": 0.5}
            python_fn: ClassVar[ControlFn | None] = staticmethod(pid_controller)
    """

    python_fn: ClassVar[ControlFn | None] = None

    dt:       float = 0.0
    julia_fn: str   = ""
    params:   dict[str, float] = {}

    model_config = {"frozen": True}
