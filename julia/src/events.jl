# Julia-tier callbacks via DiffEqCallbacks.PeriodicCallback.
#
# build_callbacks resolves each callback's julia_fn string from the already-loaded
# module scope — symbol lookup only, no eval of arbitrary code — and wraps it in a
# PeriodicCallback that fires at t0+dt, t0+2dt, …
#
# User callback signature:
#   function my_controller!(integrator, spec, params)
#       # integrator.u  — current state x (writable)
#       # integrator.p  — parameters p    (read-only)
#       # integrator.t  — current time
#       # params        — Dict{String,Float64}
#       i = state_idx(spec, "actuator.force")
#       integrator.u[i] = params["kp"] * (integrator.u[state_idx(spec,"sensor.angle")] - params["setpoint"])
#   end

using DiffEqCallbacks

_EXPLICIT_SOLVERS = Set(["Tsit5", "Dopri5", "Vern7", "Vern8", "Vern9",
                          "BS3", "BS5", "OwrenZen3", "OwrenZen4", "OwrenZen5"])

"""
    build_callbacks(spec, tspan) -> CallbackSet or nothing

Resolves each CompiledCallbackSpec.julia_fn to a callable in the loaded module
scope, wraps it in a PeriodicCallback, and returns a CallbackSet.
Returns nothing if spec.callbacks is empty.
"""
function build_callbacks(spec::CompiledSpec)
    isempty(spec.callbacks) && return nothing

    cbs = map(spec.callbacks) do cb_spec
        fn = _resolve_fn(cb_spec.julia_fn)
        params = cb_spec.params
        fire! = integrator -> fn(integrator, spec, params)
        PeriodicCallback(fire!, cb_spec.dt)
    end

    return CallbackSet(cbs...)
end

"""
    check_dae_solver(method, spec)

Warn (or error) if algebraic constraints are present but an explicit solver was chosen.
"""
function check_dae_solver(method::String, spec::CompiledSpec)
    has_algebraic = any(m -> m == 0.0, spec.differential_mask)
    if has_algebraic && method in _EXPLICIT_SOLVERS
        error(
            "DAE constraints (ContinuousField algebraic=True) require an implicit solver.\n" *
            "Received: method=\"$method\" (explicit).\n" *
            "Use an implicit solver instead: \"Rodas5P\" (recommended), \"FBDF\", or \"KenCarp4\"."
        )
    end
end

"""
    _resolve_fn(fn_string) -> Function

Resolves a "Module.function_name!" string to a callable in the loaded Julia scope.
Only performs symbol lookup — does not eval arbitrary code.
"""
function _resolve_fn(fn_string::String)
    parts = split(fn_string, ".")
    if length(parts) == 1
        return getfield(Main, Symbol(parts[1]))
    end
    mod = getfield(Main, Symbol(parts[1]))
    return getfield(mod, Symbol(parts[end]))
end
