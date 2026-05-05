"""
    solve(payload_json::String) -> SolveResult

Entry point called from Python via juliacall. Receives a JSON-encoded SolvePayload,
builds and solves the ODE system, returns a SolveResult.
"""
function solve(payload_json::String)::SolveResult
    payload   = JSON3.read(payload_json, SolvePayload)
    spec      = payload.spec
    tspan     = (payload.tspan[1], payload.tspan[2])
    dynamics! = build_dynamics(spec)
    tstops    = build_tstops(spec.discrete_dts, tspan)

    prob   = ODEProblem(dynamics!, copy(spec.x0), tspan, copy(spec.p))
    solver = getfield(OrdinaryDiffEq, Symbol(payload.method))()
    sol    = OrdinaryDiffEq.solve(
        prob,
        solver,
        tstops  = tstops,
        saveat  = tstops,
        abstol  = payload.atol,
        reltol  = payload.rtol,
    )

    x_matrix = hcat(sol.u...)
    return SolveResult(sol.t, x_matrix)
end

"""
    build_dynamics(spec::CompiledSpec) -> Function

Resolves each system's ``dynamics_fn`` string to a callable Julia function,
then returns a combined ``dynamics!(dx, x, p, t)`` closure that zeroes ``dx``,
calls every system function with ``(dx, x, p, t, spec, sys_spec)``, and returns.

Function names must be of the form ``"Module.function_name!"`` where ``Module``
is already loaded in ``Main`` (via ``include`` before calling ``solve``).
"""
function build_dynamics(spec::CompiledSpec)
    fns_and_specs = map(spec.systems) do sys_spec
        parts = split(sys_spec.dynamics_fn, ".")
        mod   = length(parts) > 1 ? getfield(Main, Symbol(parts[1])) : Main
        fn    = getfield(mod, Symbol(parts[end]))
        (fn, sys_spec)
    end

    function dynamics!(dx, x, p, t)
        fill!(dx, 0.0)
        for (f, sys_spec) in fns_and_specs
            f(dx, x, p, t, spec, sys_spec)
        end
    end

    return dynamics!
end

function build_tstops(discrete_dts::Vector{Float64}, tspan::Tuple{Float64, Float64})
    isempty(discrete_dts) && return Float64[]
    t0, tf = tspan
    times  = Float64[]
    for dt in discrete_dts
        t = t0 + dt
        while t <= tf
            push!(times, t)
            t += dt
        end
    end
    sort!(unique!(times))
end
