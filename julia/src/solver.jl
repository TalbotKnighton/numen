"""
    solve(payload_json::String) -> SolveResult

Entry point called from Python via runner.jl or server.jl.  Receives a
JSON-encoded SolvePayload, builds and solves the ODE/DAE system, returns a
SolveResult.
"""
function solve(payload_json::String)::SolveResult
    payload   = JSON3.read(payload_json, SolvePayload)
    spec      = payload.spec
    tspan     = (payload.tspan[1], payload.tspan[2])
    method    = payload.method

    check_dae_solver(method, spec)

    dynamics! = build_dynamics(spec)
    tstops    = build_tstops(spec.discrete_dts, tspan)
    cb_set    = build_callbacks(spec)

    has_algebraic = any(m -> m == 0.0, spec.differential_mask)

    if has_algebraic
        # Mass-matrix DAE path — pass Diagonal(differential_mask) to ODEFunction.
        # Algebraic slots have mask=0 (no time derivative); the dynamics fn writes
        # a constraint residual g(x)=0 for those slots.
        M      = Diagonal(spec.differential_mask)
        ode_fn = ODEFunction(dynamics!, mass_matrix = M)
    else
        ode_fn = ODEFunction(dynamics!)
    end

    prob   = ODEProblem(ode_fn, copy(spec.x0), tspan, copy(spec.p))
    solver = getfield(OrdinaryDiffEq, Symbol(method))()

    kw = (
        tstops  = tstops,
        saveat  = tstops,
        abstol  = payload.atol,
        reltol  = payload.rtol,
    )
    sol = cb_set !== nothing ?
        OrdinaryDiffEq.solve(prob, solver; kw..., callback = cb_set) :
        OrdinaryDiffEq.solve(prob, solver; kw...)

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
        (_resolve_fn(sys_spec.dynamics_fn), sys_spec)
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
