"""
    solve(payload_json::String) -> SolveResult

Entry point called from Python via runner.jl or server.jl.  Receives a
JSON-encoded SolvePayload, builds and solves the ODE/DAE system, returns a
SolveResult.
"""
function solve(payload_json::String; n_save_points::Int = 0, dtsave::Float64 = 0.0, dtmax::Float64 = 0.0, maxiters::Int = 0)::SolveResult
    payload   = JSON3.read(payload_json, SolvePayload)
    spec      = payload.spec
    tspan     = (payload.tspan[1], payload.tspan[2])
    method    = payload.method

    check_dae_solver(method, spec)

    dynamics! = build_dynamics(spec)
    tstops    = build_tstops(spec.discrete_dts, tspan)
    cb_set    = build_callbacks(spec)

    has_algebraic = any(m -> m == 0.0, spec.differential_mask)

    # Explicit finite-difference time gradient for Rosenbrock stiff solvers.
    # Rodas5P (and all Rosenbrock methods) compute ∂f/∂t via ForwardDiff by default,
    # which requires calling dynamics! with t::Dual.  OrdinaryDiffEq wraps dynamics!
    # in a FunctionWrapper{..., Float64} at problem-construction time (inferred from
    # tspan::Tuple{Float64,Float64}), which rejects Dual t at runtime.
    # Providing tgrad! explicitly bypasses that path entirely: Rosenbrock reads ∂f/∂t
    # from tgrad! (two Float64 f-evals, constant cost) while the state Jacobian still
    # uses ForwardDiff through the normal jac_prototype path (no FunctionWrapper issue
    # because the state buffers are allocated separately outside the wrapper).
    tgrad! = build_tgrad(dynamics!, spec.state_size)

    # Sparse Jacobian prototype — conservatively dense within each entity group,
    # zero across groups.  OrdinaryDiffEq uses SparseDiffTools matrix coloring so
    # the color count equals the maximum group coupling width, not the total state size.
    # For N independent entities (color count ≈ max_entity_states), Jacobian cost is
    # O(1) in N rather than O(N/chunk) for a dense ForwardDiff sweep.
    use_sparse = !isempty(spec.jac_sparsity_rows)
    jac_proto  = use_sparse ? sparse(
        spec.jac_sparsity_rows .+ 1,   # 0-based Python → 1-based Julia
        spec.jac_sparsity_cols .+ 1,
        ones(length(spec.jac_sparsity_rows)),
        spec.state_size, spec.state_size,
    ) : nothing

    if has_algebraic
        # Mass-matrix DAE path — pass Diagonal(differential_mask) to ODEFunction.
        # Algebraic slots have mask=0 (no time derivative); the dynamics fn writes
        # a constraint residual g(x)=0 for those slots.
        M      = Diagonal(spec.differential_mask)
        ode_fn = use_sparse ?
            ODEFunction(dynamics!, mass_matrix = M, tgrad = tgrad!, jac_prototype = jac_proto) :
            ODEFunction(dynamics!, mass_matrix = M, tgrad = tgrad!)
    else
        ode_fn = use_sparse ?
            ODEFunction(dynamics!, tgrad = tgrad!, jac_prototype = jac_proto) :
            ODEFunction(dynamics!, tgrad = tgrad!)
    end

    prob   = ODEProblem(ode_fn, copy(spec.x0), tspan, copy(spec.p))
    solver = getfield(OrdinaryDiffEq, Symbol(method))()

    # saveat controls output density.  Options (mutually exclusive):
    #   n_save_points > 0  →  N uniformly-spaced points via linspace
    #   dtsave > 0         →  fixed interval via t0:dtsave:tf
    #   neither            →  save at tstops only (empty = every adaptive step)
    # In all cases, tstops are merged in so no discrete-field event is missed.
    saveat_times = if n_save_points > 0
        grid = collect(range(tspan[1], tspan[2]; length = n_save_points))
        isempty(tstops) ? grid : sort!(unique!(vcat(grid, tstops)))
    elseif dtsave > 0.0
        grid = collect(tspan[1]:dtsave:tspan[2])
        # Always include the final time
        (isempty(grid) || grid[end] < tspan[2]) && push!(grid, tspan[2])
        isempty(tstops) ? grid : sort!(unique!(vcat(grid, tstops)))
    else
        tstops   # empty → every adaptive step; non-empty → events only
    end

    kw = (
        tstops  = tstops,
        saveat  = saveat_times,
        abstol  = payload.atol,
        reltol  = payload.rtol,
    )
    # dtmax > 0 limits adaptive step size — prevents missing brief transients
    # or aliasing high-frequency inputs.  0.0 means no limit (solver default).
    if dtmax > 0.0
        kw = merge(kw, (dtmax = dtmax,))
    end
    # maxiters > 0 raises the solver's iteration cap above the default (1e5).
    # Required for long chirps or fine dtmax that need millions of steps.
    if maxiters > 0
        kw = merge(kw, (maxiters = maxiters,))
    end
    sol = cb_set !== nothing ?
        OrdinaryDiffEq.solve(prob, solver; kw..., callback = cb_set) :
        OrdinaryDiffEq.solve(prob, solver; kw...)

    x_matrix = hcat(sol.u...)
    return SolveResult(sol.t, x_matrix)
end

"""
    build_tgrad(dynamics!, n_states) -> Function

Returns a central-difference time-gradient `tgrad!(du, u, p, t)` for use as the
`tgrad` keyword in `ODEFunction`.  Pre-allocates two work buffers of size `n_states`
in the closure so no allocation occurs at call time.

h = √eps(Float64) ≈ 1.5e-8 gives O(h²) accuracy (central differences).
"""
function build_tgrad(dynamics!, n_states::Int)
    du_fwd = zeros(n_states)
    du_bwd = zeros(n_states)
    function tgrad!(du, u, p, t)
        h = sqrt(eps(Float64))
        fill!(du_fwd, 0.0)
        fill!(du_bwd, 0.0)
        dynamics!(du_fwd, u, p, t + h)
        dynamics!(du_bwd, u, p, t - h)
        @. du = (du_fwd - du_bwd) / (2h)
    end
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
