"""
Persistent Julia server for JuliaServerBackend.

Usage (managed by Python — do not call directly):
    julia --project=<env> server.jl [dynamics.jl ...]

Lifecycle:
  1. Loads packages and includes user dynamics files.
  2. Writes "NUMEN_SERVER_READY\\n" to stderr — Python waits for this.
  3. Loops: reads one SolvePayload JSON line from stdin, solves, writes one
     result JSON line to stdout, flushes.  Exits cleanly on EOF or empty line.

Each request is a SolvePayload JSON object (same schema as runner.jl).
Each response is either (success — chunked over n_states+2 lines):
  Line 1:  {"n_t": <int>, "n_states": <int>}
  Line 2:  [<t0>, <t1>, ...]
  Line 3…: [<x_row_i_0>, <x_row_i_1>, ...]  (one line per state)
Or (failure — single line, server keeps running):
  {"error": "message string"}
"""

import Pkg
let proj = get(ENV, "JULIA_PROJECT", nothing)
    if proj !== nothing
        Pkg.activate(proj; io=devnull)
    end
end

include(joinpath(@__DIR__, "Numen.jl"))
using .Numen
using JSON3

# Built-in characterization dynamics (excitation_dynamics!, chirp_dynamics!)
include(joinpath(@__DIR__, "characterization.jl"))

for f in ARGS
    include(f)
end

# Precompile all dynamics functions in loaded modules so the first real solve
# runs at full compiled speed with no JIT latency.  Scans every top-level module
# in Main for functions matching the standard Numen dynamics signature.
const _DYNAMICS_SIG = (
    Vector{Float64}, Vector{Float64}, Vector{Float64},
    Float64, CompiledSpec, CompiledSystemSpec,
)

function _precompile_dynamics_modules()
    n = 0
    for sym in names(Main; all=false)
        mod = try getfield(Main, sym) catch; continue end
        mod isa Module || continue
        for fn_sym in names(mod; all=false)
            fn = try getfield(mod, fn_sym) catch; continue end
            fn isa Function || continue
            precompile(fn, _DYNAMICS_SIG) && (n += 1)
        end
    end
    println(stderr, "NUMEN_PRECOMPILE_DONE ($n functions compiled)")
    flush(stderr)
end

if get(ENV, "NUMEN_DISABLE_PRECOMPILE", "") == "1"
    println(stderr, "NUMEN_PRECOMPILE_SKIPPED")
    flush(stderr)
else
    _precompile_dynamics_modules()
end

println(stderr, "NUMEN_SERVER_READY")
flush(stderr)

while !eof(stdin)
    line = readline(stdin)
    isempty(line) && continue
    try
        raw          = JSON3.read(line)
        n_save       = get(raw, :n_save_points, 0)
        dtsave_raw   = get(raw, :dtsave, nothing)
        dtsave       = (dtsave_raw === nothing) ? 0.0 : Float64(dtsave_raw)
        dtmax_raw    = get(raw, :dtmax, nothing)
        dtmax        = (dtmax_raw === nothing) ? 0.0 : Float64(dtmax_raw)
        maxiters_raw = get(raw, :maxiters, nothing)
        maxiters     = (maxiters_raw === nothing) ? 0 : Int(maxiters_raw)
        result = Numen.solve(line; n_save_points = n_save, dtsave = dtsave, dtmax = dtmax, maxiters = maxiters)
        n_t      = length(result.t)
        n_states = size(result.x, 1)
        # Chunked protocol: header line, then t line, then one line per state row.
        # Keeps each readline() call small regardless of result size.
        println(stdout, JSON3.write((n_t=n_t, n_states=n_states)))
        println(stdout, JSON3.write(result.t))
        for i in 1:n_states
            println(stdout, JSON3.write(result.x[i, :]))
        end
    catch e
        msg = sprint(showerror, e, catch_backtrace())
        # Error is always a single line so Python's first readline() gets it.
        println(stdout, JSON3.write((error=msg,)))
    end
    flush(stdout)
end
