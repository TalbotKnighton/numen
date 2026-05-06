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
Each response is either:
  {"t": [...], "x": [[...], ...]}          on success
  {"error": "message string"}              on failure (server keeps running)
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

for f in ARGS
    include(f)
end

println(stderr, "NUMEN_SERVER_READY")
flush(stderr)

while !eof(stdin)
    line = readline(stdin)
    isempty(line) && continue
    try
        result = Numen.solve(line)
        x_rows = [result.x[i, :] for i in 1:size(result.x, 1)]
        println(stdout, JSON3.write((t=result.t, x=x_rows)))
    catch e
        msg = sprint(showerror, e, catch_backtrace())
        println(stdout, JSON3.write((error=msg,)))
    end
    flush(stdout)
end
