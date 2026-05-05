"""
Standalone Julia runner called by JuliaBackend via subprocess.

Usage:
    julia --project=<julia_env> runner.jl <payload.json> <result.json> [dynamics.jl ...]

Reads a SolvePayload from payload.json, runs the ODE solver, writes a
SolveResult to result.json, then exits.  Any extra .jl arguments are
included (in order) before solving so user dynamics modules are available.
"""

import Pkg
# Activate the juliapkg-managed environment if JULIA_PROJECT is set,
# otherwise activate the numen julia/ project directly.
let proj = get(ENV, "JULIA_PROJECT", nothing)
    if proj !== nothing
        Pkg.activate(proj; io=devnull)
    end
end

include(joinpath(@__DIR__, "Numen.jl"))
using .Numen
using JSON3

payload_file = ARGS[1]
result_file  = ARGS[2]

# Include any user-supplied dynamics files
for f in ARGS[3:end]
    include(f)
end

payload = open(payload_file) do io
    JSON3.read(io, SolvePayload)
end

result = Numen.solve(JSON3.write(payload))

open(result_file, "w") do io
    # Serialize x as a vector of row-vectors (state_size × n_steps) so Python
    # np.array produces shape (state_size, n_steps) directly without transposing.
    # JSON3 flattens Matrix to a 1-D array, so we must convert explicitly.
    x_rows = [result.x[i, :] for i in 1:size(result.x, 1)]
    JSON3.write(io, Dict("t" => result.t, "x" => x_rows))
end
