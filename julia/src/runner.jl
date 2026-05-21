"""
Standalone Julia runner called by JuliaBackend via subprocess.

Usage:
    julia --project=<julia_env> runner.jl <payload.json> <result.json> [dynamics.jl ...]

Reads a SolvePayload from payload.json, runs the ODE solver payload.reps times,
writes a SolveResult + per-solve timings to result.json, then exits.

When payload.reps > 1 the first solve includes JIT compilation of user dynamics
functions; subsequent solves are warm (compiled code, no tracing overhead).
The result trajectory is taken from the final solve.

Timing breakdown visible to Python:
  startup_ms   = wall time of this process − sum(timings_ms)
  timings_ms[0] = first solve  (JIT + dynamics)
  timings_ms[1:] = warm solves (dynamics only)
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

payload_file = ARGS[1]
result_file  = ARGS[2]

for f in ARGS[3:end]
    include(f)
end

# Read reps from the raw JSON before constructing SolvePayload (which has no reps field).
# StructTypes.Struct() ignores unknown keys, so SolvePayload deserialization is unaffected.
raw_json     = read(payload_file, String)
raw          = JSON3.read(raw_json)
reps         = get(raw, :reps, 1)
n_save       = get(raw, :n_save_points, 0)
dtsave_raw   = get(raw, :dtsave, nothing)
dtsave       = (dtsave_raw === nothing) ? 0.0 : Float64(dtsave_raw)
dtmax_raw    = get(raw, :dtmax, nothing)
dtmax        = (dtmax_raw === nothing) ? 0.0 : Float64(dtmax_raw)
maxiters_raw = get(raw, :maxiters, nothing)
maxiters     = (maxiters_raw === nothing) ? 0 : Int(maxiters_raw)
payload      = JSON3.read(raw_json, SolvePayload)

payload_json = JSON3.write(payload)

timings_ms = Float64[]
t0         = time_ns()
result     = Numen.solve(payload_json; n_save_points = n_save, dtsave = dtsave, dtmax = dtmax, maxiters = maxiters)
push!(timings_ms, (time_ns() - t0) / 1e6)
for _ in 2:reps
    t0     = time_ns()
    result = Numen.solve(payload_json; n_save_points = n_save, dtsave = dtsave, dtmax = dtmax, maxiters = maxiters)
    push!(timings_ms, (time_ns() - t0) / 1e6)
end

open(result_file, "w") do io
    x_rows = [result.x[i, :] for i in 1:size(result.x, 1)]
    JSON3.write(io, Dict("t" => result.t, "x" => x_rows, "timings_ms" => timings_ms))
end
