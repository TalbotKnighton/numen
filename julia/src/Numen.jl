module Numen

using JSON3
using OrdinaryDiffEq

include("types.jl")
include("solver.jl")
include("events.jl")

export solve, CompiledSpec, CompiledSystemSpec, SolvePayload,
       state_idx, param_idx, state_range, param_range

end
