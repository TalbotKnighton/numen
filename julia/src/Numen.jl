module Numen

using JSON3
using LinearAlgebra
using OrdinaryDiffEq
using ADTypes: AutoFiniteDiff

include("types.jl")
include("solver.jl")
include("events.jl")

export solve, CompiledSpec, CompiledSystemSpec, CompiledCallbackSpec, SolvePayload,
       state_idx, param_idx, state_range, param_range

end
