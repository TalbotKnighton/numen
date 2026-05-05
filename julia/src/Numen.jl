module Numen

using JSON3
using OrdinaryDiffEq

include("types.jl")
include("solver.jl")
include("events.jl")

export solve, CompiledSpec, SolvePayload

end
