using JSON3, StructTypes

struct CompiledSystemSpec
    dynamics_fn::String
    entity_ids::Vector{String}
    group_size::Int
end

StructTypes.StructType(::Type{CompiledSystemSpec}) = StructTypes.Struct()

struct CompiledSpec
    state_size::Int
    param_size::Int
    state_index_map::Dict{String, Vector{Int}}   # "entity.field" → [start, stop] (0-based from Python)
    param_index_map::Dict{String, Vector{Int}}
    discrete_dts::Vector{Float64}
    x0::Vector{Float64}
    p::Vector{Float64}
    systems::Vector{CompiledSystemSpec}
end

StructTypes.StructType(::Type{CompiledSpec}) = StructTypes.Struct()

struct SolvePayload
    spec::CompiledSpec
    tspan::Vector{Float64}
    method::String
    rtol::Float64
    atol::Float64
end

StructTypes.StructType(::Type{SolvePayload}) = StructTypes.Struct()

struct SolveResult
    t::Vector{Float64}
    x::Matrix{Float64}   # state_size × n_steps
end

# Converts 0-based Python slice [start, stop) to 1-based Julia range start:stop
state_range(spec::CompiledSpec, key::String) =
    let e = spec.state_index_map[key]; (e[1]+1):e[2] end

param_range(spec::CompiledSpec, key::String) =
    let e = spec.param_index_map[key]; (e[1]+1):e[2] end

# Scalar convenience — 1-based Julia index of the first element
state_idx(spec::CompiledSpec, key::String) = first(state_range(spec, key))
param_idx(spec::CompiledSpec, key::String) = first(param_range(spec, key))
