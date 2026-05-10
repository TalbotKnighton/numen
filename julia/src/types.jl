using JSON3, StructTypes

struct CompiledSystemSpec
    dynamics_fn::String
    entity_ids::Vector{String}
    group_size::Int
end

StructTypes.StructType(::Type{CompiledSystemSpec}) = StructTypes.Struct()

struct CompiledCallbackSpec
    name::String
    dt::Float64
    julia_fn::String
    params::Dict{String, Float64}
end

StructTypes.StructType(::Type{CompiledCallbackSpec}) = StructTypes.Struct()

struct CompiledSpec
    state_size::Int
    param_size::Int
    state_index_map::Dict{String, Vector{Int}}   # "entity.field" → [start, stop] (0-based from Python)
    param_index_map::Dict{String, Vector{Int}}
    discrete_dts::Vector{Float64}
    x0::Vector{Float64}
    p::Vector{Float64}
    differential_mask::Vector{Float64}           # 1.0 = integrated slot, 0.0 = algebraic slot
    systems::Vector{CompiledSystemSpec}
    callbacks::Vector{CompiledCallbackSpec}
    # COO-format sparsity pattern for the Jacobian ∂f/∂x, computed by compile_spec().
    # 0-based indices from Python; solver.jl converts to 1-based before constructing
    # the SparseMatrixCSC jac_prototype passed to ODEFunction.
    # Conservative: all states for entities in the same system group are fully coupled.
    jac_sparsity_rows::Vector{Int}
    jac_sparsity_cols::Vector{Int}
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

"""
    groups(sys::CompiledSystemSpec)

Iterate the entity groups of a system. Each iteration yields a slice of
`sys.entity_ids` of length `sys.group_size`, suitable for tuple destructuring
in dynamics functions:

    # group_size = 1 — single entity per group
    for (eid,) in groups(sys)
        i_pos = state_idx(spec, eid * ".oscillator.position")
        ...
    end

    # group_size = 3 — coupled triplet [a, middle, b]
    for (cv_a, orifice, cv_b) in groups(sys)
        i_Pa = state_idx(spec, cv_a * ".control_volume.pressure")
        ...
    end

Mirrors the Python `for entity_group in system.entity_groups:` pattern.
"""
groups(sys::CompiledSystemSpec) = Iterators.partition(sys.entity_ids, sys.group_size)
