"""
    HybridNLP{n,m,L,Q}

Represents a (N)on(L)inear (P)rogram of a trajectory optimization problem.

The kth state and control can be extracted from the concatenated state vector `Z` using
`Z[nlp.xinds[k]]`, and `Z[nlp.uinds[k]]`.

# Constructor
    HybridNLP(model, obj, init_mode, k_trans, N, x0, xf, integration; use_sparse_jacobian)

"""
struct HybridNLP{n,m,L,Q} <: MOI.AbstractNLPEvaluator
    model::L                                 # dynamics model
    obj::Vector{QuadraticCost{n,m,Float64}}  # objective function
    N::Int                                   # number of knot points
    Nmodes::Int                              # number of modes
    init_mode::Int                           # the mode ID of the initial state
    k_trans::Int                             # the start index of mode 3
    x0::MVector{n,Float64}                   # initial condition
    xf::MVector{n,Float64}                   # final condition
    modes::Vector{Int}                       # mode ID
    xinds::Vector{SVector{n,Int}}            # Z[xinds[k]] gives states for time step k
    uinds::Vector{SVector{m,Int}}            # Z[uinds[k]] gives controls for time step k
    cinds::Vector{UnitRange{Int}}            # indices for each of the constraints
    lb::Vector{Float64}                      # lower bounds on the constraints
    ub::Vector{Float64}                      # upper bounds on the constraints
    zL::Vector{Float64}                      # lower bounds on the primal variables
    zU::Vector{Float64}                      # upper bounds on the primal variables
    rows::Vector{Int}                        # rows for Jacobian sparsity
    cols::Vector{Int}                        # columns for Jacobian sparsity
    use_sparse_jacobian::Bool
    function HybridNLP(model, obj::Vector{<:QuadraticCost{n,m}}, init_mode::Integer, k_trans::Integer,
        N::Integer, x0::AbstractVector, xf::AbstractVector,
        integration::Type{<:QuadratureRule}=RK4; use_sparse_jacobian::Bool=false
    ) where {n,m}
        # Create indices
        xinds = [SVector{n}((k - 1) * (n + m) .+ (1:n)) for k = 1:N]
        uinds = [SVector{m}((k - 1) * (n + m) .+ (n+1:n+m)) for k = 1:N-1]

        # Contact schedule
        modes = map(1:N) do k
            (k < k_trans) ? init_mode : 3
        end
        Nmodes = 2

        # Equality constraints
        c_init_inds = 1:n                                                                                # initial constraint
        c_term_inds = (c_init_inds[end]+1):(c_init_inds[end]+n-1)                                        # terminal constraint
        c_dyn_inds = (c_term_inds[end]+1):(c_term_inds[end]+(N-1)*n)                                     # dynamics constraints

        # c_init_contact_inds = (c_dyn_inds[end]+1):(c_dyn_inds[end]+2*N)                                  # contact constraints of the initial touch-down feet
        # c_another_contact_inds = (c_init_contact_inds[end]+1):(c_init_contact_inds[end]+2*(N-k_trans+1)) # contact constraints of another feet

        # c_final_ctrl_inds = (c_another_contact_inds[end]+1):(c_another_contact_inds[end]+1)              # final control constraint

        # Inequality constraints
        # c_body_pos_inds = (c_final_ctrl_inds[end]+1):(c_final_ctrl_inds[end]+N)  # body position constraints

        # c_kin_inds = (c_body_pos_inds[end]+1):(c_body_pos_inds[end]+2*N)                         # kinematic constraints (2 per time step)

        cinds = [c_init_inds, c_term_inds, c_dyn_inds]
        m_nlp = cinds[end][end]

        # Constraints bounds
        lb = fill(0.0, m_nlp) # lower bounds on the constraints
        ub = fill(0.0, m_nlp) # upper bounds on the constraints

        # ub[c_body_pos_inds] .= Inf
        # ub[c_kin_inds] .= model.l1 + model.l2 + model.lb/2

        n_nlp = n * N + (N - 1) * m
        zL = fill(-Inf, n_nlp)
        zU = fill(+Inf, n_nlp)
        rows = Int[]
        cols = Int[]

        new{n,m,typeof(model),integration}(
            model, obj,
            N, Nmodes, init_mode, k_trans, x0, xf, modes,
            xinds, uinds, cinds, lb, ub, zL, zU, rows, cols, use_sparse_jacobian
        )
    end
end
Base.size(nlp::HybridNLP{n,m}) where {n,m} = (n, m, nlp.N)
num_primals(nlp::HybridNLP{n,m}) where {n,m} = n * nlp.N + m * (nlp.N - 1)
num_duals(nlp::HybridNLP) = nlp.cinds[end][end]

"""
    packZ(nlp, X, U)

Take a vector state vectors `X` and controls `U` and stack them into a single vector Z.
"""
function packZ(nlp, X, U)
    Z = zeros(num_primals(nlp))
    for k = 1:nlp.N-1
        Z[nlp.xinds[k]] = X[k]
        Z[nlp.uinds[k]] = U[k]
    end
    Z[nlp.xinds[end]] = X[end]
    return Z
end

"""
    unpackZ(nlp, Z)

Take a vector of all the states and controls and return a vector of state vectors `X` and
controls `U`.
"""
function unpackZ(nlp, Z)
    X = [Z[xi] for xi in nlp.xinds]
    U = [Z[ui] for ui in nlp.uinds]
    return X, U
end
