"""
    HybridNLP{n,m,L,Q}

Represents a (N)on(L)inear (P)rogram of a trajectory optimization problem,
with a dynamics model of type `L`, a quadratic cost function, horizon `T`, 
and initial and final state `x0`, `xf`.

The kth state and control can be extracted from the concatenated state vector `Z` using
`Z[nlp.xinds[k]]`, and `Z[nlp.uinds[k]]`.

# Constructor
    HybridNLP(model, obj, tf, N, M, x0, xf, [integration])

# Basic Methods
    Base.size(nlp)    # returns (n,m,T)
    num_ineq(nlp)     # number of inequality constraints
    num_eq(nlp)       # number of equality constraints
    num_primals(nlp)  # number of primal variables
    num_duals(nlp)    # total number of dual variables
    packZ(nlp, X, U)  # Stacks state `X` and controls `U` into one vector `Z`

# Evaluating the NLP
The NLP supports the following API for evaluating various pieces of the NLP:

    eval_f(nlp, Z)         # evaluate the objective
    grad_f!(nlp, grad, Z)  # gradient of the objective
    eval_c!(nlp, c, Z)     # evaluate the constraints
    jac_c!(nlp, c, Z)      # constraint Jacobian
"""
struct HybridNLP{n,m,L,Q} <: MOI.AbstractNLPEvaluator
    model::L                                 # dynamics model
    obj::Vector{QuadraticCost{n,m,Float64}}  # objective function
    N::Int                                   # number of knot points
    Nmodes::Int                              # number of modes
    init_mode::Int                           # the mode ID of the initial state
    t_trans::Float64                         # the staet time of mode 3 (sec)
    k_trans::Int                             # the start index of mode 3
    tf::Float64                              # total time (sec)
    x0::MVector{n,Float64}                   # initial condition
    xf::MVector{n,Float64}                   # final condition
    times::Vector{Float64}                   # vector of times
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
    blocks::BlockViews
    function HybridNLP(model, obj::Vector{<:QuadraticCost{n,m}}, init_mode::Integer,
            tf::Real, N::Integer, x0::AbstractVector, xf::AbstractVector,
            integration::Type{<:QuadratureRule}=RK4; use_sparse_jacobian::Bool=false
        ) where {n,m}
        # Create indices
        xinds = [SVector{n}((k-1)*(n+m) .+ (1:n)) for k = 1:N]
        uinds = [SVector{m}((k-1)*(n+m) .+ (n+1:n+m)) for k = 1:N-1]
        times = collect(range(0, tf, length=N))
        
        # Contact schedule
        modes = map(1:N) do k
            (times[k] < t_trans) ? init_mode : 3
        end
        Nmodes = 2

        # Calculate k_trans
        k_trans = 0
        for k = 1:N-1
            if times[k] < t_trans && times[k+1] >= t_trans
                k_trans = k + 1
                break
            end
        end
        
        # Constraints
        c_init_inds = 1:n                                                                               # initial constraint
        c_term_inds = (c_init_inds[end]+1):(c_init_inds[end]+n)                                         # terminal constraint
        c_dyn_inds = (c_term_inds[end]+1):(c_term_inds[end]+(N-1)*n)                                    # dynamics constraints
        c_init_contact_inds = (c_dyn_inds[end]+1):(c_dyn_inds[end]+2*N)                                 # contact constraints of the initial mode (2 per time step)
        c_other_contact_inds = (c_init_contact_inds[end]+1):(c_init_contact_inds[end]+2*(N-k_trans+1))  # contact constraints of another leg (2 per time step)
        c_kin_inds = (c_other_contact_inds[end]+1):(c_other_contact_inds[end]+2*N)                      # kinematic constraints (2 per time step)
        # c_col_inds = (c_kin_inds[end]+1):(c_kin_inds[end]+N)                                            # self-collision constraints (1 per time step)
        
        cinds = [c_init_inds, c_term_inds, c_dyn_inds, c_init_contact_inds, c_other_contact_inds, c_kin_inds]
        m_nlp = c_kin_inds[end]  # total number of constraints

        # Constraints bounds
        lb = fill(0.0, m_nlp) # lower bounds on the constraints
        ub = fill(0.0, m_nlp) # upper bounds on the constraints

        ub[c_kin_inds] .= model.l1 + model.l2 + model.lb/2
        # ub[c_col_inds] .= 2*(model.l1+model.l2) + model.lb

        n_nlp = n*N + (N-1)*m
        zL = fill(-Inf, n_nlp)
        zU = fill(+Inf, n_nlp)
        rows = Int[]
        cols = Int[]
        blocks = BlockViews(m_nlp, n_nlp)
        
        new{n,m,typeof(model), integration}(
            model, obj,
            N, Nmodes, init_mode, t_trans, k_trans, tf, x0, xf, times, modes,
            xinds, uinds, cinds, lb, ub, zL, zU, rows, cols, use_sparse_jacobian, blocks
        )
    end
end
Base.size(nlp::HybridNLP{n,m}) where {n,m} = (n,m,nlp.N)
num_primals(nlp::HybridNLP{n,m}) where {n,m} = n*nlp.N + m*(nlp.N-1)
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
