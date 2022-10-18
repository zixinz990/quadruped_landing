function MOI.eval_objective(prob::HybridNLP, x)
    eval_f(prob, x)
end

function MOI.eval_objective_gradient(prob::HybridNLP, grad_f, x)
    grad_f!(prob, grad_f, x)
    return nothing
end

function MOI.eval_constraint(prob::HybridNLP, g, x)
    eval_c!(prob, g, x)
    return nothing
end

function MOI.eval_constraint_jacobian(prob::HybridNLP, vec, x)
    n_nlp, m_nlp = num_primals(prob), num_duals(prob)
    if prob.use_sparse_jacobian
        jac = vec
    else
        jac = reshape(vec, m_nlp, n_nlp)
    end
    jac_c!(nlp, jac, x)
    return nothing
end

function MOI.features_available(prob::HybridNLP)
    return [:Grad, :Jac]
end

MOI.initialize(prob::HybridNLP, features) = nothing
function MOI.jacobian_structure(nlp::HybridNLP)
    vec(Tuple.(CartesianIndices(zeros(num_duals(nlp), num_primals(nlp)))))
end

"""
    solve(x0, nlp::HybridNLP; tol, c_tol, max_iter)

Solve the NLP `nlp` using Ipopt via MathOptInterface, providing `x0` as an initial guess.

# Keyword Arguments
The following arguments are sent to Ipopt
* `tol`: overall optimality tolerance
* `c_tol`: constraint feasibility tolerance
* `max_iter`: maximum number of solver iterations
"""
function solve(x0, prob::HybridNLP;
    tol=1.0e-6, c_tol=1.0e-6, max_iter=1000)
    n_nlp, m_nlp = num_primals(prob), num_duals(prob)
    N = prob.N

    x_l, x_u = fill(-Inf, n_nlp), fill(+Inf, n_nlp)

    for k = 1:N
        # lower and upper bound of body orientation
        x_l[3+20*(k-1)] = -pi / 2 # theta >= -pi/2
        x_u[3+20*(k-1)] = pi / 2  # theta <= pi/2
        
        if k < N
            # lower bound and upper bound of dt
            x_l[20+20*(k-1)] = 0.01
            x_u[20+20*(k-1)] = 0.04
            
            # lower bound of F
            x_l[22+20*(k-1)] = 0.0
            x_l[24+20*(k-1)] = 0.0
        end
    end

    # initial GRFs are 0
    x_l[16:19] .= 0.0
    x_u[16:19] .= 0.0

    c_l, c_u = prob.lb, prob.ub

    println("Creating NLP Block Data...")
    nlp_bounds = MOI.NLPBoundsPair.(c_l, c_u)
    has_objective = true
    block_data = MOI.NLPBlockData(nlp_bounds, prob, has_objective)

    println("Creating Ipopt...")
    solver = Ipopt.Optimizer()
    solver.options["max_iter"] = max_iter
    solver.options["tol"] = tol
    solver.options["constr_viol_tol"] = c_tol
    # solver.options["print_level"] = 8

    x = MOI.add_variables(solver, n_nlp)

    println("Adding constraints...")
    for i = 1:n_nlp
        xi = MOI.SingleVariable(x[i])
        MOI.add_constraint(solver, xi, MOI.LessThan(x_u[i]))
        MOI.add_constraint(solver, xi, MOI.GreaterThan(x_l[i]))
        MOI.set(solver, MOI.VariablePrimalStart(), x[i], x0[i])
    end

    # Solve the problem
    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    println("starting Ipopt Solve...")
    MOI.optimize!(solver)

    # Get the solution
    res = MOI.get(solver, MOI.VariablePrimal(), x)

    return res, solver
end