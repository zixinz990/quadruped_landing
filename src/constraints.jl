# dynamics constraints
# contact constraints (2 per time step)
# kinematic constraints (2 per time step)

"""
    dynamics_constraint!(nlp, c, Z)

Calculate the dynamics constraints for the hybrid dynamics.
"""
function dynamics_constraint!(nlp::HybridNLP{n,m}, c, Z) where {n,m}
    xi, ui = nlp.xinds, nlp.uinds
    model = nlp.model
    init_mode = nlp.init_mode
    N = nlp.N
    k_trans = nlp.k_trans

    d = reshape(view(c, nlp.cinds[3]), (n, N - 1)) # size = n * (N-1)
    # d = c_x1_1 c_x1_2 ... c_x1_k ... c_x1_N-1
    #     c_x2_1 c_x2_2 ... c_x2_k ... c_x2_N-1
    #     ...    ...    ... ...    ... ...
    #     c_xn_1 c_xn_2 ... c_xn_k ... c_xn_N-1

    for k = 1:N-1
        x, u = Z[xi[k]], Z[ui[k]]
        x_next = Z[xi[k+1]]

        d[:, k] .= contact3_dynamics_rk4(model, x, u) - x_next
    end

    return vec(d)
end

"""
    contact_constraints!(nlp, c, Z)
    
Calculate the contact constraints for each time step.
"""
function contact_constraints!(nlp::HybridNLP{n,m}, c, Z) where {n,m}
    d = view(c, nlp.cinds[4])

    xi = nlp.xinds
    N = nlp.N
    init_mode = nlp.init_mode

    for k = 1:N
        x = Z[xi[k]]
        d[2*k-1] = x[5]  # y1 = 0
        d[2*k] = x[7]    # y2 = 0
    end

    return d
end

"""
    body_pos_constraints!(nlp, c, Z)

Calculate the body position constraints.
"""
function body_pos_constraints!(nlp::HybridNLP{n,m}, c, Z) where {n,m}
    d = view(c, nlp.cinds[6])

    xi = nlp.xinds
    N = nlp.N
    lb = nlp.model.lb

    for k = 1:N
        x = Z[xi[k]]
        yb = x[2]
        theta = x[3]
        d[k] = yb - lb / 2 * norm(sin(theta))
    end

    return d
end

# """
#     kinematics_constraints!(nlp, c, Z)

# Calculate the kinematics constraints.
# """
# function kinematics_constraints!(nlp::HybridNLP{n,m}, c, Z) where {n,m}
#     d = view(c, nlp.cinds[6])

#     xi = nlp.xinds
#     N = nlp.N

#     for k = 1:N
#         x = Z[xi[k]]

#         pb = x[1:2]
#         p1 = x[4:5]
#         p2 = x[6:7]

#         d[2*k-1] = norm(pb - p1)
#         d[2*k] = norm(pb - p2)
#     end

#     return d
# end

"""
    eval_c!(nlp, c, Z)

Evaluate all the constraints
"""
function eval_c!(nlp::HybridNLP, c, Z)
    xi = nlp.xinds
    ui = nlp.uinds

    c[nlp.cinds[1]] .= Z[xi[1]] - nlp.x0
    c[nlp.cinds[2]] .= Z[xi[end]][1:14] - nlp.xf[1:14]
    dynamics_constraint!(nlp, c, Z)
    contact_constraints!(nlp, c, Z)
    c[nlp.cinds[5]] .= Z[ui[end]][2] + Z[ui[end]][4] .+ nlp.model.mb * nlp.model.g
    body_pos_constraints!(nlp, c, Z)
    # kinematics_constraints!(nlp, c, Z)
end

# dynamics_jacobian!
# jac_c!

"""
    dynamics_jacobian!(nlp, jac, Z)

Calculate the Jacobian of the dynamics constraints, storing the result in the matrix `jac`.
"""
function dynamics_jacobian!(nlp::HybridNLP{n,m}, jac, Z) where {n,m}
    # Create a view of the portion of the Jacobian for the dynamics constraints
    D = view(jac, nlp.cinds[3], :)

    xi, ui = nlp.xinds, nlp.uinds
    model = nlp.model
    N = nlp.N

    ci = 1:n

    for k = 1:N-1
        x, u = Z[xi[k]], Z[ui[k]]
        
        D[ci, [xi[k]; ui[k]]] .= contact3_jacobian(model, x, u)
        D[ci, xi[k+1]] .= -I(n)

        ci = ci .+ n
    end

    return nothing
end

"""
    jac_c!(nlp, jac, Z)

Evaluate the constraint Jacobians.
"""
function jac_c!(nlp::HybridNLP{n,m}, jac, Z) where {n,m}
    xi, ui = nlp.xinds, nlp.uinds
    N = nlp.N
    init_mode = nlp.init_mode
    k_trans = nlp.k_trans

    # Create views for each portion of the Jacobian
    jac_init = view(jac, nlp.cinds[1], xi[1])
    jac_term = view(jac, nlp.cinds[2], xi[end])
    jac_dynamics = view(jac, nlp.cinds[3], :)
    jac_contact = view(jac, nlp.cinds[4], :)
    jac_final_ctrl = view(jac, nlp.cinds[5], ui[end])
    jac_body_pos = view(jac, nlp.cinds[6], :)
    # jac_kinematics = view(jac, nlp.cinds[7], :)

    jac_init .= I(n)
    jac_term .= I(n)[1:n-1, :]

    # jac_dynamics
    dynamics_jacobian!(nlp, jac, Z)

    # jac_contact
    for k = 1:N
        jac_contact[2*k-1, xi[k][5]] = 1.0  # y1
        jac_contact[2*k, xi[k][7]] = 1.0    # y2
    end

    # jac_final_ctrl
    jac_final_ctrl = [0.0 1.0 0.0 1.0 0.0]

    # jac_body_pos
    for k = 1:N
        x = Z[xi[k]]
        theta = x[3]

        jac_body_pos[k, xi[k][2]] = 1.0 # yb

        if theta > 0
            jac_body_pos[k, xi[k][3]] = -lb / 2 * cos(theta)
        else
            jac_body_pos[k, xi[k][3]] = lb / 2 * cos(theta)
        end
    end

    # # jac_kinematics
    # for k = 1:N
    #     x = Z[xi[k]]

    #     d1 = x[1:2] - x[7:8]
    #     d2 = x[1:2] - x[9:10]

    #     jac_kinematics[2*k-1, xi[k][1:2]] .= d1 / norm(d1)  # pb
    #     jac_kinematics[2*k-1, xi[k][4:5]] .= -d1 / norm(d1) # p1

    #     jac_kinematics[2*k, xi[k][1:2]] .= d2 / norm(d2)    # pb
    #     jac_kinematics[2*k, xi[k][6:7]] .= -d2 / norm(d2)   # p2
    # end

    return nothing
end
