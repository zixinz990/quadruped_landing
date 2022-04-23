# dynamics constraints
# contact constraints of the initial mode (2 per time step)
# contact constraints of another leg (2 per time step)
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
    dt = nlp.times[2]
    k_trans = nlp.k_trans

    # Grab a view of the indices for the dynamics constraints
    d = reshape(view(c, nlp.cinds[3]), n, N-1)

    for k = 1:N-1
        x, u = Z[xi[k]], Z[ui[k]]
        x_next = Z[xi[k+1]]
        if k < k_trans - 1  # in mode 1 or 2
            if init_mode == 1
                d[:, k] = contact1_dynamics_rk4(model, x, u, dt) - x_next
            else
                d[:, k] = contact2_dynamics_rk4(model, x, u, dt) - x_next
            end
        elseif k == k_trans - 1  # before transition, jump!
            if init_mode == 1
                d[:, k] = jump1_map(contact1_dynamics_rk4(model, x, u, dt)) - x_next
            else
                d[:, k] = jump2_map(contact2_dynamics_rk4(model, x, u, dt)) - x_next
            end
        else  # in mode 3
            d[:, k] = contact3_dynamics_rk4(model, x, u, dt) - x_next
        end
    end

    return vec(d)
end

"""
    contact_init_constraints!(nlp, c, Z)

Calculate the contact constraints of the initial mode for each time step.
"""
function contact_init_constraints!(nlp::HybridNLP{n,m}, c, Z) where {n,m}
    d = view(c, nlp.cinds[4])

    xi, ui = nlp.xinds, nlp.uinds
    N = nlp.N
    init_mode = nlp.init_mode

    for k = 1:N
        x = Z[xi[k]]
        if init_mode == 1
            d[2*k-1] = x[4] # x1 = 0
            d[2*k] = x[5]   # y1 = 0
        else
            d[2*k-1] = x[6] # x2 = 0
            d[2*k] = x[7]   # y2 = 0
        end
    end

    return d
end

"""
    contact_constraints_other!(nlp, c, Z)
    
Calculate the contact constraints of another leg for each time step after transition
"""
function contact_another_constraints!(nlp::HybridNLP{n,m}, c, Z) where {n,m}
    d = view(c, nlp.cinds[5])

    # Some useful variables
    xi, ui = nlp.xinds, nlp.uinds
    N = nlp.N
    init_mode = nlp.init_mode
    k_trans = nlp.k_trans

    for k = 1:N-k_trans+1
        x = Z[xi[k]]
        if init_mode == 1
            d[2*k-1] = x[6] # x2 = 0
            d[2*k] = x[7]   # y2 = 0
        else
            d[2*k-1] = x[4] # x1 = 0
            d[2*k] = x[5]   # y1 = 0
        end
    end

    return d
end

"""
    kinematics_constraints!(nlp, c, Z)

Calculate the kinematics constraints.
"""
function kinematics_constraints!(nlp::HybridNLP{n,m}, c, Z) where {n,m}
    d = view(c, nlp.cinds[6])

    xi, ui = nlp.xinds, nlp.uinds
    N = nlp.N
    init_mode = nlp.init_mode

    for k = 1:N
        x = Z[xi[k]]

        pb = x[1:2]
        p1 = x[4:5]
        p2 = x[6:7]

        d[2*k-1] = norm(pb - p1)
        d[2*k] = norm(pb - p2)
    end

    return d
end

"""
    self_collision_constraints!(nlp, c, Z)

Calculate the self-collision constraints.
"""
# function self_collision_constraints!(nlp::HybridNLP{n,m}, c, Z) where {n,m}
#     # Create a view for the portion for the length constraints
#     d = view(c, nlp.cinds[7])

#     # Some useful variables
#     xi, ui = nlp.xinds, nlp.uinds
#     N = nlp.N                      # number of time steps
#     Nmodes = nlp.Nmodes            # number of mode sequences (N ÷ M)
#     init_mode = nlp.init_mode

#     for k = 1:N
#         x = Z[xi[k]]

#         p1 = x[7:8]
#         p2 = x[9:10]

#         d[k] = norm(p1 - p2)
#     end

#     return d   # for easy Jacobian checking
# end

"""
    eval_c!(nlp, c, Z)

Evaluate all the constraints
"""
function eval_c!(nlp::HybridNLP, c, Z)
    xi = nlp.xinds
    c[nlp.cinds[1]] .= Z[xi[1]] - nlp.x0
    c[nlp.cinds[2]] .= Z[xi[end]] - nlp.xf
    dynamics_constraint!(nlp, c, Z)
    contact_init_constraints!(nlp, c, Z)
    contact_another_constraints!(nlp, c, Z)
    kinematics_constraints!(nlp, c, Z)
    # self_collision_constraints!(nlp, c, Z)
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

    # Some useful variables
    xi, ui = nlp.xinds, nlp.uinds
    model = nlp.model
    init_mode = nlp.init_mode
    N = nlp.N
    dt = nlp.times[2]
    k_trans = nlp.k_trans

    ci = 1:n

    for k = 1:N-1
        x, u = Z[xi[k]], Z[ui[k]]

        if k < k_trans - 1  # in mode 1 or 2
            if init_mode == 1
                D[ci, [xi[k]; ui[k]]] = contact1_jacobian(model, x, u, dt)
            else
                D[ci, [xi[k]; ui[k]]] = contact2_jacobian(model, x, u, dt)
            end
        elseif k == k_trans - 1  # before transition, jump!
            if init_mode == 1
                D[ci, [xi[k]; ui[k]]] .= jump1_jacobian() * contact1_jacobian(model, x, u, dt)
            else
                D[ci, [xi[k]; ui[k]]] .= jump2_jacobian() * contact2_jacobian(model, x, u, dt)
            end
        else  # in mode 3
            D[ci, [xi[k]; ui[k]]] = contact3_jacobian(model, x, u, dt)
        end

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
    t_trans = nlp.t_trans

    k_trans = 0
    for k = 1:N-1
        if times[k] < t_trans && times[k+1] >= t_trans
            k_trans = k + 1
            break
        end
    end

    # Create views for each portion of the Jacobian
    jac_init = view(jac, nlp.cinds[1], xi[1])
    jac_term = view(jac, nlp.cinds[2], xi[end])
    jac_dynamics = view(jac, nlp.cinds[3], :)
    jac_contact_init = view(jac, nlp.cinds[4], :)
    jac_contact_another = view(jac, nlp.cinds[5], :)
    jac_kinematics = view(jac, nlp.cinds[6], :)
    # jac_self_collision = view(jac, nlp.cinds[7], :)

    jac_init .= I(n)
    jac_term .= I(n)

    # jac_dynamics
    dynamics_jacobian!(nlp, jac, Z)

    # jac_contact_init
    if init_mode == 1
        for k = 1:N
            jac_contact_init[k, xi[k][4]] = 1.0 # x1
            jac_contact_init[k, xi[k][5]] = 1.0 # y1
        end
    else
        for k = 1:N
            jac_contact_init[k, xi[k][6]] = 1.0 # x2
            jac_contact_init[k, xi[k][7]] = 1.0 # y2
        end
    end

    # jac_contact_another
    if init_mode == 1
        for k = k_trans:N
            i = k - k_trans + 1
            jac_contact_another[i, xi[k][6]] = 1.0 # x2
            jac_contact_another[i, ui[k][7]] = 1.0 # y2
        end
    else
        for k = k_trans:N
            i = k - k_trans + 1
            jac_contact_another[i, xi[k][4]] = 1.0 # x1
            jac_contact_another[i, ui[k][5]] = 1.0 # y1
        end
    end

    # jac_kinematics
    for k = 1:N
        x = Z[xi[k]]

        d1 = x[1:2] - x[7:8]
        d2 = x[1:2] - x[9:10]

        jac_kinematics[2*k-1, xi[k][1:2]] = d1 / norm(d1)  # pb
        jac_kinematics[2*k-1, xi[k][4:5]] = -d1 / norm(d1) # p1

        jac_kinematics[2*k, xi[k][1:2]] = d2 / norm(d2)  # pb
        jac_kinematics[2*k, xi[k][6:7]] = -d2 / norm(d2) # p2
    end

    # jac_self_collision
    # for k = 1:N
    #     x = Z[xi[k]]

    #     d12 = x[7:8] - x[9:10]

    #     jac_self_collision[k, xi[k][7:8]] = d12 / norm(d12)
    #     jac_self_collision[k, xi[k][9:10]] = -d12 / norm(d12)
    # end

    return nothing
end