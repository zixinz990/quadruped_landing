"""
    reference_trajectory(model, N, k_trans, xterm, init_mode)

Return a reference trajectory.
"""
function reference_trajectory(model::PlanarQuadruped, N, k_trans, xterm, dt)
    n, m = size(model)

    Xref = zeros(n, N)
    Uref = zeros(m, N - 1)

    # initialize Xref
    Xref[:, 1:end] .= xterm
    Xref[end, 1:end] = range(0, dt * (N - 1), length=N)

    # initialize Uref
    # if init_mode == 1
    #     Uref[2, 1:k_trans-1] .= -mb * g      # F1_y in contact mode 1

    #     Uref[2, k_trans:end] .= -mb * g / 2  # F1_y in contact mode 3
    #     Uref[4, k_trans:end] .= -mb * g / 2  # F2_y in contact mode 3
    # else
    #     Uref[4, 1:k_trans-1] .= -mb * g      # F2_y in contact mode 2

    #     Uref[4, k_trans:end] .= -mb * g / 2  # F2_y in contact mode 3
    #     Uref[2, k_trans:end] .= -mb * g / 2  # F1_y in contact mode 3
    # end

    Uref[end, 1:k_trans-1] .= 0.015  # dt in contact mode 1 or 2
    Uref[end, k_trans:end] .= 0.02   # dt in contact mode 3

    # Convert to a trajectory
    Xref = [SVector{n}(x) for x in eachcol(Xref)]
    Uref = [SVector{m}(u) for u in eachcol(Uref)]
    return Xref, Uref
end
