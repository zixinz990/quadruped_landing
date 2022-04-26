"""
    reference_trajectory(model, times)

Return a reference trajectory that translates the walker from an x position of `xinit` to `xterm`,
with a nominal body height of `height` meters.
"""
function reference_trajectory(model::PlanarQuadruped, times, t_trans, xinit, xterm, init_mode)
    # Some useful variables
    n, m = size(model)
    tf = times[end]
    N = length(times)
    Δx = xterm - xinit
    dt = times[2]

    g, mb, lb, l1, l2 = model.g, model.mb, model.lb, model.l1, model.l2

    # initialization
    k_trans = 0
    for k = 1:N-1
        if times[k] < t_trans && times[k+1] >= t_trans
            k_trans = k + 1
            break
        end
    end

    Xref = zeros(n,N)
    Uref = zeros(m,N)

    # Xref[:, 1] .= xinit
    Xref[:, 1:end] .= xterm

    # a = 2 * Δx[4:7] / (t_trans^2) # acceleration of feet (assume feet have constant acceleration and zero velocity since touching the ground)

    # for k = 2:k_trans-1
    #     Xref[11:14, :] .= xinit[11:14] + a * (k-1)*dt # v1 and v2
    # end

    if init_mode == 1
        Uref[2, :] .= mb*g/2           # F1_y
        Uref[4, k_trans:end] .= mb*g/2 # F2_y
    else
        Uref[2, k_trans:end] .= mb*g/2 # F1_y
        Uref[4, :] .= mb*g/2           # F2_y
    end

    # Convert to a trajectory
    Xref = [SVector{n}(x) for x in eachcol(Xref)]
    Uref = [SVector{m}(u) for u in eachcol(Uref)]
    return Xref, Uref
end
