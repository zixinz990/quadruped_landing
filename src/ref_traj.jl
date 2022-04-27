"""
    reference_trajectory(model, N, k_trans, xterm, init_mode)

Return a reference trajectory that translates the walker from an x position of `xinit` to `xterm`,
with a nominal body height of `height` meters.
"""
function reference_trajectory(model::PlanarQuadruped, N, k_trans, xterm, init_mode, dt)
    n, m = size(model)

    g, mb = model.g, model.mb

    Xref = zeros(n, N)
    Uref = zeros(m, N)

    # initialize Xref
    Xref[:, 1:end] .= xterm
    Xref[end, 1:end] = range(0, dt * (N - 1), length=N)

    # initialize Uref
    if init_mode == 1
        Uref[2, 1:k_trans-1] .= mb * g     # F1_y in contact mode 1
        
        Uref[2, k_trans:end] .= mb * g / 2 # F1_y in contact mode 3
        Uref[4, k_trans:end] .= mb * g / 2 # F2_y in contact mode 3
    else
        Uref[4, 1:k_trans-1] .= mb * g     # F2_y in contact mode 2
        
        Uref[4, k_trans:end] .= mb * g / 2 # F2_y in contact mode 3
        Uref[2, k_trans:end] .= mb * g / 2 # F1_y in contact mode 3
    end
    
    Uref[end, :] .= dt

    # Convert to a trajectory
    Xref = [SVector{n}(x) for x in eachcol(Xref)]
    Uref = [SVector{m}(u) for u in eachcol(Uref)]
    return Xref, Uref
end
