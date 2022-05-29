"""
    reference_trajectory(model, N, xterm, init_mode)

Return a reference trajectory.
"""
function reference_trajectory(model::RealQuadruped, N, xterm, dt)
    n, m = size(model)

    g, mb = model.g, model.mb

    Xref = zeros(n, N)
    Uref = zeros(m, N-1)

    # initialize Xref
    Xref[:, 1:end] .= xterm
    Xref[end, 1:end] = range(0, dt * (N - 1), length=N)

    # initialize Uref
    Uref[3, :] .= -mb * g / 4   # F1_z
    Uref[6, :] .= -mb * g / 4   # F2_z
    Uref[9, :] .= -mb * g / 4   # F3_z
    Uref[12, :] .= -mb * g / 4  # F4_z
    
    Uref[end, :] .= dt

    # Convert to a trajectory
    Xref = [SVector{n}(x) for x in eachcol(Xref)]
    Uref = [SVector{m}(u) for u in eachcol(Uref)]
    return Xref, Uref
end
