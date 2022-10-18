function generate_guess(xinit, xterm, N, k_trans, dt_l, dt_u)
    pos₀ = xinit[1:6]
    v₀ = xinit[19:24]

    tf = dt_l * (k_trans - 1) + dt_u * (N - k_trans)

    t = 0.0
    

    for k = 1:N
        
        aₜ = k₀ + k₁ * t
        vₜ = v₀ + k₀ * t + k₁ * t^2 / 2
        posₜ = pos₀ + v₀ * t + k₀ * t^2 / 2 + k₁ * t^3 / 6

        if k < k_trans
            t = t + dt_l
        else
            t = t + dt_u
        end
end
