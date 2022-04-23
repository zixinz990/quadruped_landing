using GeometryBasics
using CoordinateTransformations, Rotations
using RobotDynamics
using Colors
using StaticArrays 
using MeshCat
using Blink
using LinearAlgebra
using TrajOptPlots

Base.@kwdef struct PlanarQuadruped <: AbstractModel
    g::Float64 = 9.81   # gravity
    mb::Float64 = 10.0  # body mass
    mf::Float64 = 0.1   # foot mass
    lb::Float64 = 1.0   # body length
    l1::Float64 = 0.3   # thigh length
    l2::Float64 = 0.3   # calf length
end

RobotDynamics.state_dim(::PlanarQuadruped) = 10
RobotDynamics.control_dim(::PlanarQuadruped) = 8

function contact1_dynamics(model::PlanarQuadruped, x, u)
    g = model.g
    mb = model.mb
    lb = model.lb
    Ib = mb*lb^2 / 12

    # state = [pb, θ, vb, ω, p1, p2]
    pb = x[1:2]   # body link position
    θ  = x[3]     # body link orientation
    p1 = x[4:5]   # foot 1 position
    p2 = x[6:7]   # foot 2 poisition

    # vb = x[8:9]
    # w = x[10]
    # v1 = x[11:12]
    # v2 = x[13:14]

    # control = [v1, v2, F1, F2]
    F1 = u[1:2]
    F2 = u[3:4]

    x_dot = zeros(length(x))

    x_dot[1:7] .= x[8:14]
    
    # body_dynamics
    temp_acc = (F1+F2) ./ mb
    x_dot[8] = temp_acc[1]
    x_dot[9] = temp_acc[2] + g
    # x_dot[8:9] .= (F1+F2) ./ mb
    τF = -F1[1]*(p1[2]-pb[2]) + F1[2]*(p1[1]-pb[1]) - F2[1]*(p2[2]-pb[2]) + F2[2]*(p2[1]-pb[1])
    x_dot[10] = τF / Ib

    # foot 1 constraints
    x_dot[11:12] = 0

    #foot 2 dynamics
    x_dot[13:14] .= F2 ./ mf


        
    # # foot 1 should be on the ground
    # x_dot[4:5] .= 0
    # # choose contact mode: 2
    # p1[2] = 0.0    # foot 1 position in y = 0
    # v1[2] = 0.0    # foot 1 linear velocity in y = 0
    # F2 = zero(F1)  # foot 2 GRF = 0

    # # dynamics
    # τF = -F1[1]*p1[2] + F1[2]*p1[1] - F2[1]*p2[2] + F2[2]*p2[1]
    # Ib = mb*lb^2 / 12

    # v̇b = (F1+F2)./mb .+ g
    # α = τF/Ib
    
    # ẋ = [vb; ω; v̇b; α; v1; v2]

    return x_dot
end

function contact2_dynamics(model::PlanarQuadruped, x, u)
    g = model.g
    mb = model.mb
    lb = model.lb
    Ib = mb*lb^2 / 12

    # state = [pb, θ, vb, ω, p1, p2]
    pb = x[1:2]   # body link position
    θ  = x[3]     # body link orientation
    p1 = x[4:5]   # foot 1 position
    p2 = x[6:7]   # foot 2 poisition

    # vb = x[8:9]
    # w = x[10]
    # v1 = x[11:12]
    # v2 = x[13:14]

    # control = [v1, v2, F1, F2]
    F1 = u[1:2]
    F2 = u[3:4]

    x_dot = zeros(length(x))

    x_dot[1:7] .= x[8:14]
    
    # body_dynamics
    temp_acc = (F1+F2) ./ mb
    x_dot[8] = temp_acc[1]
    x_dot[9] = temp_acc[2] + g
    τF = -F1[1]*(p1[2]-pb[2]) + F1[2]*(p1[1]-pb[1]) - F2[1]*(p2[2]-pb[2]) + F2[2]*(p2[1]-pb[1])
    x_dot[10] = τF / Ib

    # foot 1 constraints
    x_dot[11:12] = F1 ./mf

    #foot 2 dynamics
    x_dot[13:14] .= 0
    # g = model.g
    # mb = model.mb
    # lb = model.lb

    # # state = [pb, θ, vb, ω, p1, p2]
    # pb = x[1:2]   # body link position
    # θ  = x[3]     # body link orientation
    # vb = x[4:5]   # body link linear velocity
    # ω  = x[6]     # body link angualr velocity

    # p1 = x[7:8]   # foot 1 position
    # p2 = x[9:10]  # foot 2 position

    # # control = [v1, v2, F1, F2]
    # v1 = u[1:2]   # foot 1 linear velocity
    # v2 = u[3:4]   # foot 2 linear velocity
    # F1 = u[5:6]   # foot 1 GRF
    # F2 = u[7:8]   # foot 2 GRF

    # # choose contact mode: 2
    # p2[2] = 0.0    # foot 2 position in y = 0
    # v2[2] = 0.0    # foot 2 linear velocity in y = 0
    # F1 = zero(F1)  # foot 1 GRF = 0
    
    # # dynamics
    # τF = -F1[1]*p1[2] + F1[2]*p1[1] - F2[1]*p2[2] + F2[2]*p2[1]
    # Ib = mb*lb^2 / 12

    # v̇b = (F1+F2)./mb .+ g
    # α = τF/Ib
    
    # ẋ = [vb; ω; v̇b; α; v1; v2]

    return x_dot
end

function contact3_dynamics(model::PlanarQuadruped, x, u)
    g = model.g
    mb = model.mb
    lb = model.lb
    Ib = mb*lb^2 / 12

    # state = [pb, θ, vb, ω, p1, p2]
    pb = x[1:2]   # body link position
    θ  = x[3]     # body link orientation
    p1 = x[4:5]   # foot 1 position
    p2 = x[6:7]   # foot 2 poisition

    # vb = x[8:9]
    # w = x[10]
    # v1 = x[11:12]
    # v2 = x[13:14]

    # control = [v1, v2, F1, F2]
    F1 = u[1:2]
    F2 = u[3:4]

    x_dot = zeros(length(x))

    x_dot[1:7] .= x[8:14]
    
    # body_dynamics
    temp_acc = (F1+F2) ./ mb
    x_dot[8] = temp_acc[1]
    x_dot[9] = temp_acc[2] + g
    τF = -F1[1]*(p1[2]-pb[2]) + F1[2]*(p1[1]-pb[1]) - F2[1]*(p2[2]-pb[2]) + F2[2]*(p2[1]-pb[1])
    x_dot[10] = τF / Ib

    # foot 1 constraints
    x_dot[11:12] .= 0.

    #foot 2 dynamics
    x_dot[13:14] .= 0.
    # g = model.g
    # mb = model.mb
    # lb = model.lb

    # # state = [pb, θ, vb, ω, p1, p2]
    # pb = x[1:2]   # body link position
    # θ  = x[3]     # body link orientation
    # vb = x[4:5]   # body link linear velocity
    # ω  = x[6]     # body link angualr velocity

    # p1 = x[7:8]   # foot 1 position
    # p2 = x[9:10]  # foot 2 position

    # # control = [v1, v2, F1, F2]
    # v1 = u[1:2]   # foot 1 linear velocity
    # v2 = u[3:4]   # foot 2 linear velocity
    # F1 = u[5:6]   # foot 1 GRF
    # F2 = u[7:8]   # foot 2 GRF

    # # choose contact mode: 3
    # p1[2] = 0.0    # foot 1 position in y = 0
    # p2[2] = 0.0    # foot 2 position in y = 0

    # v1[2] = 0.0    # foot 1 linear velocity in y = 0
    # v2[2] = 0.0    # foot 2 linear velocity in y = 0

    # F1 = zero(F1)  # foot 1 GRF = 0
    # F2 = zero(F2)  # foot 2 GRF = 0

    # # dynamics
    # τF = -F1[1]*p1[2] + F1[2]*p1[1] - F2[1]*p2[2] + F2[2]*p2[1]
    # Ib = mb*lb^2 / 12

    # v̇b = (F1+F2)./mb .+ g
    # α = τF/Ib
    
    # ẋ = [vb; ω; v̇b; α; v1; v2]

    return ẋ
end

function contact1_dynamics_rk4(model, x, u, h)
    # RK4 integration with zero-order hold on u
    f1 = contact1_dynamics(model, x, u)
    f2 = contact1_dynamics(model, x + 0.5*h*f1, u)
    f3 = contact1_dynamics(model, x + 0.5*h*f2, u)
    f4 = contact1_dynamics(model, x + h*f3, u)
    return x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
end

function contact2_dynamics_rk4(model, x, u, h)
    # RK4 integration with zero-order hold on u
    f1 = contact2_dynamics(model, x, u)
    f2 = contact2_dynamics(model, x + 0.5*h*f1, u)
    f3 = contact2_dynamics(model, x + 0.5*h*f2, u)
    f4 = contact2_dynamics(model, x + h*f3, u)
    return x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
end

function contact3_dynamics_rk4(model, x, u, h)
    # RK4 integration with zero-order hold on u
    f1 = contact3_dynamics(model, x, u)
    f2 = contact3_dynamics(model, x + 0.5*h*f1, u)
    f3 = contact3_dynamics(model, x + 0.5*h*f2, u)
    f4 = contact3_dynamics(model, x + h*f3, u)
    return x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
end

function contact1_jacobian(model::PlanarQuadruped, x, u, dt)
    xi = SVector{10}(1:14)
    ui = SVector{8}(1:2) .+ 14
    f(z) = contact1_dynamics_rk4(model, z[xi], z[ui], dt)
    return ForwardDiff.jacobian(f, [x; u])
end

function contact2_jacobian(model::PlanarQuadruped, x, u, dt)
    xi = SVector{10}(1:14)
    ui = SVector{8}(1:2) .+ 14
    f(z) = contact2_dynamics_rk4(model, z[xi], z[ui], dt)
    return ForwardDiff.jacobian(f, [x; u])
end

function contact3_jacobian(model::PlanarQuadruped, x, u, dt)
    xi = SVector{10}(1:14)
    ui = SVector{8}(1:2) .+ 14
    f(z) = contact3_dynamics_rk4(model, z[xi], z[ui], dt)
    return ForwardDiff.jacobian(f, [x; u])
end

function jump1_map(x)
    # from mode 1 to 3
    xn = [x[1:4]; 0.0; x[6]; 0.0 ; x[8:10]; zeros(4)]
    return xn
end

function jump2_map(x)
    # from mode 2 to 3
    xn = [x[1:4]; 0.0; x[6]; 0.0 ; x[8:10]; zeros(4)]
    return xn
end

jump1_jacobian() = Diagonal(SA[1,1,1,1, 0 ,1, 0 ,1,1,1, 0,0,0,0])
jump1_jacobian() = Diagonal(SA[1,1,1,1, 0 ,1, 0 ,1,1,1, 0,0,0,0])
