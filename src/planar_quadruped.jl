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
    g::Float64 = -9.81  # gravity

    mb::Float64 = 10.0  # body mass
    mf::Float64 = 0.1   # foot mass

    lb::Float64 = 0.5   # body length
    l1::Float64 = 0.25  # thigh length
    l2::Float64 = 0.25  # calf length
end

# state: xₖ = [pbₖ; p1ₖ; p2ₖ; vbₖ; v1ₖ; v2ₖ; tₖ], dim = 15
# control: uₖ = [F1ₖ; F2ₖ; hₖ], dim = 5
# x[k+1] = [rk4(hₖ); tₖ+hₖ]
RobotDynamics.state_dim(::PlanarQuadruped) = 15
RobotDynamics.control_dim(::PlanarQuadruped) = 5

"""
    contact1_dynamics(model, x, u)

Calculate dynamics of contact mode 1.
Only return the first 14 elements.
The input state dim should be 14.
The input control dim should be 5.
"""
function contact1_dynamics(model::PlanarQuadruped, x, u)
    g = model.g
    mb = model.mb
    lb = model.lb
    mf = model.mf
    Ib = mb * lb^2 / 12

    pb = x[1:2]  # body link position
    θ = x[3]    # body link orientation
    p1 = x[4:5]  # foot 1 position
    p2 = x[6:7]  # foot 2 position
    v = x[8:14] # velocities

    vb = x[8:9]
    ω = x[10]
    v1 = x[11:12]
    v2 = x[13:14]

    F1x = u[1]
    F1y = u[2]
    F2x = u[3]
    F2y = u[4]

    # body_dynamics
    body_acc_x = (F1x + F2x) / mb
    body_acc_y = (F1y + F2y) / mb + g

    τF = -F1x * (p1[2] - pb[2]) + F1y * (p1[1] - pb[1]) - F2x * (p2[2] - pb[2]) + F2y * (p2[1] - pb[1])
    body_w = τF / Ib

    # foot 1 constraints
    foot_1_v   = zeros(2)
    foot_1_acc = zeros(2)

    # foot 2 dynamics
    foot_2_acc_x = -F2x / mf
    foot_2_acc_y = -F2y / mf + g

    # x_dot = zeros(length(x))
    # x_dot = [v; body_acc_x; body_acc_y; body_w; foot_1_a; foot_2_a]
    x_dot = [vb; ω; foot_1_v; v2; body_acc_x; body_acc_y; body_w; foot_1_acc; foot_2_acc_x; foot_2_acc_y]

    return x_dot
end

"""
    contact2_dynamics(model, x, u)

Calculate dynamics of contact mode 2.
Only return the first 14 elements.
The input state dim should be 14.
The input control dim should be 5.
"""
function contact2_dynamics(model::PlanarQuadruped, x, u)
    g = model.g
    mb = model.mb
    lb = model.lb
    mf = model.mf
    Ib = mb * lb^2 / 12

    pb = x[1:2]  # body link position
    θ = x[3]    # body link orientation
    p1 = x[4:5]  # foot 1 position
    p2 = x[6:7]  # foot 2 position
    v = x[8:14] # velocities

    vb = x[8:9]
    ω = x[10]
    v1 = x[11:12]
    v2 = x[13:14]

    F1x = u[1]
    F1y = u[2]
    F2x = u[3]
    F2y = u[4]

    # body_dynamics
    body_acc_x = (F1x + F2x) / mb
    body_acc_y = (F1y + F2y) / mb + g

    τF = -F1x * (p1[2] - pb[2]) + F1y * (p1[1] - pb[1]) - F2x * (p2[2] - pb[2]) + F2y * (p2[1] - pb[1])
    body_w = τF / Ib

    # foot 1 dynamics
    foot_1_acc_x = -F1x / mf
    foot_1_acc_y = -F1y / mf + g

    # foot 2 constraints
    foot_2_v   = zeros(2)
    foot_2_acc = zeros(2)

    # x_dot = zeros(length(x))
    # x_dot = [velocities; body_acc_x; body_acc_y; body_w; foot_1_a; foot_2_a]
    x_dot = [vb; ω; v1; foot_2_v; body_acc_x; body_acc_y; body_w; foot_1_acc_x; foot_1_acc_y; foot_2_acc]

    return x_dot
end

"""
    contact3_dynamics(model, x, u)

Calculate dynamics of contact mode 3.
Only return the first 14 elements.
The input state dim should be 14.
The input control dim should be 5.
"""
function contact3_dynamics(model::PlanarQuadruped, x, u)
    g = model.g
    mb = model.mb
    lb = model.lb
    mf = model.mf
    Ib = mb * lb^2 / 12

    pb = x[1:2]  # body link position
    θ  = x[3]    # body link orientation
    p1 = x[4:5]  # foot 1 position
    p2 = x[6:7]  # foot 2 position
    v  = x[8:14] # velocities

    vb = x[8:9]
    ω  = x[10]
    v1 = x[11:12]
    v2 = x[13:14]

    F1x = u[1]
    F1y = u[2]
    F2x = u[3]
    F2y = u[4]

    # body_dynamics
    body_acc_x = (F1x + F2x) / mb
    body_acc_y = (F1y + F2y) / mb + g

    τF = -F1x * (p1[2] - pb[2]) + F1y * (p1[1] - pb[1]) - F2x * (p2[2] - pb[2]) + F2y * (p2[1] - pb[1])
    body_w = τF / Ib

    # foot 1 constraints
    foot_1_v   = zeros(2)
    foot_1_acc = zeros(2)

    # foot 2 constraints
    foot_2_v   = zeros(2)
    foot_2_acc = zeros(2)

    # x_dot = zeros(length(x))
    # x_dot = [velocities; body_acc_x; body_acc_y; body_w; foot_1_a; foot_2_a]
    x_dot = [vb; ω; foot_1_v; foot_2_v; body_acc_x; body_acc_y; body_w; foot_1_acc; foot_2_acc]

    return x_dot
end

# the input state dim should be 15!
# the input control dim should be 5!
function contact1_dynamics_rk4(model, x, u)
    # RK4 integration with zero-order hold on u
    h = u[end]
    f1 = contact1_dynamics(model, x[1:end-1], u)
    f2 = contact1_dynamics(model, x[1:end-1] + 0.5 * h * f1, u)
    f3 = contact1_dynamics(model, x[1:end-1] + 0.5 * h * f2, u)
    f4 = contact1_dynamics(model, x[1:end-1] + h * f3, u)
    return [x[1:end-1] + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4); x[end] + u[end]]
end

# the input state dim should be 15!
# the input control dim should be 5!
function contact2_dynamics_rk4(model, x, u)
    # RK4 integration with zero-order hold on u
    h = u[end]
    f1 = contact2_dynamics(model, x[1:end-1], u)
    f2 = contact2_dynamics(model, x[1:end-1] + 0.5 * h * f1, u)
    f3 = contact2_dynamics(model, x[1:end-1] + 0.5 * h * f2, u)
    f4 = contact2_dynamics(model, x[1:end-1] + h * f3, u)
    return [x[1:end-1] + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4); x[end] + u[end]]
end

# the input state dim should be 15!
# the input control dim should be 5!
function contact3_dynamics_rk4(model, x, u)
    # RK4 integration with zero-order hold on u
    h = u[end]
    f1 = contact3_dynamics(model, x[1:end-1], u)
    f2 = contact3_dynamics(model, x[1:end-1] + 0.5 * h * f1, u)
    f3 = contact3_dynamics(model, x[1:end-1] + 0.5 * h * f2, u)
    f4 = contact3_dynamics(model, x[1:end-1] + h * f3, u)
    return [x[1:end-1] + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4); x[end] + u[end]]
end

# the input state dim should be 15!
# the input control dim should be 5!
function contact1_jacobian(model::PlanarQuadruped, x, u)
    xi = SVector{15}(1:15)
    ui = SVector{5}(1:5) .+ 15
    f(z) = contact1_dynamics_rk4(model, z[xi], z[ui])
    return ForwardDiff.jacobian(f, [x; u])
end

# the input state dim should be 15!
# the input control dim should be 5!
function contact2_jacobian(model::PlanarQuadruped, x, u)
    xi = SVector{15}(1:15)
    ui = SVector{5}(1:5) .+ 15
    f(z) = contact2_dynamics_rk4(model, z[xi], z[ui])
    return ForwardDiff.jacobian(f, [x; u])
end

# the input state dim should be 15!
# the input control dim should be 5!
function contact3_jacobian(model::PlanarQuadruped, x, u)
    xi = SVector{15}(1:15)
    ui = SVector{5}(1:5) .+ 15
    f(z) = contact3_dynamics_rk4(model, z[xi], z[ui])
    return ForwardDiff.jacobian(f, [x; u])
end

function jump1_map(x)
    # from mode 1 to 3
    xn = [x[1:4]; 0.0; x[6]; 0.0; x[8:10]; zeros(4); x[15]] # y1 = y2 = 0
    return xn
end

function jump2_map(x)
    # from mode 2 to 3
    xn = [x[1:4]; 0.0; x[6]; 0.0; x[8:10]; zeros(4); x[15]] # y1 = y2 = 0
    return xn
end

jump1_jacobian() = Diagonal(SA[1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0])
jump2_jacobian() = Diagonal(SA[1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0])
