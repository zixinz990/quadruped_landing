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

# state:   xₖ = [pbₖ; p1ₖ; p2ₖ; vbₖ; v1ₖ; v2ₖ; tₖ], dim = 7+7+1 = 15
# control: uₖ = [F1ₖ; F2ₖ; hₖ], dim = 2+2+1 = 5
# x[k+1] = [rk4(hₖ); tₖ+hₖ]
RobotDynamics.state_dim(::PlanarQuadruped) = 15
RobotDynamics.control_dim(::PlanarQuadruped) = 5

# this function only return the derivative of the first 14 elements!
# the input state dim should be 14!
# the input control dim should be 5!
function contact1_dynamics(model::PlanarQuadruped, x, u)
    g = model.g
    mb = model.mb
    lb = model.lb
    mf = model.mf
    Ib = mb * lb^2 / 12

    # state = [pb, θ, vb, ω, p1, p2]
    pb = x[1:2]  # body link position
    p1 = x[4:5]  # foot 1 position
    p2 = x[6:7]  # foot 2 poisition
    velocities = x[8:14]

    # control = [v1, v2, F1, F2]
    F1 = u[1:2]
    F2 = u[3:4]

    # body_dynamics
    temp_acc = (F1 + F2) ./ mb

    body_acc_x = temp_acc[1]
    body_acc_y = temp_acc[2] + g

    τF = -F1[1] * (p1[2] - pb[2]) + F1[2] * (p1[1] - pb[1]) - F2[1] * (p2[2] - pb[2]) + F2[2] * (p2[1] - pb[1])
    body_w = τF / Ib

    # foot 1 constraints
    foot_1_a = zeros(2)

    # foot 2 dynamics
    foot_2_a = -F2 ./ mf

    x_dot = zeros(length(x))
    x_dot = [velocities; body_acc_x; body_acc_y; body_w; foot_1_a; foot_2_a]

    return x_dot
end

# this function only return the derivative of the first 14 elements!
# the input state dim should be 14!
# the input control dim should be 5!
function contact2_dynamics(model::PlanarQuadruped, x, u)
    g = model.g
    mb = model.mb
    lb = model.lb
    mf = model.mf
    Ib = mb * lb^2 / 12

    # state = [pb, θ, vb, ω, p1, p2]
    pb = x[1:2]  # body link position
    p1 = x[4:5]  # foot 1 position
    p2 = x[6:7]  # foot 2 poisition
    velocities = x[8:14]

    # control = [v1, v2, F1, F2]
    F1 = u[1:2]
    F2 = u[3:4]

    # body_dynamics
    temp_acc = (F1 + F2) ./ mb

    body_acc_x = temp_acc[1]
    body_acc_y = temp_acc[2] + g

    τF = -F1[1] * (p1[2] - pb[2]) + F1[2] * (p1[1] - pb[1]) - F2[1] * (p2[2] - pb[2]) + F2[2] * (p2[1] - pb[1])
    body_w = τF / Ib

    # foot 1 dynamics
    foot_1_a = -F1 ./ mf

    # foot 2 constraints
    foot_2_a = zeros(2)

    x_dot = zeros(length(x))
    x_dot = [velocities; body_acc_x; body_acc_y; body_w; foot_1_a; foot_2_a]

    return x_dot
end

# this function only return the derivative of the first 14 elements!
# the input state dim should be 14!
# the input control dim should be 5!
function contact3_dynamics(model::PlanarQuadruped, x, u)
    g = model.g
    mb = model.mb
    lb = model.lb
    mf = model.mf
    Ib = mb * lb^2 / 12

    # state = [pb, θ, vb, ω, p1, p2]
    pb = x[1:2]  # body link position
    p1 = x[4:5]  # foot 1 position
    p2 = x[6:7]  # foot 2 poisition
    velocities = x[8:14]

    # control = [v1, v2, F1, F2]
    F1 = u[1:2]
    F2 = u[3:4]

    # body_dynamics
    temp_acc = (F1 + F2) ./ mb

    body_acc_x = temp_acc[1]
    body_acc_y = temp_acc[2] + g

    τF = -F1[1] * (p1[2] - pb[2]) + F1[2] * (p1[1] - pb[1]) - F2[1] * (p2[2] - pb[2]) + F2[2] * (p2[1] - pb[1])
    body_w = τF / Ib

    # foot 1 constraints
    foot_1_a = zeros(2)

    # foot 2 constraints
    foot_2_a = zeros(2)

    x_dot = zeros(length(x))
    x_dot = [velocities; body_acc_x; body_acc_y; body_w; foot_1_a; foot_2_a]

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
jump1_jacobian() = Diagonal(SA[1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0])
