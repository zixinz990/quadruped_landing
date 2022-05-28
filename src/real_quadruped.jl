using GeometryBasics
using CoordinateTransformations, Rotations
using RobotDynamics
using Colors
using StaticArrays
using MeshCat
using Blink
using LinearAlgebra
using TrajOptPlots

Base.@kwdef struct RealQuadruped <: AbstractModel
    g::Float64 = -9.81  # gravity

    mb::Float64 = 15.0  # body mass
    mf::Float64 = 0.1   # foot mass

    lb::Float64 = 0.5   # body length
    a::Float64 = 0.3    # length (x axis)
    b::Float64 = 0.2    # width (y axis)
    c::Float64 = 0.15   # height (z axis)

    l1::Float64 = 0.25  # thigh length
    l2::Float64 = 0.25  # calf length
end

# state: xₖ = [pbₖ; θbₖ; p1ₖ; p2ₖ; p3ₖ; p4ₖ;
#              vbₖ; ωbₖ; v1ₖ; v2ₖ; v3ₖ; v4ₖ;
#              tₖ], dim = 37
# control: uₖ = [F1ₖ; F2ₖ; F3ₖ; F4ₖ; hₖ], dim = 13
# x[k+1] = [rk4(hₖ); tₖ+hₖ]
RobotDynamics.state_dim(::RealQuadruped) = 37
RobotDynamics.control_dim(::RealQuadruped) = 13

"""
    contact1_dynamics(model, x, u)

Calculate dynamics of contact mode 1, foot 1 and 3 are on the ground
Only return the first 36 elements.
The input state dim should be 36.
The input control dim should be 13.
"""
function contact1_dynamics(model::RealQuadruped, x, u)
    g = model.g
    mb = model.mb
    a = model.a
    b = model.b
    c = model.c
    mf = model.mf

    pb = x[1:3]    # body position

    ϕ = x[4]       # roll
    θ = x[5]       # pitch
    ψ = x[6]       # yaw

    p1 = x[7:9]    # foot 1 position
    p2 = x[10:12]  # foot 2 position
    p3 = x[13:15]  # foot 3 position
    p4 = x[16:18]  # foot 4 poistion

    vb = x[19:21]  # body linear velocity
    ω = x[22:24]   # body angular velocity
    v1 = zeros(3)  # foot 1 velocity
    v2 = x[28:30]  # foot 2 velocity
    v3 = zeros(3)  # foot 3 velocity
    v4 = x[34:36]  # foot 4 velocity

    F1 = u[1:3]
    F2 = u[4:6]
    F3 = u[7:9]
    F4 = u[10:12]

    # body linear acceleration in world frame
    ab = (F1 + F2 + F3 + F4) / mb + [0; 0; g]

    # body Euler angular velocity (can be approximated)
    J = [cos(ψ)/cos(θ) sin(ψ)/cos(θ) 0; -sin(ψ) cos(ψ) 0; cos(ψ)*tan(θ) sin(ψ)*tan(θ) 1]
    # J = [cos(ψ) sin(ψ) 0; -sin(ψ) cos(ψ) 0; 0 0 1]
    Θ_dot = J * ω

    # torque in world frame
    τF = cross(p1 - pb, F1) + cross(p2 - pb, F2) + cross(p3 - pb, F3) + cross(p4 - pb, F4)

    # yaw
    Rz = [cos(ψ) -sin(ψ) 0; sin(ψ) cos(ψ) 0; 0 0 1]
    # pitch
    Ry = [cos(θ) 0 sin(θ); 0 1 0; -sin(θ) 0 cos(θ)]
    # roll
    Rx = [1 0 0; 0 cos(ϕ) -sin(ϕ); 0 sin(ϕ) cos(ϕ)]
    # rotation matrix
    R = Rz * Ry * Rx

    # inertia tensor in body frame
    Ib = [(mb/12)*(b^2+c^2) 0 0; 0 (mb/12)*(a^2+c^2) 0; 0 0 (mb/12)*(a^2+b^2)]

    # inertia tensor in world frame (can be approximated)
    Iw = R * Ib * R'
    # Iw = Rz * Ib * Rz

    # body angular acceleration in world frame (can be approximated)
    ω_dot = inv(Iw) * (τF - cross(ω, Iw * ω))
    # ω_dot = inv(Iw) * τF

    # foot 1 and 3 are on the ground
    a1 = zeros(3)
    a3 = zeros(3)

    # foot 2 and 4 are in the air
    a2 = -F2 / mf + [0; 0; g]
    a4 = -F4 / mf + [0; 0; g]

    x_dot = [vb; Θ_dot; v1; v2; v3; v4; ab; ω_dot; a1; a2; a3; a4]

    return x_dot
end

"""
    contact2_dynamics(model, x, u)

Calculate dynamics of contact mode 2, foot 2 and 4 are on the ground
Only return the first 36 elements.
The input state dim should be 36.
The input control dim should be 13.
"""
function contact2_dynamics(model::RealQuadruped, x, u)
    g = model.g
    mb = model.mb
    a = model.a
    b = model.b
    c = model.c
    mf = model.mf

    pb = x[1:3]    # body position

    ϕ = x[4]       # roll
    θ = x[5]       # pitch
    ψ = x[6]       # yaw

    p1 = x[7:9]    # foot 1 position
    p2 = x[10:12]  # foot 2 position
    p3 = x[13:15]  # foot 3 position
    p4 = x[16:18]  # foot 4 poistion

    vb = x[19:21]  # body linear velocity
    ω = x[22:24]   # body angular velocity
    v1 = x[25:27]  # foot 1 velocity
    v2 = zeros(3)  # foot 2 velocity
    v3 = x[31:33]  # foot 3 velocity
    v4 = zeros(3)  # foot 4 velocity

    F1 = u[1:3]
    F2 = u[4:6]
    F3 = u[7:9]
    F4 = u[10:12]

    # body linear acceleration in world frame
    ab = (F1 + F2 + F3 + F4) / mb + [0; 0; g]

    # body Euler angular velocity (can be approximated)
    J = [cos(ψ)/cos(θ) sin(ψ)/cos(θ) 0; -sin(ψ) cos(ψ) 0; cos(ψ)*tan(θ) sin(ψ)*tan(θ) 1]
    # J = [cos(ψ) sin(ψ) 0; -sin(ψ) cos(ψ) 0; 0 0 1]
    Θ_dot = J * ω

    # torque in world frame
    τF = cross(p1 - pb, F1) + cross(p2 - pb, F2) + cross(p3 - pb, F3) + cross(p4 - pb, F4)

    # yaw
    Rz = [cos(ψ) -sin(ψ) 0; sin(ψ) cos(ψ) 0; 0 0 1]
    # pitch
    Ry = [cos(θ) 0 sin(θ); 0 1 0; -sin(θ) 0 cos(θ)]
    # roll
    Rx = [1 0 0; 0 cos(ϕ) -sin(ϕ); 0 sin(ϕ) cos(ϕ)]
    # rotation matrix
    R = Rz * Ry * Rx

    # inertia tensor in body frame
    Ib = [(mb/12)*(b^2+c^2) 0 0; 0 (mb/12)*(a^2+c^2) 0; 0 0 (mb/12)*(a^2+b^2)]

    # inertia tensor in world frame (can be approximated)
    Iw = R * Ib * R'
    # Iw = Rz * Ib * Rz

    # body angular acceleration in world frame (can be approximated)
    ω_dot = inv(Iw) * (τF - cross(ω, Iw * ω))
    # ω_dot = inv(Iw) * τF

    # foot 1 and 3 are in the air
    a1 = -F1 / mf + [0; 0; g]
    a3 = -F3 / mf + [0; 0; g]

    # foot 2 and 4 are on the ground
    a2 = zeros(3)
    a4 = zeros(3)

    x_dot = [vb; Θ_dot; v1; v2; v3; v4; ab; ω_dot; a1; a2; a3; a4]

    return x_dot
end

"""
    contact3_dynamics(model, x, u)

Calculate dynamics of contact mode 3, all feet are on the ground
Only return the first 36 elements.
The input state dim should be 36.
The input control dim should be 13.
"""
function contact3_dynamics(model::RealQuadruped, x, u)
    g = model.g
    mb = model.mb
    a = model.a
    b = model.b
    c = model.c
    mf = model.mf

    pb = x[1:3]    # body position

    ϕ = x[4]       # roll
    θ = x[5]       # pitch
    ψ = x[6]       # yaw

    p1 = x[7:9]    # foot 1 position
    p2 = x[10:12]  # foot 2 position
    p3 = x[13:15]  # foot 3 position
    p4 = x[16:18]  # foot 4 poistion

    vb = x[19:21]  # body linear velocity
    ω = x[22:24]   # body angular velocity
    v1 = zeros(3)  # foot 1 velocity
    v2 = zeros(3)  # foot 2 velocity
    v3 = zeros(3)  # foot 3 velocity
    v4 = zeros(3)  # foot 4 velocity

    F1 = u[1:3]
    F2 = u[4:6]
    F3 = u[7:9]
    F4 = u[10:12]

    # body linear acceleration in world frame
    ab = (F1 + F2 + F3 + F4) / mb + [0; 0; g]

    # body Euler angular velocity (can be approximated)
    J = [cos(ψ)/cos(θ) sin(ψ)/cos(θ) 0; -sin(ψ) cos(ψ) 0; cos(ψ)*tan(θ) sin(ψ)*tan(θ) 1]
    # J = [cos(ψ) sin(ψ) 0; -sin(ψ) cos(ψ) 0; 0 0 1]
    Θ_dot = J * ω

    # torque in world frame
    τF = cross(p1 - pb, F1) + cross(p2 - pb, F2) + cross(p3 - pb, F3) + cross(p4 - pb, F4)

    # yaw
    Rz = [cos(ψ) -sin(ψ) 0; sin(ψ) cos(ψ) 0; 0 0 1]
    # pitch
    Ry = [cos(θ) 0 sin(θ); 0 1 0; -sin(θ) 0 cos(θ)]
    # roll
    Rx = [1 0 0; 0 cos(ϕ) -sin(ϕ); 0 sin(ϕ) cos(ϕ)]
    # rotation matrix
    R = Rz * Ry * Rx

    # inertia tensor in body frame
    Ib = [(mb/12)*(b^2+c^2) 0 0; 0 (mb/12)*(a^2+c^2) 0; 0 0 (mb/12)*(a^2+b^2)]

    # inertia tensor in world frame (can be approximated)
    Iw = R * Ib * R'
    # Iw = Rz * Ib * Rz

    # body angular acceleration in world frame (can be approximated)
    ω_dot = inv(Iw) * (τF - cross(ω, Iw * ω))
    # ω_dot = inv(Iw) * τF

    # foot 1 and 3 are on the ground
    a1 = zeros(3)
    a3 = zeros(3)

    # foot 2 and 4 are on the ground
    a2 = zeros(3)
    a4 = zeros(3)

    x_dot = [vb; Θ_dot; v1; v2; v3; v4; ab; ω_dot; a1; a2; a3; a4]

    return x_dot
end

# the input state dim should be 37!
# the input control dim should be 13!
function contact1_dynamics_rk4(model, x, u)
    # RK4 integration with zero-order hold on u
    h = u[end]
    f1 = contact1_dynamics(model, x[1:end-1], u)
    f2 = contact1_dynamics(model, x[1:end-1] + 0.5 * h * f1, u)
    f3 = contact1_dynamics(model, x[1:end-1] + 0.5 * h * f2, u)
    f4 = contact1_dynamics(model, x[1:end-1] + h * f3, u)
    return [x[1:end-1] + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4); x[end] + u[end]]
end

# the input state dim should be 37!
# the input control dim should be 13!
function contact2_dynamics_rk4(model, x, u)
    # RK4 integration with zero-order hold on u
    h = u[end]
    f1 = contact2_dynamics(model, x[1:end-1], u)
    f2 = contact2_dynamics(model, x[1:end-1] + 0.5 * h * f1, u)
    f3 = contact2_dynamics(model, x[1:end-1] + 0.5 * h * f2, u)
    f4 = contact2_dynamics(model, x[1:end-1] + h * f3, u)
    return [x[1:end-1] + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4); x[end] + u[end]]
end

# the input state dim should be 37!
# the input control dim should be 13!
function contact3_dynamics_rk4(model, x, u)
    # RK4 integration with zero-order hold on u
    h = u[end]
    f1 = contact3_dynamics(model, x[1:end-1], u)
    f2 = contact3_dynamics(model, x[1:end-1] + 0.5 * h * f1, u)
    f3 = contact3_dynamics(model, x[1:end-1] + 0.5 * h * f2, u)
    f4 = contact3_dynamics(model, x[1:end-1] + h * f3, u)
    return [x[1:end-1] + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4); x[end] + u[end]]
end

# the input state dim should be 37!
# the input control dim should be 13!
function contact1_jacobian(model::RealQuadruped, x, u)
    xi = SVector{37}(1:37)
    ui = SVector{13}(1:13) .+ 37
    f(z) = contact1_dynamics_rk4(model, z[xi], z[ui])
    return ForwardDiff.jacobian(f, [x; u])
end

# the input state dim should be 37!
# the input control dim should be 13!
function contact2_jacobian(model::RealQuadruped, x, u)
    xi = SVector{37}(1:37)
    ui = SVector{13}(1:13) .+ 37
    f(z) = contact2_dynamics_rk4(model, z[xi], z[ui])
    return ForwardDiff.jacobian(f, [x; u])
end

# the input state dim should be 37!
# the input control dim should be 13!
function contact3_jacobian(model::RealQuadruped, x, u)
    xi = SVector{37}(1:37)
    ui = SVector{13}(1:13) .+ 37
    f(z) = contact3_dynamics_rk4(model, z[xi], z[ui])
    return ForwardDiff.jacobian(f, [x; u])
end

function jump1_map(x)
    # from mode 1 to 3, set v2 and v4 zero
    xn = [x[1:27]; 0.0; 0.0; 0.0; x[31:33]; 0.0; 0.0; 0.0; x[37]]
    return xn
end

function jump2_map(x)
    # from mode 2 to 3, set v1 and v3 zero
    xn = [x[1:24]; 0.0; 0.0; 0.0; x[28:30]; 0.0; 0.0; 0.0; x[34:37]]
    return xn
end

jump1_jacobian() = Diagonal(SA[1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,0, 1,1,1, 0,0,0, 1])
jump2_jacobian() = Diagonal(SA[1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 0,0,0, 1,1,1, 0,0,0, 1,1,1, 1])
