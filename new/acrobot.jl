using GeometryBasics
using CoordinateTransformations, Rotations
import RobotZoo.Acrobot
using Colors
using StaticArrays 
using MeshCat
using Blink
using LinearAlgebra

#True model with friction
function true_dynamics(model::Acrobot, x, u)
    g = 9.81
    
    #Perturb model parameters
    m1 = model.m[1] + 0.01
    m2 = model.m[2] - 0.01
    l1 = model.l[2] - 0.005
    l2 = model.l[2] + 0.004
    J1 = (1.0/12)*m1*l1*l1
    J2 = (1.0/12)*m2*l2*l2
    
    θ1,    θ2    = x[1], x[2]
    θ1dot, θ2dot = x[3], x[4]
    s1,c1 = sincos(θ1)
    s2,c2 = sincos(θ2)
    c12 = cos(θ1 + θ2)

    # mass matrix
    m11 = m1*l1^2 + J1 + m2*(l1^2 + l2^2 + 2*l1*l2*c2) + J2
    m12 = m2*(l2^2 + l1*l2*c2 + J2)
    m22 = l2^2*m2 + J2
    M = @SMatrix [m11 m12; m12 m22]

    # bias term
    tmp = l1*l2*m2*s2
    b1 = -(2 * θ1dot * θ2dot + θ2dot^2)*tmp
    b2 = tmp * θ1dot^2
    B = @SVector [b1, b2]

    # friction
    c = 1.0
    C = @SVector [0.1*tanh(5*θ1dot) + c*θ1dot, 0.1*tanh(5*θ2dot) + c*θ2dot] #add nonlinear friction to model

    # gravity term
    g1 = ((m1 + m2)*l2*c1 + m2*l2*c12) * g
    g2 = m2*l2*c12*g
    G = @SVector [g1, g2]

    # equations of motion
    τ = @SVector [0, u[1]]
    θddot = M\(τ - B - G - C)
    return @SVector [θ1dot, θ2dot, θddot[1], θddot[2]]
end

function true_dynamics_rk4(model, x, u, h)
    #RK4 integration with zero-order hold on u
    f1 = true_dynamics(model, x, u)
    f2 = true_dynamics(model, x + 0.5*h*f1, u)
    f3 = true_dynamics(model, x + 0.5*h*f2, u)
    f4 = true_dynamics(model, x + h*f3, u)
    return x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
end
