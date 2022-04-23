import Pkg; Pkg.activate(joinpath(@__DIR__,"..")); Pkg.instantiate()
using LinearAlgebra
using ForwardDiff
using RobotDynamics
using Ipopt
using MathOptInterface
using TrajOptPlots
const MOI = MathOptInterface
using Random
using DataFrames
using DelimitedFiles
using CSV
include("quadratic_cost.jl")
include("planar_quadruped.jl")
include("sparseblocks.jl")
include("utils.jl")
include("nlp.jl")
include("moi.jl")
include("costs.jl")
include("constraints.jl")

# Base.@kwdef struct PlanarQuadruped <: AbstractModel
#     g::Float64 = 9.81   # gravity
#     mb::Float64 = 10.0  # body mass
#     mf::Float64 = 0.1   # foot mass
#     lb::Float64 = 1.0   # body length
#     l1::Float64 = 0.3   # thigh length
#     l2::Float64 = 0.3   # calf length
# end
model = PlanarQuadruped()
mb = 10.0 
g = -9.81

x0 = [-0.5; 0.3; -0.5; 0; 0; -1; 0.2; 0; -0.3; 0; 0; 0; 0; -0.3]
u1 = [0; -mb*g/2; 0; 0]
u2 = [0; -mb*g/2 + 30; 0; -mb*g/2 + 30]

T = 100
h = 0.05
X = zeros(T, size(x0)[1])
X[1,:] .= x0
global on_the_ground_flag = false
global u = u1
for i = 1:T-1
    if on_the_ground_flag
        solution = contact3_dynamics_rk4(model, X[i,:], u, h)
    else
        solution = contact1_dynamics_rk4(model, X[i,:], u, h)
    end
    if solution[7] < 0
        solution = jump1_map(solution)
        global on_the_ground_flag = true
        global u = u2
    end
    X[i+1, :] .= solution
end

writedlm( "data.csv",  X, ',')