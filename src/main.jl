import Pkg;
Pkg.activate(joinpath(@__DIR__, ".."));
Pkg.instantiate();
using DelimitedFiles
using CSV
using LinearAlgebra
using ForwardDiff
using RobotDynamics
using Ipopt
using MathOptInterface
const MOI = MathOptInterface
using Random;

include("quadratic_cost.jl")
include("planar_quadruped.jl")
include("sparseblocks.jl")
include("nlp.jl")
include("moi.jl")
include("costs.jl")
include("constraints.jl")
include("ref_traj.jl");

# Dynamics model
model = PlanarQuadruped()
g, mb, lb, l1, l2 = model.g, model.mb, model.lb, model.l1, model.l2

# Discretization
dt = 0.01
N = 51
times = range(0, dt * (N - 1), length=N)
k_trans = 21
n = 15
m = 5;

# Initial condition. Currently, we assume the initial mode ID is 1
xinit = zeros(n)
xinit[1] = -lb / 2.5                # xb
xinit[2] = sqrt(l1^2 + l2^2) + 0.1  # yb
xinit[3] = -20 * pi / 180           # theta
xinit[6] = -lb                      # x2
xinit[7] = 0.25                     # y2
xinit[9] = -2.0                     # yb_dot
xinit[12] = -2.0                    # y2_dot

# Desired final state
xterm = zeros(n)
xterm[1] = -lb / 2           # xb
xterm[2] = sqrt(l1^2 + l2^2) # yb
xterm[6] = -lb               # x2

init_mode = 1

# Reference Trajectory
Xref, Uref = reference_trajectory(model, N, k_trans, xterm, init_mode, dt);

# Objective
Q = Diagonal([10.0; 10.0; 1.0; 10.0; 10.0; 10.0; 10.0; 10.0; 10.0; 1.0; 10.0; 10.0; 10.0; 10.0; 0.0])
R = Diagonal(fill(1e-3, 5))
R[end, end] = 0.0
Qf = Q

obj = map(1:N-1) do k
    LQRCost(Q, R, Xref[k], Uref[k])
end
push!(obj, LQRCost(Qf, R * 0, Xref[N], Uref[1]))

# Define the NLP
nlp = HybridNLP(model, obj, init_mode, k_trans, N, xinit, xterm);

# Initial guess
Random.seed!(1)

# initialize Xguess
Xguess = [zeros(n) for x in Xref]
k_trans = nlp.k_trans

for k = 1:N
    if k <= k_trans
        Xguess[k] = xinit + (xterm - xinit) / (k_trans - 1) * (k - 1)
    else
        Xguess[k][1:14] = xterm[1:14]
    end
    
    Xguess[k][end] = dt * (k - 1)
end

# initialize Uguess
Uguess = [zeros(m) + rand(m) for u in Uref]
for k = 1: N-1
    Uguess[k][end] = dt
end

# pack x and u
Z0 = packZ(nlp, Xguess, Uguess)

# solve!
Z_sol, solver = solve(Z0, nlp, c_tol=1e-4, tol=1e-3)

Z_sol[1:15] - xinit

Z_sol[end-14:end] - xterm

y2 = zeros(N)
for k = 1:N
    y2[k] = Z_sol[7+20*(k-1)]
end
@show y2
