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
using Random

include("quadratic_cost.jl")
include("planar_quadruped.jl")
include("sparseblocks.jl")
include("nlp.jl")
include("moi.jl")
include("costs.jl")
include("constraints.jl")
include("ref_traj.jl")

# Dynamics model
model = PlanarQuadruped()
g, mb, lb, l1, l2 = model.g, model.mb, model.lb, model.l1, model.l2

# Discretization
dt = 0.01
N = 101
times = range(0, dt * (N - 1), length=N)
k_trans = 21
n = 15
m = 5;

# Initial condition. Currently, we assume the initial mode ID is 1
xinit = zeros(n)
xinit[1] = -lb / 2.5                # xb
xinit[2] = sqrt(l1^2 + l2^2) + 0.05 # yb
xinit[3] = -30 * pi / 180           # theta
xinit[6] = -lb                      # x2
xinit[7] = 0.25                     # y2
xinit[9] = -1.0                     # yb_dot
xinit[12] = -1.0                    # y2_dot

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
Qf = Q

obj = map(1:N-1) do k
    LQRCost(Q, R, Xref[k], Uref[k])
end
push!(obj, LQRCost(Qf, R * 0, Xref[N], Uref[1]))

# Define the NLP
nlp = HybridNLP(model, obj, init_mode, k_trans, N, xinit, xterm);

# Initial guess
xtransit = zeros(n)
xtransit[1] = -lb / 2 # xb
xtransit[2] = l1      # yb
xtransit[3] = 0       # theta
xtransit[6] = -lb     # x2

Random.seed!(1)

# Uguess = [u + 0.1*randn(length(u)) for u in Uref]
# Xguess = [x + 0.1*randn(length(x)) for x in Xref]

Uguess = [zeros(m) + rand(m) for u in Uref]
Xguess = [zeros(n) for x in Xref]

k_trans = nlp.k_trans

for i = 1:k_trans
    Xguess[i] = xinit + (xtransit - xinit) / k_trans * i + 0.1 * rand(n)
end

# display((xtransit - xinit)/timesteps_phase_1)

Z0 = packZ(nlp, Xguess, Uguess);

# Z_sol, solver = solve(Z0, nlp, c_tol=1e-6, tol=1e-6)
Z_sol, solver = solve(Z0, nlp, c_tol=1e-4, tol=1e-3)

Z_sol[1:15] - xinit

Z_sol[end-14:end] - xterm

y2 = zeros(N)
for k = 1:N
    y2[k] = Z_sol[7+20*(k-1)]
end
@show y2