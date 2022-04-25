import Pkg; Pkg.activate(joinpath(@__DIR__,"..")); Pkg.instantiate()
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
include("utils.jl")
include("nlp.jl")
include("moi.jl")
include("costs.jl")
include("constraints.jl")
include("ref_traj.jl")
2000
g, mb, lb, l1, l2 = model.g, model.mb, model.lb, model.l1, model.l2

# Discretization
tf = 0.6
dt = 0.02
N = Int(ceil(tf/dt)) + 1
times = range(0,tf, length=N)
t_trans = 0.2

n = 14
m = 4

# Initial Conditions
# currently, we assume the initial mode ID is 1
# xinit = [-0.4;0.6;-3.1415/9;        0.0;0.0; -1.0;0.2; 0.0;-0.5;0.0; 0.0;0.0; 0.0;-0.5]
# xterm = [-lb/2;sqrt(l1^2+l2^2);0.0; 0.0;0.0; -lb;0.0;  0.0;0.0;0.0;  0.0;0.0; 0.0;0.0]
xinit = zeros(n)
xinit[1] = -lb/2                  # xb
xinit[2] = sqrt(l1^2+l2^2) + 0.05 # yb
xinit[3] = 10 * pi / 180          # theta
xinit[6] = -lb                    # x2
xinit[7] = 0.05                   # y2
xinit[9] = -3.0                   # yb_dot
xinit[12] = -3.0                  # y2_dot

xterm = zeros(n)
xterm[1] = -lb/2           # xb
xterm[2] = sqrt(l1^2+l2^2) # yb
xterm[6] = -lb             # x2

init_mode = 1

# Reference Trajectory
Xref, Uref = reference_trajectory(model, times, t_trans, xinit, xterm, init_mode)

# Objective
Random.seed!(1)
Q = Diagonal([1.0;1.0;1.0; 1.0;1.0; 1.0;1.0; 1.0;1.0;1.0; 1.0;1.0; 1.0;1.0])
R = Diagonal(fill(1e-3,4))
Qf = Q

obj = map(1:N-1) do k
    LQRCost(Q,R,Xref[k],Uref[k])
end
push!(obj, LQRCost(Qf, R*0, Xref[N], Uref[1]))

# Define the NLP
nlp = HybridNLP(model, obj, init_mode, tf, N, Xref[1], Xref[end]);

# Initial guess
xtransit = zeros(n)
xtransit[1] = -lb/2 # xb
xtransit[2] = l1    # yb
xtransit[3] = 0     # theta
xtransit[6] = -lb   # x2

# Random.seed!(1)
# Uguess = [u + 0.1*randn(length(u)) for u in Uref]
# Xguess = [x + 0.1*randn(length(x)) for x in Xref]

Uguess = [zeros(length(u)) for u in Uref]
Xguess = [zeros(length(x)) for x in Xref]

timesteps_phase_1 = nlp.k_trans

for i = 1:timesteps_phase_1
    Xguess[i] = xinit + (xtransit - xinit)/timesteps_phase_1 * i
end

# display((xtransit - xinit)/timesteps_phase_1)

Z0 = packZ(nlp, Xguess, Uguess);
# nlp = HybridNLP(model, obj, init_mode, tf, N, Xref[1], Xref[end], use_sparse_jacobian=false);

# Z_sol, solver = solve(Z0, nlp, c_tol=1e-6, tol=1e-6)
Z_sol, solver = solve(Z0, nlp, c_tol=1e-4, tol=1e-2)

@show Δx0 = Z_sol[1:14] - xinit
@show Δxf = Z_sol[end-13:end] - xterm
