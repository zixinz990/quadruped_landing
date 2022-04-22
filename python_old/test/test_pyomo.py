# import libraries
# Pyomo stuff
from pyomo.environ import*
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition

# other
import sympy as sym
import numpy as np

from IPython.display import display  # for pretty printing

import matplotlib.pyplot as plt
import matplotlib.animation as ani
from IPython.display import HTML

# create the model
m = ConcreteModel()

# Sets
N = 100  # how many points are in the trajectory
# For defining ordered/numerical sets. Works like 'range' in python.
m.N = RangeSet(N)
m.L = RangeSet(2)

# Parameters
m.g = Param(initialize=9.81)
m.X0 = Param(initialize=0.0)  # position of fixed base
m.Y0 = Param(initialize=2.0)
m.h = Param(initialize=0.02)  # time step

m.m = Param(m.L, initialize=1.0)  # mass of links
m.len = Param(m.L, initialize=1.0)  # length of links


def calculate_In(m, l):  # function for calculating moment of intertia from mass and length
    return m.m[l]*m.len[l]**2/12
# m here is a local variable: the model given as input to the function, not necessarily our global model 'm'
# l is just an iterator: it doesn't have to have the same name as the set


m.In = Param(m.L, initialize=calculate_In)  # moment of inertia

# Variables
m.th = Var(m.N, m.L)  # position
m.dth = Var(m.N, m.L)  # velocity
m.ddth = Var(m.N, m.L)  # acceleration

m.Tc = Var(m.N)  # torque at second joint

# Constraints
# Integration constraints


def BwEuler_p(m, n, l):  # for positions
    if n > 1:
        return m.th[n, l] == m.th[n-1, l] + m.h*m.dth[n-1, l]
    else:
        # use this to leave out members of a set that the constraint doesn't apply to
        return Constraint.Skip


m.integrate_p = Constraint(m.N, m.L, rule=BwEuler_p)


def BwEuler_v(m, n, l):  # for velocities
    if n > 1:
        return m.dth[n, l] == m.dth[n-1, l] + m.h*m.ddth[n-1, l]
    else:
        return Constraint.Skip


m.integrate_v = Constraint(m.N, m.L, rule=BwEuler_v)

# Code from last time - Generates symbolic EOM for double pendulum

# create symbolic variables

# system parameters
X0, Y0 = sym.symbols(['X0', 'Y0'])  # fixed position of first link
g = sym.symbols('g')
m1, m2 = sym.symbols(['m1', 'm2'])  # mass of links
l1, l2 = sym.symbols(['l1', 'l2'])  # length of links
In1, In2 = sym.symbols(['In1', 'In2'])  # moment of intertia of links

# generalized coordinates
th1, th2 = sym.symbols(['theta1', 'theta2'])  # position
dth1, dth2 = sym.symbols(
    ['\dot{\\theta}_{1}', '\dot{\\theta}_{2}'])  # velocity
ddth1, ddth2 = sym.symbols(
    ['\ddot{\\theta}_{1}', '\ddot{\\theta}_{2}'])  # acceleration

q = sym.Matrix([[th1], [th2]])  # group into matrices
dq = sym.Matrix([[dth1], [dth2]])
ddq = sym.Matrix([[ddth1], [ddth2]])

# STEP 1: write expressions for the system space coordinates in terms of the generalized coordinates and parameters
th1a = th1  # absolute angle
th2a = th2 + th1

x1 = X0 + 0.5*l1*sym.sin(th1a)
y1 = Y0 - 0.5*l1*sym.cos(th1a)

x2 = X0 + l1*sym.sin(th1a) + 0.5*l2*sym.sin(th2a)
y2 = Y0 - l1*sym.cos(th1a) - 0.5*l2*sym.cos(th2a)

# STEP 2: generate expressions for the system space velocities
p1 = sym.Matrix([x1, y1, th1])
[dx1, dy1, dth1a] = p1.jacobian(q)*dq

p2 = sym.Matrix([x2, y2, th2a])
[dx2, dy2, dth2a] = p2.jacobian(q)*dq

# STEP 3: generate expressions for the kinetic and potential energy

T = sym.Matrix([0.5*m1*(dx1**2+dy1**2) + 0.5*m2 *
               (dx2**2+dy2**2) + 0.5*In1*dth1a**2 + 0.5*In2*dth2a**2])
V = sym.Matrix([m1*g*y1 + m2*g*y2])

# STEP 4: calculate each term of the Lagrange equation
# term 1
Lg1 = sym.zeros(1, len(q))
for i in range(len(q)):
    dT_ddq = sym.diff(T, dq[i])  # get partial of T in dq_i
    # ...then get time derivative of that partial
    Lg1[i] = dT_ddq.jacobian(q)*dq + dT_ddq.jacobian(dq)*ddq

# term 3
Lg3 = T.jacobian(q)  # partial of T in q

# term 4
Lg4 = V.jacobian(q)  # partial of U in q

# STEP 5: calculate generalized forces
# control torque
tau = sym.symbols('tau')

Ftau = sym.Matrix([[0], [0], [tau]])

rtau = sym.Matrix([[X0 + l1*sym.sin(th1)],
                  [Y0 - l1*sym.cos(th1)],
                  [th2]])

Jtau = rtau.jacobian(q)

Qtau = Jtau.transpose()*Ftau

Qall = Qtau

# put it all together
EOM = Lg1 - Lg3 + Lg4 - Qall.transpose()

sym.printing.latex(EOM[1].simplify())

# Lambdify the EOM
func_map = {'sin': sin, 'cos': cos}
# You need to tell 'lambdify' which symbolic toolbox functions = which functions from other modules.
# Here, we want the symbolic sin and cos to map to pyomo's sin and cos.
# (Yes, pyomo has its own trig functions that are distinct from numpy's or math's. You need to use them.)

sym_list = [X0, Y0, g,
            th1, th2, dth1, dth2, ddth1, ddth2, tau,
            m1, m2, l1, l2, In1, In2]  # list of the symbols that will be substituted with inputs

lambEOM1 = sym.lambdify(sym_list, EOM[0], modules=[func_map])
lambEOM2 = sym.lambdify(sym_list, EOM[1], modules=[func_map])

# create the constraints


def EOM1(m, n):  # for theta1
    # list the model versions of all quantities in the same order as sym_list
    var_list = [m.X0, m.Y0, m.g,
                m.th[n, 1], m.th[n, 2], m.dth[n, 1], m.dth[n,
                                                           2], m.ddth[n, 1], m.ddth[n, 2], m.Tc[n],
                m.m[1], m.m[2], m.len[1], m.len[2], m.In[1], m.In[2]]
    return lambEOM1(*var_list) == 0


m.EOM1 = Constraint(m.N, rule=EOM1)


def EOM2(m, n):  # for theta2
    var_list = [m.X0, m.Y0, m.g,
                m.th[n, 1], m.th[n, 2], m.dth[n, 1], m.dth[n,
                                                           2], m.ddth[n, 1], m.ddth[n, 2], m.Tc[n],
                m.m[1], m.m[2], m.len[1], m.len[2], m.In[1], m.In[2]]
    return lambEOM2(*var_list) == 0


m.EOM2 = Constraint(m.N, rule=EOM2)

# Cost function


def CostFun(m):
    torque_sum = 0
    for n in range(1, N+1):
        torque_sum += m.Tc[n]**2
    return torque_sum


m.Cost = Objective(rule=CostFun)

# variable bounds
# a mildly annoying thing about pyomo is that variables are individual objects, so you have to use a loop to bound them:
# (I think you can set up a default bound when you create the variable, though)

for n in range(1, N+1):
    m.Tc[n].setlb(-50)
    m.Tc[n].setub(50)

    for l in range(1, 3):
        m.th[n, l].setlb(-np.pi*2)  # lower bound
        m.th[n, l].setub(np.pi*2)  # upper bound


# initialization
for n in range(1, N+1):
    m.Tc[n].value = 1

    for l in range(1, 3):
        m.th[n, l].value = np.random.uniform(-np.pi, np.pi)
        m.dth[n, l].value = 1
        m.ddth[n, l].value = 10


# Boundary conditions
# you should to do these after initialization so the values you want to be fixed don't accidentally end up being changed

# initial condition
# if a variable's value is fixed, the solver treats it like a parameter
m.th[1, 1].fixed = True
m.th[1, 1].value = 0
m.th[1, 2].fixed = True
m.th[1, 2].value = 0

m.dth[1, 1].fixed = True
m.dth[1, 1].value = 0
m.dth[1, 2].fixed = True
m.dth[1, 2].value = 0

# final condition
m.th[N, 1].fixed = True
m.th[N, 1].value = np.pi
m.th[N, 2].fixed = True
m.th[N, 2].value = 0

m.dth[N, 1].fixed = True
m.dth[N, 1].value = 0
m.dth[N, 2].fixed = True
m.dth[N, 2].value = 0

# solving
opt = SolverFactory('ipopt')  # standard issue, garden variety ipopt

# If you've managed to install your own version of ipopt, you can call it like:
#opt = SolverFactory('ipopt',executable = 'C:/cygwin64/home/Stacey/CoinIpopt/build/bin/ipopt.exe')
#opt.options["linear_solver"] = 'ma86'

# solver options
# prints a log with each iteration (you want to this - it's the only way to see progress.)
opt.options["print_level"] = 5
opt.options["max_iter"] = 30000  # maximum number of iterations
opt.options["max_cpu_time"] = 300  # maximum cpu time in seconds
# the tolerance for feasibility. Considers constraints satisfied when they're within this margin.
opt.options["Tol"] = 1e-6

results = opt.solve(m, tee=True)

# For debugging:
# tells you if the solver had any errors/ warnings
print(results.solver.status)
# tells you if the solution was (locally) optimal, feasible, or neither.
print(results.solver.termination_condition)

# animate it


fig1, ax1 = plt.subplots(1, 1)  # create axes


def plot_pendulum(i, m, ax):  # update function for animation
    ax.clear()
    ax.set_xlim([-2, 2])
    ax.set_ylim([0, 4])

    # plot link 1
    L1topx = m.X0.value
    L1topy = m.Y0.value
    L1bottomx = m.X0.value + m.len[1]*np.sin(m.th[i, 1].value)
    L1bottomy = m.Y0.value - m.len[1]*np.cos(m.th[i, 1].value)
    ax.plot([L1topx, L1bottomx], [L1topy, L1bottomy], color='xkcd:black')

    # plot link 2
    L2bottomx = L1bottomx + m.len[2] * \
        np.sin(m.th[i, 1].value + m.th[i, 2].value)
    L2bottomy = L1bottomy - m.len[2] * \
        np.cos(m.th[i, 1].value + m.th[i, 2].value)
    ax.plot([L1bottomx, L2bottomx], [L1bottomy, L2bottomy], color='xkcd:black')


def update(i): return plot_pendulum(i, m, ax1)  # lambdify update function


animate = ani.FuncAnimation(
    fig1, update, range(1, N+1), interval=50, repeat=True)

plt.show()
