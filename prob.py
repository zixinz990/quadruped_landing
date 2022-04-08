import numpy as np


class PlanarQuadrupedLandingProblem:
    def __init__(self):
        # state x = [x_com, y_com, theta_com], body CoM position and orientation
        # control u = [F1, F2], ground reaction forces
        self.n = 3  # state dim
        self.m = 4  # control dim
        self.K = 101  # horizon
        self.tf = 2.0  # final time (sec)

        # TODO: make sure x0 is updated at each time step during falling
        self.x0 = np.zeros(3)
        # TODO: define xf
        self.xf = np.zeros(3)

        # J = x'Qx + q'x + u'Ru + r'u
        self.Q = np.zeros(self.n, self.n)
        self.q = np.zeros(self.n, 1)
        self.R = np.zeros(self.m, self.m)
        self.r = np.zeros(self.m, 1)
        self.Qf = np.zeros(self.n, self.n)
        self.qf = np.zeros(self.n, 1)

    def get_initial_trajectory(self):
        times = np.linspace(0, self.tf, num=self.N)
        # TODO: generate initial trajectory
        X = 0
        U = 0
        return X, U
