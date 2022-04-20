import numpy as np


class PlanarQuadrupedLanding:
    def __init__(self, body_mass: float, body_length: float, thigh_length: float, calf_length: float, x0, xf, initial_mode, t_trans):
        self.m = body_mass
        self.L = body_length
        self.l1 = thigh_length
        self.l2 = calf_length

        self.g = 9.8

        self.n = 10  # dim of state vector
        self.m = 8  # dim of input vector

        self.x0 = x0
        self.xf = xf

        self.K = 101  # horizon
        self.T = 2.0  # final time (sec)
        self.times = np.linspace(0, self.T, num=self.K)

        self.dt = self.times[1]
        self.k_trans = t_trans / self.dt # mode transition happens

        self.initial_mode = initial_mode

        self.contact_schedule = initial_mode * np.ones(self.K)
        self.contact_schedule[self.k_trans:] = 3 * np.ones(self.K - self.k_trans)

    def dynamics(self, x: np.ndarray, u: np.ndarray):
        """
        :param x: state at this timestep
        :param u: control input at this timestep
        :return: xdot at this timestep
        """
        x_dot = np.zeros((self.n, 1))

        xb = x[0, 0]
        yb = x[1, 0]
        theta = x[2, 0]

        xb_dot = x[3, 0]
        xy_dot = x[4, 0]
        theta_dot = x[5, 0]

        x1 = x[6, 0]
        x2 = x[7, 0]
        y1 = x[8, 0]
        y2 = x[9, 0]

        x1_dot = u[0, 0]
        y1_dot = u[1, 0]
        x2_dot = u[2, 0]
        y2_dot = u[3, 0]

        F1_x = u[4, 0]
        F1_y = u[5, 0]
        F2_x = u[6, 0]
        F2_y = u[7, 0]

        Ib = self.m * self.L * self.L / 12

        p1_x = x1 - xb
        p1_y = y1 - yb

        p2_x = x2 - xb
        p2_y = y2 - yb

        tau_F = -F1_x*p1_y + F1_y*p1_x - F2_x*p2_y + F2_y*p2_x

        x_dot[0, 0] = xb_dot
        x_dot[1, 0] = xy_dot
        x_dot[2, 0] = theta_dot

        x_dot[3, 0] = (F1_x + F2_x) / self.m + self.g
        x_dot[4, 0] = (F1_y + F2_y) / self.m + self.g
        x_dot[5, 0] = tau_F / Ib

        x_dot[6, 0] = x1_dot
        x_dot[7, 0] = y1_dot
        x_dot[8, 0] = x2_dot
        x_dot[9, 0] = y2_dot

        return x_dot

    def dynamics_rk4(self, x: np.ndarray, u: np.ndarray):
        f1 = self.dynamics(x, u)
        f2 = self.dynamics(x + 0.5*self.dt*f1, u)
        f3 = self.dynamics(x + 0.5*self.dt*f2, u)
        f4 = self.dynamics(x + self.dt*f3, u)

        return x + (self.dt/6.0)*(f1 + 2*f2 + 2*f3 + f4)

    def rollout(self, x0: np.ndarray, U: list):
        X = [np.zeros((self.n, 1)) for k in range(self.K)]
        X[0] = x0

        self.dt = self.times[1]

        for k in range(self.K-1):
            X[k+1] = self.dynamics_rk4(X[k], U[k], self.dt)

        return X

    def contact_schedule(self, initial_mode):
        t = 0.5  # mode transition happens
        self.dt = self.times[1]
        k = t / self.dt

        schedule = initial_mode * np.ones(self.K)
        schedule[k:] = 3 * np.ones(self.K - k)

    def get_initial_trajectory(self, x0: np.ndarray, xN: np.ndarray, initial_mode):
        # TODO: generate initial trajectory
        X = [np.zeros((self.n, 1)) for k in range(self.K)]
        U = [np.zeros((self.m, 1)) for k in range(self.K)]

        if initial_mode == 1:
            for k in range(self.K):
                X[k][7, 0] = 0.0
                U[k][1, 0] = 0.0
        else:
            for k in range(self.K):
                X[k][9, 0] = 0.0
                U[k][3, 0] = 0.0

        return X, U
