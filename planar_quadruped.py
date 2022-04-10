import numpy as np


class PlanarQuadruped:
    def __init__(self, body_mass: float, body_length: float, thigh_length: float, calf_length: float):
        self.m = body_mass
        self.b_l = body_length
        self.t_l = thigh_length
        self.c_l = calf_length

    def dynamics(self, x: np.ndarray, u: np.ndarray, t):
        # x = [x_com, y_com, theta_com], body CoM position and orientation
        # u = [F1, F2], ground reaction forces
        # TODO: return xdot at time t
        return 0.0

    def dynamics_jacobians(self, x: np.ndarray, u: np.ndarray, t):
        # we want to get the discrete-time state space model x[k+1] = Ax[k] + Bu[k]
        # TODO: return A and B at time t
        return 0.0
