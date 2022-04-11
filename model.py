import numpy as np


class PlanarQuadruped:
    def __init__(self, body_mass: float, body_length: float, thigh_length: float, calf_length: float):
        self.m = body_mass
        self.b_l = body_length
        self.t_l = thigh_length
        self.c_l = calf_length
        self.g = 9.8

    def dynamics(self, x: np.ndarray, u: np.ndarray, r: np.ndarray):
        """
        :param x = [x_com, y_com, theta_com, x_com_dot, y_com_dot, theta_com_dot], body CoM position and orientation, column vector with dim 6
        :param u = [F1, F2], ground reaction forces, column vector with dim 4
        :param r = [r1, r2], foot positions, column vector with dim 4
        :return: x_dot = [x_com_dot, y_com_dot, theta_com_dot, x_com_ddot, y_com_ddot, theta_com_ddot], column vector
        """
        x_dot = np.zeros((6, 1))

        F_1x = u[0]
        F_1y = u[1]
        F_1 = u[0:2]

        F_2x = u[2]
        F_2y = u[3]
        F_2 = u[2:4]

        r_1x = r[0]
        r_1y = r[1]
        r_2x = r[2]
        r_2y = r[3]

        x_dot[0:3] = x[3:6]  # x_dot, y_dot, theta_dot 0 1 2
        x_dot[3:5] = F_1 / self.m + F_2 / self.m + np.array([[0], [self.g]])  # x_ddot, y_ddot 3 4
        x_dot[5] = (F_1y * r_1x - F_1x * r_1y + F_2y * r_2x - F_2x * r_2y) / (self.m * self.b_l * self.b_l / 12)  # theta_ddot 5

        return x_dot

    def dynamics_jacobians(self, x: np.ndarray, u: np.ndarray, r: np.ndarray):
        """
        :param x = [x_com, y_com, theta_com, x_com_dot, y_com_dot, theta_com_dot], body CoM position and orientation, column vector with dim 6
        :param u = [F1, F2], ground reaction forces, column vector with dim 4
        :param r = [r1, r2], foot positions, column vector with dim 4
        :return: A = df/dx, matrix with dimension 6 x 6
                 B = df/du, matrix with dimension 6 x 4
        """
        # we want to get the discrete-time state space model x[k+1] = Ax[k] + Bu[k]
        F_1x = u[0]
        F_1y = u[1]
        F_1 = u[0:2]

        F_2x = u[2]
        F_2y = u[3]
        F_2 = u[2:4]

        r_1x = r[0]
        r_1y = r[1]
        r_2x = r[2]
        r_2y = r[3]

        A = np.zeros((6, 6))
        A[0, 3] = 1.0
        A[1, 4] = 1.0
        A[2, 5] = 1.0

        B = np.zeros((6, 4))
        B[3, 0] = 1 / self.m
        B[3, 2] = 1 / self.m
        B[4, 1] = 1 / self.m
        B[4, 3] = 1 / self.m
        B[5, 0] = -12 * r_1y / (self.m * self.b_l * self.b_l)
        B[5, 1] = 12 * r_1x / (self.m * self.b_l * self.b_l)
        B[5, 2] = -12 * r_2y / (self.m * self.b_l * self.b_l)
        B[5, 3] = 12 * r_2x / (self.m * self.b_l * self.b_l)
        return A, B
