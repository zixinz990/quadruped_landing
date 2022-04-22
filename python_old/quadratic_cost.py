import numpy as np


class QuadraticCost:
    def __init__(self, Q: np.ndarray, R: np.ndarray, Qf: np.ndarray):
        self.Q = Q
        self.R = R
        # self.q = q
        # self.r = r

        self.Qf = Qf
        # self.qf = qf

    def stage_cost(self, x_err: np.ndarray, u_err: np.ndarray):
        Q, R = self.Q, self.R
        return 0.5 * np.matmul(np.matmul(x_err.transpose(), Q), x_err) + \
            0.5 * np.matmul(np.matmul(u_err.transpose(), R), u_err)

    def term_cost(self, x_err: np.ndarray):
        Qf = self.Qf
        return 0.5 * np.matmul(np.matmul(x_err.transpose(), Qf), x_err)

    def cost(self, X_err: list, U_err: list, K: int):
        # X = [x0, x1, x2, ..., xK-1, xK]
        # U = [u0, u1, u2, ..., uK-1]
        J = 0.0
        for k in range(K):  # k = 0, 1, 2, ..., K-1
            J = J + self.stage_cost(X_err[k], U_err[k])
        J = J + self.term_cost(X_err[K])
        return J
