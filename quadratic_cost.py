import numpy as np


class QuadraticCost:
    def __init__(self, Q: np.ndarray, R: np.ndarray, q: np.ndarray, r: np.ndarray, Qf: np.ndarray, qf: np.ndarray):
        self.Q = Q
        self.R = R
        self.q = q
        self.r = r

        self.Qf = Qf
        self.qf = qf

    def stage_cost(self, x: np.ndarray, u: np.ndarray):
        Q, R, q, r = self.Q, self.R, self.q, self.r
        return 0.5 * np.matmul(np.matmul(x.transpose(), Q), x) + np.matmul(q.transpose(), x) + \
               0.5 * np.matmul(np.matmul(u.transpose(), Q), u) + np.matmul(r.transpose(), u)

    def term_cost(self, x: np.ndarray):
        Qf, qf = self.Qf, self.qf
        return 0.5 * np.matmul(np.matmul(x.transpose(), Qf), x) + np.matmul(qf.transpose(), x)

    def cost(self, X: np.ndarray, U: np.ndarray, T: int):
        # X = [x0, x1, x2, ..., xT-1, xT]
        # U = [u0, u1, u2, ..., uT-1]
        J = 0.0
        for k in range(T):  # k = 0, 1, 2, ..., T-1
            J = J + self.stage_cost(X[:, k], U[:, k])
        J = J + self.term_cost(X[:, T])
        return J
