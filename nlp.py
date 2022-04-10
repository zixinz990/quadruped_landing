from prob import *
from quadratic_cost import *
from planar_quadruped import *


class NLP:
    def __init__(self, prob: PlanarQuadrupedLandingProblem, model: PlanarQuadruped, cost: QuadraticCost):
        # x[k+1] = Ax[k] + Bu[k]
        # A: n x n
        # x: n x 1
        # B: n x m
        # u: m x 1
        self.prob = prob
        self.model = model
        self.cost = cost

        self.K = prob.K
        self.n = prob.n
        self.m = prob.m

        self.x0 = prob.x0
        self.xf = prob.xf

        self.times = np.linspace(0, prob.tf, num=self.K)

        # xinds: a list of length K, each element is an array
        # [0, 1, 2], [7, 8, 9], [14, 15, 16], ..., [(K-1)(n+m), (K-1)(n+m+1), (K-1)(n+m+2)]
        self.xinds = [np.arange(self.n) + k * (self.n + self.m) for k in range(self.K)]

        # uinds: a list of length K, each element is an array
        # [3, 4, 5, 6], [10, 11, 12, 13], [17, 18, 19, 20], ..., [(K-1)(n+m)+3, (K-1)(n+m+1)+3, (K-1)(n+m+2)+3, (K-1)(n+m+2)+4]
        self.uinds = [np.arange(self.n, self.n + self.m) + k * (self.n + self.m) for k in range(self.K)]

        self.f = [np.zeros((self.n, 1)) for k in range(self.K)]

        self.A = [np.zeros((self.n, self.n)) for k in range(self.K)]
        self.B = [np.zeros((self.n, self.m)) for k in range(self.K)]

        self.Am = [[np.zeros((self.n, self.n)) for i in range(3)] for k in range(self.K)]
        self.Bm = [[np.zeros((self.n, self.m)) for i in range(3)] for k in range(self.K)]

        self.fm = [np.zeros((self.n, 1)) for k in range(self.K)]
        self.xm = [np.zeros((self.n, 1)) for k in range(self.K)]
        self.um = [np.zeros((self.m, 1)) for k in range(self.K)]

        self.Np = (self.n + self.m) * self.K
        self.Nd = self.n * (self.K + 1)

        self.num_primals = (self.n + self.m) * self.K
        self.num_eq = self.n * self.K + self.n
        self.num_ineq = 0
        self.num_duals = self.num_eq + self.num_ineq

    def packZ(self, X: list, U: list):
        """
        :param X: [x0, x1, x2, ..., xK-2, xK-1], the trajectory of states, a list of length K
        :param U: [u0, u1, u2, ..., uK-2, uK-1], the trajectory of controls, a list of length K
        :return: Z: a 1D array contains all the elements in X and U
        """
        Z = np.zeros(self.num_primals)
        for k in range(self.K):
            Z[self.xinds[k]] = X[k]
            Z[self.uinds[k]] = U[k]
        return Z

    def unpackZ(self, Z: np.ndarray):
        """
        :param Z: a 1D array contains all the elements in X and U
        :return: X: [x0, x1, x2, ..., xK-2, xK-1], the trajectory of states, a list of length K
                 U: [u0, u1, u2, ..., uK-2, uK-1], the trajectory of controls, a list of length K
        """
        X = [Z[xi] for xi in self.xinds]
        U = [Z[ui] for ui in self.uinds]
        return X, U
