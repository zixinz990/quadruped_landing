import numpy as np
from prob import PlanarQuadrupedLandingProblem


class NLP:
    def __init__(self, prob: PlanarQuadrupedLandingProblem):
        # x[k+1] = Ax[k] + Bu[k]
        # A: n x n
        # x: n x 1
        # B: n x m
        # u: m x 1
        self.f = [np.zeros(prob.n) for k in range(prob.K)]

        self.A = [np.zeros((prob.n, prob.n)) for k in range(prob.K)]
        self.B = [np.zeros((prob.n, prob.m)) for k in range(prob.K)]

        self.Am = [np.zeros((prob.n, prob.n)) for k in range(prob.K)]
        self.Bm = [np.zeros((prob.n, prob.m)) for k in range(prob.K)]

        self.fm = [np.zeros(prob.n) for k in range(prob.K)]
        self.xm = [np.zeros(prob.n) for k in range(prob.K)]
        self.um = [np.zeros(prob.m) for k in range(prob.K)]

        self.Np = (prob.n + prob.m) * prob.K
        self.Nd = prob.n * (prob.K + 1)
