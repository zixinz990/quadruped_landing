from nlp import *
from model import *


def eval_dynamics(nlp: NLP, Z):
    ix, iu = nlp.xinds, nlp.uinds
    for k in range(nlp.K):
        t = nlp.times[k]
        x, u = Z[ix[k]], Z[iu[k]]
        nlp.f[k] = nlp.model.dynamics(x, u, t)


def eval_dynamics_jacobians(nlp: NLP, Z):
    ix, iu = nlp.xinds, nlp.uinds
    for k in range(nlp.K):
        t = nlp.times[k]
        x, u = Z[ix[k]], Z[iu[k]]
        nlp.A[k], nlp.B[k] = nlp.model.dynamics_jacobians(x, u, t)


def eval_midpoints(nlp: NLP, Z):
    ix, iu = nlp.xinds, nlp.uinds
    for k in range(nlp.K - 1):
        h = nlp.times[k + 1] - nlp.times[k]
        t = nlp.times[k]
        x1, x2 = Z[ix[k]], Z[ix[k + 1]]
        u1, u2 = Z[iu[k]], Z[iu[k + 1]]
        f1 = nlp.f[k]
        f2 = nlp.f[k + 1]

        xm = (x1 + x2) / 2 + h / 8 * (f1 - f2)
        um = (u1 + u2) / 2

        fm = nlp.model.dynamics(xm, um, t + h / 2)

        nlp.fm[k] = fm
        nlp.xm[k] = xm
        nlp.um[k] = um


def eval_midpoint_jacobians(nlp: NLP, Z):
    for k in range(nlp.K - 1):
        h = nlp.times[k + 1] - nlp.times[k]
        t = nlp.times[k]
        A1, A2 = nlp.A[k], nlp.A[k + 1]
        B1, B2 = nlp.B[k], nlp.B[k + 1]

        xm, um = nlp.xm[k], nlp.um[k]

        Am, Bm = nlp.model.dynamics_jacobians(xm, um, t + h / 2)

        dxmx1 = (np.identity(nlp.n) / 2 + h / 8 * A1)  # (n,n)
        dxmu1 = h / 8 * B1  # (n,m)
        dxmx2 = (np.identity(nlp.n) / 2 - h / 8 * A2)  # (n,n)
        dxmu2 = -h / 8 * B2  # (n,m)

        nlp.Am[k][0], nlp.Am[k][1], nlp.Am[k][2] = Am, dxmx1, dxmx2
        nlp.Bm[k][0], nlp.Bm[k][1], nlp.Bm[k][2] = Bm, dxmu1, dxmu2
