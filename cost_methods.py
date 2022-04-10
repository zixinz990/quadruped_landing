from nlp import *
from dynamics import *
from planar_quadruped import *
from quadratic_cost import *


def eval_f(nlp: NLP, Z):
    K = nlp.K
    J = 0.0
    eval_midpoints(nlp, Z)

    ix, iu = nlp.xinds, nlp.uinds
    for k in range(K - 1):
        x1, u1 = Z[ix[k]], Z[iu[k]]
        x2, u2 = Z[ix[k + 1]], Z[iu[k + 1]]

        h = nlp.times[k + 1] - nlp.times[k]

        J += h / 6 * (nlp.cost.stage_cost(x1, u1) + 4 * nlp.cost.stage_cost(nlp.xm[k], nlp.um[k]) + nlp.cost.stage_cost(x2, u2))
    J += nlp.cost.term_cost(Z[ix[K - 1]])
    return J


def grad_f(nlp: NLP, Z):
    eval_dynamics(nlp, Z)
    eval_midpoints(nlp, Z)
    eval_dynamics_jacobians(nlp, Z)
    eval_midpoint_jacobians(nlp, Z)

    ix, iu = nlp.xinds, nlp.uinds
    n = nlp.n
    m = nlp.m
    K = nlp.K

    Q = nlp.cost.Q
    q = nlp.cost.q
    R = nlp.cost.R
    r = nlp.cost.r

    Qf = nlp.cost.Qf
    qf = nlp.cost.qf

    grad = np.array(((n + m) * K, 1))

    for k in range(K - 1):
        x1, x2 = Z[ix[k]], Z[ix[k + 1]]
        u1, u2 = Z[iu[k]], Z[iu[k + 1]]
        xm = nlp.xm[k]
        um = nlp.um[k]
        h = nlp.times[k + 1] - nlp.times[k]

        # TASK: Compute the cost gradient
        dxmx1 = nlp.Am[k, 2]
        dxmx2 = nlp.Am[k, 3]

        dxmu1 = nlp.Bm[k, 2]
        dxmu2 = nlp.Bm[k, 3]

        dJ1x1 = Q * x1 + q
        dJ2x2 = Q * x2 + q
        dJmxm = Q * xm + q

        dJ1u1 = R * u1 + r
        dJ2u2 = R * u2 + r
        dJmum = R * um + r

        grad[ix[k]] = grad[ix[k]] + h / 6.0 * (dJ1x1 + 4.0 * dxmx1.transpose() * dJmxm)
        grad[iu[k]] = grad[iu[k]] + h / 6.0 * (dJ1u1 + 4.0 * dxmu1.transpose() * dJmxm + 4.0 * 0.5 * dJmum)

        grad[ix[k + 1]] = grad[ix[k + 1]] + h / 6.0 * (dJ2x2 + 4.0 * dxmx2.transpose() * dJmxm)
        grad[iu[k + 1]] = grad[iu[k + 1]] + h / 6.0 * (dJ2u2 + 4.0 * dxmu2.transpose() * dJmxm + 4.0 * 0.5 * dJmum)

    grad[ix[K - 1]] = grad[ix[K - 1]] + Qf * Z[ix[K - 1]] + qf

    return grad
