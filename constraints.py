from dynamics import *


def eval_c(nlp: NLP, model: PlanarQuadruped, Z):
    """
    :param nlp:
    :param model:
    :param Z:
    :return: c: an array with dim n(K+1) x 1
    """
    K = nlp.K
    n = nlp.n
    xi, ui = nlp.xinds, nlp.uinds
    idx = xi[0]  # [0, 1, 2]

    c = np.zeros((n * (K + 1), 1))
    c[idx] = Z[xi[0]] - nlp.x0

    eval_dynamics(nlp, model, Z)
    eval_midpoints(nlp, model, Z)

    for k in range(K - 1):
        idx = idx + n  # [3, 4, 5], [6, 7, 8], ..., [3(K-1), 3(K-1)+1, 3(K-1)+2]
        x1, x2 = Z[xi[k]], Z[xi[k + 1]]
        h = nlp.times[k + 1] - nlp.times[k]

        f1 = nlp.f[k]
        f2 = nlp.f[k + 1]
        fm = nlp.fm[k]
        c[idx] = h / 6 * (f1 + 4 * fm + f2) + x1 - x2

    idx = idx + n  # [3K, 3K+1, 3K+2]
    c[idx] = Z[xi[K]] - nlp.xf

    return c


def jac_c(nlp: NLP, model: PlanarQuadruped, Z):
    """
    :param nlp:
    :param model:
    :param Z:
    :return: jac: a matrix with dim n(K+1) x (n+m)K
    """
    eval_dynamics_jacobians(nlp, model, Z)
    eval_midpoint_jacobians(nlp, model, Z)

    K = nlp.K
    n = nlp.n
    m = nlp.m

    jac = np.zeros((n * (K + 1), (n + m) * K))

    for i in range(n):
        jac[i, i] = 1

    xi, ui = nlp.xinds, nlp.uinds

    idx = xi[0]  # [0, 1, 2]

    for k in range(K - 1):
        idx = idx + n  # [3, 4, 5], [6, 7, 8], ..., [3(K-1), 3(K-1)+1, 3(K-1)+2]
        h = nlp.times[k + 1] - nlp.times[k]

        A1 = nlp.A[k]
        B1 = nlp.B[k]
        A2 = nlp.A[k + 1]
        B2 = nlp.B[k + 1]
        Am = nlp.Am[k, 1]
        Bm = nlp.Bm[k, 1]

        dxmx1 = nlp.Am[k, 2]
        dxmx2 = nlp.Am[k, 3]
        dxmu1 = nlp.Bm[k, 2]
        dxmu2 = nlp.Bm[k, 3]
        dumu1 = np.identity(nlp.m) / 2
        dumu2 = np.identity(nlp.m) / 2

        jac[idx, xi[k]] = h / 6 * (A1 + 4 * Am * dxmx1) + np.identity(n)
        jac[idx, ui[k]] = h / 6 * (B1 + 4 * Bm * dumu1 + 4 * Am * dxmu1)
        jac[idx, xi[k + 1]] = h / 6 * (4 * Am * dxmx2 + A2) - np.identity(n)
        jac[idx, ui[k + 1]] = h / 6 * (4 * Bm * dumu2 + 4 * Am * dxmu2 + B2)

    idx = idx + n  # [3K, 3K+1, 3K+2]

    # TODO: Terminal constraint
    for i in range(n):
        jac[idx[i], xi[K - 1][i]] = 1

    return jac
