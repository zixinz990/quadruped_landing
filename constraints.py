from nlp import *


# TODO: modify formulation in the google drive
def dynamics_constraints(nlp: NLP, Z):
    X, U = nlp.unpackZ(Z)

    n = nlp.n
    K = nlp.K

    dynamics_C = np.zeros((K-1)*n)

    for k in range(K-1):
        # check one step simulation
        dynamics_C[k*n:(k+1)*n] = nlp.prob.dynamics_rk4(X[k], U[k]) - X[k+1]

    return dynamics_C


def contact_constraints(nlp: NLP, Z):
    X, U = nlp.unpackZ(Z)

    K = nlp.K
    k_trans = nlp.prob.k_trans
    initial_mode = nlp.prob.initial_mode

    contact_C = np.zeros(4*K-2*k_trans)

    # contact constraints for the early-landing foot
    for k in range(K):
        if initial_mode == 1:
            # y1 = 0
            contact_C[k*2] = X[k][7]
            # y1_dot = 0
            contact_C[k*2+1] = U[k][1]
        else:
            # y2 = 0
            contact_C[k*2] = X[k][9]
            # y2_dot = 0
            contact_C[k*2+1] = U[k][3]

    # contact constraints for the later-landing foot
    for k in range(k_trans, K):
        if initial_mode == 1:
            # y2 = 0
            contact_C[2*K+2*(k-k_trans)+1] = X[k][9]
            # y2_dot = 0
            contact_C[2*K+2*(k-k_trans)+2] = U[k][3]
        else:
            # y1 = 0
            contact_C[2*K+2*(k-k_trans)+1] = X[k][7]
            # y1_dot = 0
            contact_C[2*K+2*(k-k_trans)+2] = X[k][1]

    return contact_C


def kinematics_constraints(nlp: NLP, Z):
    # TODO
    return 0


# def eval_c(nlp: NLP, Z):
#     """
#     :param nlp:
#     :param Z:
#     :return: c: an array with dim n(K+1) x 1
#     """
#     K = nlp.K
#     n = nlp.n
#     xi, ui = nlp.xinds, nlp.uinds
#     idx = xi[0]  # [0, 1, 2]

#     c = np.zeros((n * (K + 1), 1))
#     c[idx] = Z[xi[0]] - nlp.x0

#     eval_dynamics(nlp, Z)
#     eval_midpoints(nlp, Z)

#     for k in range(K - 1):
#         # [3, 4, 5], [6, 7, 8], ..., [3(K-1), 3(K-1)+1, 3(K-1)+2]
#         idx = idx + n
#         x1, x2 = Z[xi[k]], Z[xi[k + 1]]
#         h = nlp.times[k + 1] - nlp.times[k]

#         f1 = nlp.f[k]
#         f2 = nlp.f[k + 1]
#         fm = nlp.fm[k]
#         c[idx] = h / 6 * (f1 + 4 * fm + f2) + x1 - x2

#     idx = idx + n  # [3K, 3K+1, 3K+2]
#     c[idx] = Z[xi[K]] - nlp.xf

#     return c


# def jac_c(nlp: NLP, Z):
#     """
#     :param nlp:
#     :param Z:
#     :return: jac: a matrix with dim n(K+1) x (n+m)K
#     """
#     eval_dynamics_jacobians(nlp, Z)
#     eval_midpoint_jacobians(nlp, Z)

#     K = nlp.K
#     n = nlp.n
#     m = nlp.m

#     jac = np.zeros((n * (K + 1), (n + m) * K))

#     for i in range(n):
#         jac[i, i] = 1

#     xi, ui = nlp.xinds, nlp.uinds

#     idx = xi[0]  # [0, 1, 2]

#     for k in range(K - 1):
#         # [3, 4, 5], [6, 7, 8], ..., [3(K-1), 3(K-1)+1, 3(K-1)+2]
#         idx = idx + n
#         h = nlp.times[k + 1] - nlp.times[k]

#         A1 = nlp.A[k]
#         B1 = nlp.B[k]
#         A2 = nlp.A[k + 1]
#         B2 = nlp.B[k + 1]
#         Am = nlp.Am[k, 1]
#         Bm = nlp.Bm[k, 1]

#         dxmx1 = nlp.Am[k, 2]
#         dxmx2 = nlp.Am[k, 3]
#         dxmu1 = nlp.Bm[k, 2]
#         dxmu2 = nlp.Bm[k, 3]
#         dumu1 = np.identity(nlp.m) / 2
#         dumu2 = np.identity(nlp.m) / 2

#         jac[idx, xi[k]] = h / 6 * (A1 + 4 * Am * dxmx1) + np.identity(n)
#         jac[idx, ui[k]] = h / 6 * (B1 + 4 * Bm * dumu1 + 4 * Am * dxmu1)
#         jac[idx, xi[k + 1]] = h / 6 * (4 * Am * dxmx2 + A2) - np.identity(n)
#         jac[idx, ui[k + 1]] = h / 6 * (4 * Bm * dumu2 + 4 * Am * dxmu2 + B2)

#     idx = idx + n  # [3K, 3K+1, 3K+2]

#     # TODO: Terminal constraint
#     for i in range(n):
#         jac[idx[i], xi[K - 1][i]] = 1

#     return jac
