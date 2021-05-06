from os import altsep
import numpy as np
from numpy.lib.financial import _ipmt_dispatcher
from numpy import reshape as rs
import cvxpy as cvx
import normals as nr
import rcp_simgen as simgen
import lambdas_max as lmax
import support_vecs as svc


def init_chain(F, xi, sys):
    """ Assign control inputs to the vertices of the initial simplex"""
    eps = 1e-4
    n, m =  np.shape(sys.B)
    v0 = rs(F[0, :], [n, 1])
    v1 = rs(F[1, :], [n, 1])

    # Optimization Problem
    u0 = cvx.Variable([m, 1])
    u1 = cvx.Variable([m, 1])

    constraints = []
    # Flow condition on v0
    Av0Flo = xi.T @ sys.B
    bv0Flo = xi.T @ sys.A @ v0 + xi.T @ sys.a
    constraints += [Av0Flo @ u0 + bv0Flo >= eps]
    # Flow condition on v1
    Av1Flo = xi.T @ sys.B
    bv1Flo = xi.T @ sys.A @ v1 + xi.T @ sys.a
    constraints += [Av1Flo @ u1 + bv1Flo >= eps]
    #Constraints on u
    umax = 6
    u_max = np.matrix([[6], [6]])
    M = np.kron( np . matrix ([[1] ,[ -1]]) , np . eye (m))
    p = np.ones([2*m, 1]) * umax
    constraints += [M @ u0 <= p, M @ u1 <= p]

    # Objective function calculations:
    xi_n = nr.normal_2(xi)
    # Max flow condition on v0
    Av0Mflo = xi_n.T @ sys.B
    bv0Mflo = xi_n.T @ sys.A @ v0 + xi_n.T @ sys.a
    # Max flow condition on v1
    Av1Mflo = xi_n.T @ sys.B
    bv1Mflo = xi_n.T @ sys.A @ v1 + xi_n.T @ sys.a
    # Objective Function
    obj =  cvx.norm(Av0Mflo @ u0 + bv0Mflo)  + cvx.norm(Av1Mflo @ u1 + bv1Mflo)

    # CVX problem
    prob = cvx.Problem(cvx.Minimize(obj), constraints)
    result = prob.solve()

    # u matrix
    uMat = np.zeros([n, m])
    uMat[0, :] = np.reshape(u0.value, [1, m])
    uMat[1, :] = np.reshape(u1.value, [1, m])

    return uMat


def prop_chain(F, uMat, sys, s_in, del_s):
    """ Propagates the simplex chain"""
    s_o, xi = svc.chain_sup(s_in, del_s)
    # First one
    F0 = F
    u0 = rs(uMat[0, :], [2, 1])
    alpha0 = sys.A @ rs(F0[0,:], [2, 1])  + sys.B @ u0 + sys.a
    L0max = lmax.lambda_max(rs(F0[0, :], [2, 1]), alpha0, s_in, s_o)
    S0, r0 = simgen.rcp_simgen(F0, u0, sys, xi, L0max)
    # Second one
    F1 = np.flip(F, 0)
    u1 = rs(uMat[1, :], [2, 1])
    alpha1 = sys.A @ rs(F1[0,:], [2, 1])  + sys.B @ u1 + sys.a
    L1max = lmax.lambda_max(rs(F1[0, :], [2, 1]), alpha1, s_in, s_o)
    S1, r1 = simgen.rcp_simgen(F1, u1, sys, xi, L1max)
    if r0 <= r1:
        Simplex = S0
    else:
        Simplex = S1
    Simplex.set_xi(xi)
    return Simplex



if __name__=="__main__":
    import space as spc
    import support_vecs as svc
    import system as ss

    s_in = np.matrix([[1], [-1]])
    F_init = spc.I
    del_s = 1
    s_o, xi_init = svc.chain_sup(s_in, del_s)
    uMat = init_chain(F_init, xi_init, ss.lsys)
    Sim = prop_chain(F_init, uMat, ss.lsys, s_in, del_s)
