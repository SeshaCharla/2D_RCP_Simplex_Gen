import numpy as np
from numpy import reshape as rs
import cvxpy as cvx
import normals as nr
import rcp_simgen as simgen
import lambdas_max as lmax
import support_vecs as svc
import space as spc


def init_chain(n, asys, F, xi, Lmax, u_max, u_min, phi):
    """ Create initial simplex"""
    eps = 1e-6
    *_, m =  np.shape(asys.B)

    # Optimization problem
    u = [cvx.Variable((m, 1)) for i in range(n)]
    constraints = []
    obj = 0
    for i in range(n):
        obj += xi.T @ (asys.A @ rs(F[i, :], [n, 1]) + asys.B @ u[i] + asys.a)
        # Flow constraints
        constraints += [xi.T @ (asys.A @ rs(F[i, :], [n, 1]) + asys.B @ u[i] + asys.a) >= eps]
        # input constraints
        constraints += [u[i] <= u_max, u[i]>= u_min]
    prob = cvx.Problem(cvx.Maximize(obj), constraints=constraints)
    if not prob.is_dcp():
        raise(ValueError("The problem doesn't follow DCP rules!!"))
    prob.solve()
    if prob.status in ["infeasible", "unbounded"]:
        raise(ValueError("The optimization problem is {}.\nCheck control input Limits!!".format(prob.status)))
    splxs = []
    for ui in u:
        splxs.append(simgen.rcp_simgen(n, asys, F, ui.value, xi, Lmax, u_max, u_min, phi))
    c_err = np.array([s.centering_err for s in splxs])
    return splxs[np.argmin(c_err)]

def prop_chain(n, sys, old_spx,  xi, Lmax, u_max, u_min, phi):
    """ Propagates the simplex chain"""
    n = 2
    s_o, xi = svc.chain_sup(s_in, del_s)
    # First one
    F0 = F
    u0 = rs(uMat[0, :], [n, 1])
    alpha0 = sys.A @ rs(F0[0,:], [n, 1])  + sys.B @ u0 + sys.a
    L0max = lmax.lambda_max(rs(F0[0, :], [n, 1]), alpha0, s_in, s_o)
    S0, r0 = simgen.rcp_simgen(F0, u0, sys, xi, L0max, spc.W)    # F, u0, sys, xi, Lmax, phi
    # Second one
    F1 = np.flip(F, 0)
    u1 = rs(uMat[1, :], [n, 1])
    alpha1 = sys.A @ rs(F1[0,:], [n, 1])  + sys.B @ u1 + sys.a
    L1max = lmax.lambda_max(rs(F1[0, :], [n, 1]), alpha1, s_in, s_o)
    S1, r1 = simgen.rcp_simgen(F1, u1, sys, xi, L1max, spc.W)
    if S0.qlty <= S1.qlty:
        Simplex = S0
    else:
        Simplex = S1
    return Simplex

def term_chain():
    """Terminate the chain of simplices by creating simplx with equilibrium inside"""



if __name__=="__main__":
    import space as spc
    import support_vecs as svc
    import system as ss

    F = spc.I
    Lmax = 3
    u_max = 2*np.ones([2, 1])
    u_min = -2*np.ones([2, 1])
    xi = np.matrix([[0], [-1]])
    Sim = init_chain(2, ss.lsys, F, xi, Lmax, u_max, u_min, spc.W)
