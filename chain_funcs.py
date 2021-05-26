from matplotlib.pyplot import axis
import numpy as np
from numpy import reshape as rs
import cvxpy as cvx
import normals as nr
import rcp_simgen as simgen
import space as spc


def init_chain(n, asys, F, s_in, del_s, u_max, u_min, phi, ptope_list):
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
    # Possible facets list
    F_list = []
    F_aug = np.append(F, F, axis=0)
    for i in range(n):
        F_list.append(F_aug[i:i+n, :])
    spx_list= []
    for ui, Fi in zip(u, F_list):
        spx_list.append(simgen.rcp_simgen(n, asys, Fi, ui.value, s_in, del_s, u_max, u_min, phi, ptope_list))
    c_err = np.array([s.centering_err for s in spx_list])
    return spx_list[np.argmin(c_err)]

def prop_chain(n, asys, old_spx, del_s, u_max, u_min, phi, ptope_list):
    """ Propagates the simplex chain"""
    F = old_spx.vMat[1:, :]
    uMat_F = old_spx.uMat[1:, :]   # Corresponding inputs
    _, m = np.shape(asys.B)
    # Possible restricted vertices and corresping exit facets list
    F_list = []
    u0_list = []
    F_aug = np.append(F, F, axis=0)
    for i in range(n):
        F_list.append(F_aug[i:i+n, :])
        u0_list.append(rs(uMat_F[i, :], [m,1]))
    spx_list = []
    for ui, Fi in zip(u0_list, F_list):
        spx_list.append(simgen.rcp_simgen(n, asys, Fi, ui, old_spx.so, del_s, u_max, u_min, phi, ptope_list))
    c_err = np.array([s.centering_err for s in spx_list])
    return spx_list[np.argmin(c_err)]



def term_chain(n, asys, old_spx,  xi, del_max, u_max, u_min, phi, ptope_list):
    """Terminate the chain of simplices by creating simplx with equilibrium inside"""



if __name__=="__main__":
    import space as spc
    import support_vecs as svc
    import system as ss

    F = spc.I
    s_in = np.matrix([[1], [-1]])
    del_s = 1
    u_max = 2*np.ones([2, 1])
    u_min = -2*np.ones([2, 1])
    xi = np.matrix([[0], [-1]])
    Sim = init_chain(2, ss.lsys, F, s_in, del_s, u_max, u_min, spc.W, spc.ptope_list)
    Sim2 = prop_chain(2, ss.lsys, Sim, del_s, u_max, u_min, spc.W, spc.ptope_list)
