import numpy as np
from numpy import reshape as rs
import cvxpy as cvx
from numpy.lib.shape_base import _apply_along_axis_dispatcher
import normals as nr
import rcpSimgen as simgen
import rcpSimplex as rspx
import space as spc
import support_vecs as svc


def init_chain(n, asys, F, s_in, u_max, u_min, phi, ptope_list):
    """ Create initial simplex"""
    eps = 1e-6
    *_, m =  np.shape(asys.B)

    # s_o and xi
    s_o, xi = svc.chain_flow(n, s_in, phi)

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
    ld_list = []
    for i in range(n):
        alpha_r = asys.A @ rs(F_list[i][0, :], [n, 1]) + asys.B @ u[i].value + asys.a
        spx, ld = simgen.rcp_simgen(n, asys, F_list[i], u[i].value, alpha_r, s_in, u_max, u_min, phi, ptope_list)
        spx_list.append(spx)
        ld_list.append(ld)
    return spx_list[np.argmax(ld_list)]

def prop_chain(n, asys, old_spx, u_max, u_min, phi, ptope_list):
    """ Propagates the simplex chain"""
    F = old_spx.vMat[1:, :]
    uMat_F = old_spx.uMat[1:, :]   # Corresponding inputs
    alphaMat_F = old_spx.alphaMat[1:, :]    # Corresponding closed vf
    _, m = np.shape(asys.B)
    # Possible restricted vertices and corresping exit facets list
    F_list = []
    u0_list = []
    alphar_list = []
    F_aug = np.append(F, F, axis=0)
    for i in range(n):
        F_list.append(F_aug[i:i+n, :])
        u0_list.append(rs(uMat_F[i, :], [m,1]))
        alphar_list.append(rs(alphaMat_F[i, :], [n, 1]))
    spx_list= []
    ld_list = []
    for i in range(n):
        alpha_r = asys.A @ rs(F_list[i][0, :], [n, 1]) + asys.B @ u0_list[i] + asys.a
        spx, ld = simgen.rcp_simgen(n, asys, F_list[i], u0_list[i], alphar_list[i], old_spx.so, u_max, u_min, phi, ptope_list)
        spx_list.append(spx)
        ld_list.append(ld)
    return spx_list[np.argmax(ld_list)]


def term_chain(n, asys, old_spx, u_max, u_min, phi):
    """Terminate the chain of simplices by creating simplx with equilibrium inside"""
    F = old_spx.vMat[1:, :]
    return rspx.terminalSimplex(n, asys, F, phi, u_max, u_min)
