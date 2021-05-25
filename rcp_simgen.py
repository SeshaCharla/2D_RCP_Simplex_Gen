from cvxpy.atoms import axis_atom
from pypoman import duality
import rcpSimplex as rsp
import numpy as np
from numpy import reshape as rs
import cvxpy as cvx
import pypoman as pp
import normals as nr



def rcp_simgen(n, asys, F, u0,  xi, Lmax, u_max, u_min, phi):
    """Returns an RCP simplex with the proper control inputs (column vector) and velocity vectors"""
    eps = 1e-6
    m =  np.shape(asys.B)[1]
    alpha0 = asys.A @ rs(F[0, :], [n, 1]) + asys.B @ u0 + asys.a
    # Sanity Checks
    if n != np.shape(F)[0] or n!=np.shape(F)[1]:
        raise(ValueError("Dimensions don't match!"))
    if m!=np.shape(u0)[0]:
        raise(ValueError("Dimensions of u don't match!"))

    # Finding the outward normals
    v_n = F[0, :] + Lmax*rs(alpha0, [1, n])
    vMat_ = np.append(F, v_n, axis=0)
    dummy_sim = rsp.Simplex(n, vMat_)
    h = dummy_sim.h

    # Optimization problem
    u = [cvx.Variable((m, 1)) for i in range(1, n+1)]
    l_gen = cvx.Variable()
    constraints = [l_gen <= Lmax, l_gen >= -Lmax]
    obj = 0
    for i in range(1, n):
        obj += xi.T @ (asys.A @ rs(F[i, :], [n, 1]) + asys.B @ u[i-1] + asys.a)
        # Flow constraints
        constraints += [xi.T @ (asys.A @ rs(F[i, :], [n, 1]) + asys.B @ u[i-1] + asys.a) >= eps]
        # Invariance Constraints
        I = list(np.arange(1, n+1))    # Index Set
        _ = I.pop(i-1)                      # Pop the index opposite to current face
        for j in I:
            hj = rs(h[j, :], [n, 1])
            constraints += [hj.T @ (asys.A @ rs(F[i, :], [n, 1]) + asys.B @ u[i-1] + asys.a) <= -eps]
        # input constraints
        constraints += [u[i-1] <= u_max, u[i-1]>= u_min]
    # For the new point
    i = n
    obj += xi.T @ (asys.A @ (rs(F[0, :], [n, 1])  + l_gen*alpha0)+ asys.B @ u[i-1] + asys.a)
    # Flow constraints
    constraints += [xi.T @ (asys.A @ (rs(F[0, :], [n, 1])  + l_gen*alpha0) + asys.B @ u[i-1] + asys.a) >= eps]
    # Invariance Constraints
    I = list(np.arange(1, n+1))    # Index Set
    _ = I.pop(i-1)                      # Pop the index opposite to current face
    for j in I:
        hj = rs(h[j, :], [n, 1])
        constraints += [hj.T @ (asys.A @ (rs(F[0, :], [n, 1])  + l_gen*alpha0) + asys.B @ u[i-1] + asys.a) <= -eps]
    # input constraints
    constraints += [u[i-1] <= u_max, u[i-1]>= u_min]
    # The problem
    prob = cvx.Problem(cvx.Maximize(obj), constraints=constraints)
    if not prob.is_dcp():
        raise(ValueError("The problem doesn't follow DCP rules!!"))
    prob.solve()
    if prob.status in ["infeasible", "unbounded"]:
        raise(ValueError("The optimization problem is {}.\nCheck control input Limits!!".format(prob.status)))
    #u Matrix
    uMat = np.zeros([n+1, m])
    uMat[0, :] = rs(u0, [1, m])
    for i in range(1, n+1):
        uMat[i, :] = rs(u[i-1].value, [1, m])

    # v Matrix
    v_n = F[0, :] + l_gen.value*rs(alpha0, [1, n])
    vMat = np.append(F, v_n, axis=0)

    S = rsp.rcpSimplex(n, asys, vMat, uMat, phi, xi, u_max, u_min)  # (n, asys, vMat, uMat, phi, xi_gen, u_max, u_min)
    return S


if __name__=="__main__":
    import system as sys
    import pypoman as p
    import matplotlib.pyplot as plt
    import space as spc
    ##############################################################################################
    F = spc.I
    u0 = np.matrix([[1], [0]])
    Lmax = 3
    u_max = 2*np.ones([2, 1])
    u_min = -2*np.ones([2, 1])
    xi = np.matrix([[0], [-1]])
    Sim = rcp_simgen(2, sys.lsys, F, u0, xi, Lmax, u_max, u_min, spc.W)
