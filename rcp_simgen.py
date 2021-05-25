from cvxpy.atoms import axis_atom
from pypoman import duality
import rcpSimplex as rsp
import numpy as np
from numpy import reshape as rs
import cvxpy as cvx
import pypoman as pp
import normals as nr



def rcp_simgen(n, F, u0, alpha0, asys, xi, Lmax, ulims, phi):
    """Returns an RCP simplex with the proper control inputs (column vector) and velocity vectors"""
    eps = 1e-3
    m =  np.shape(sys.B)[1]
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

    # Matrix for input limits
    M = np.kron(np.matrix([[1] ,[ -1]]) , np.eye(m))

    # Optimization problem
    u = [cvx.Variable([m, 1]) for i in range(1, n+1)]
    L = cvx.Variable(1)
    constraints = []
    obj = 0
    constraints = [L>=0, L<=Lmax]
    for i in range(1, n):
            obj += xi.T @ (asys.A @ rs(F[i, :], [n, 1]) + asys.B @ u[i-1] + asys.a)
            # Flow constraints
            constraints += [xi.T @ (asys.A @ rs(F[i, :], [n, 1]) + asys.B @ u[i-1] + asys.a) >= eps]
            # Invariance Constraints
            I = list(np.arange(1, n+1))    # Index Set
            _ = I.pop(i)                      # Index Set
            for j in I:
                hj = rs(h[j, :], [n, 1])
                constraints += [hj.T @ (asys.A @ rs(F[i, :], [n, 1]) + asys.B @ u[i-1] + asys.a) <= -eps]
            # input constraints
            constraints += [M @ u[i-1] <= u_lims]
    # For nth new vertex
    obj += xi.T @ (asys.A @ (rs(F[0, :], [n, 1]) + L * alpha0) + asys.B @ u[n] + asys.a)
    # Flow constraints
    constraints += [xi.T @ (asys.A @ (rs(F[0, :], [n, 1]) + L * alpha0) + asys.B @ u[n] + asys.a) >= eps]
    # Invariance Constraints
    I = list(np.arange(1, n+1))    # Index Set
    _ = I.pop(n)                      # Index Set
    for j in I:
        hj = rs(h[j, :], [n, 1])
        constraints += [hj.T @ (asys.A @ (rs(F[0, :], [n, 1]) + L * alpha0) + asys.B @ u[i-1] + asys.a) <= -eps]
    # input constraints
    constraints += [M @ u[n] <= u_lims]
    # Setting up the problem
    prob = cvx.Problem(cvx.Maximize(obj), constraints=constraints)
    if not prob.is_dcp():
        raise(ValueError("The problem doesn't follow DCP rules!!"))
    prob.solve()
    # u matrix
    uMat = np.zeros([n+1, m])
    uMat[0, :] = np.reshape(u0, [1, m])
    for i in range(1, n+1):
        uMat[i, :] = rs(u[i-1].value, [1, m])

    # v Matrix
    v_n = F[0, :] + L.value*rs(alpha0, [1, n])
    vMat_ = np.append(F, v_n, axis=0)

    S = rsp.rcpSimplex(n, asys, vMat, uMat, phi, xi, u_lims)  # (n, asys, vMat, uMat, phi, xi_gen, u_lims)
    return S


if __name__=="__main__":
    import system as sys
    import pypoman as p
    import matplotlib.pyplot as plt
    import space as spc
    ##############################################################################################
    F = np.matrix([[0, 0],
                   [0, 1]])
    u0 = np.matrix([[1.2], [0]])
    alpha0 = sys.lsys.A @ rs(F[0, :],[2, 1]) + sys.lsys.B @ u0 + sys.lsys.a
    Lmax = 1
    umax = 12
    u_lims = np.ones([2*3, 1]) * umax
    xi = np.matrix([[1], [0]])
    Sim = rcp_simgen(2, F, u0, alpha0, sys.lsys, xi, Lmax, u_lims, spc.W)
    # xi1 = np.matrix([[1], [0.5]])
    # F1 = np.matrix([Sim.vMat[1, :], Sim.vMat[2,:]])
    # u1 = np.reshape(Sim.uMat[1,:], [2, 1])
    # Sim1 = rcp_simgen(F1, u1, sys.lsys, xi1, Lmax, spc.W)
    # xi2 = np.matrix([[1], [-0.5]])
    # F2 = np.matrix([Sim1.vMat[1, :], Sim1.vMat[2,:]])
    # u2 = np.reshape(Sim1.uMat[1,:], [2, 1])
    # Sim2 = rcp_simgen(F2, u2, sys.lsys, xi2, Lmax, spc.W)
    # plt.figure()
    # pp.plot_polygon(pp.duality.compute_polytope_vertices(Sim.A, Sim.b))
    # for i in range(3):
    #     plt.plot([Sim.vMat[i, 0], 0.2*(Sim.alphaMat[i, 0]/np.linalg.norm(Sim.alphaMat[i,:]))+Sim.vMat[i, 0]], [Sim.vMat[i, 1], 0.2*(Sim.alphaMat[i, 1]/np.linalg.norm(Sim.alphaMat[i,:]))+Sim.vMat[i, 1]])
    #     plt.plot([Sim.vMat[i, 0], 0.5*(xi[0, 0])+Sim.vMat[i, 0]], [Sim.vMat[i, 1], 0.5*(xi[1, 0])+Sim.vMat[i, 1]], "--k")
    # pp.plot_polygon(pp.duality.compute_polytope_vertices(Sim1.A, Sim1.b))
    # for i in range(3):
    #     plt.plot([Sim1.vMat[i, 0], 0.2*(Sim1.alphaMat[i, 0]/np.linalg.norm(Sim1.alphaMat[i,:]))+Sim1.vMat[i, 0]], [Sim1.vMat[i, 1], 0.2*(Sim1.alphaMat[i, 1]/np.linalg.norm(Sim1.alphaMat[i,:]))+Sim1.vMat[i, 1]])
    #     plt.plot([Sim1.vMat[i, 0], 0.5*(xi1[0, 0])+Sim1.vMat[i, 0]], [Sim1.vMat[i, 1], 0.5*(xi1[1, 0])+Sim1.vMat[i, 1]], "--k")
    # pp.plot_polygon(pp.duality.compute_polytope_vertices(Sim2.A, Sim2.b))
    # for i in range(3):
    #     plt.plot([Sim2.vMat[i, 0], 0.2*(Sim2.alphaMat[i, 0]/np.linalg.norm(Sim2.alphaMat[i,:]))+Sim2.vMat[i, 0]], [Sim2.vMat[i, 1], 0.2*(Sim2.alphaMat[i, 1]/np.linalg.norm(Sim2.alphaMat[i,:]))+Sim2.vMat[i, 1]])
    #     plt.plot([Sim2.vMat[i, 0], 0.5*(xi2[0, 0])+Sim2.vMat[i, 0]], [Sim2.vMat[i, 1], 0.5*(xi2[1, 0])+Sim2.vMat[i, 1]], "--r")
    # ###############################################################################################
    # plt.show()
