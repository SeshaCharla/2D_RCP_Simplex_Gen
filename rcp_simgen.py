from pypoman import duality
from rcpSimplex import rcpSimplex2
import numpy as np
import cvxpy as cvx
import pypoman as pp
import normals as nr



def rcp_simgen(F, u0, sys, xi, Lmax):
    """Returns an RCP simplex with the proper control inputs (column vector) and velocity vectors"""
    v0 = np.reshape(F[0, :], [2, 1])
    v1 = np.reshape(F[1, :], [2, 1])
    alpha = sys.A @ v0 + sys.B @ u0 + sys.a
    print(alpha)
    alpha0 = alpha/np.linalg.norm(alpha)
    n, m =  np.shape(sys.B)

    # Finding the required normals
    h2 = nr.normal_2(v0-v1)
    if (h2.T @ alpha0)[0,0] > 0:
        h2 = -h2
    h1 = nr.normal_2(alpha0)
    if (h1.T @ (v1-v0))[0,0] > 0:
        h1 = -h1

    # Optimization Problem
    u1 = cvx.Variable([m, 1])
    u2 = cvx.Variable([m, 1])
    L = cvx.Variable(1)

    constraints = [L>=0, L<=Lmax]
    # Invariance Constraints on v1
    Av1Inv = (h2.T @ sys.B)
    bv1Inv = (h2.T @ sys.A @ v1) + (h2.T @ sys.a)
    constraints += [Av1Inv @ u1 + bv1Inv <=0]
    # Flow condition on v1
    Av1Flo = xi.T @ sys.B
    bv1Flo = xi.T @ sys.A @ v1 + xi.T @ sys.a
    constraints += [Av1Flo @ u1 + bv1Flo >= 0]
    # Invariance Constraints on v2
    AlInv = (h1.T @ sys.A @ alpha0)
    Au2Inv = h1.T @ sys.B
    bv2Inv = h1.T @ sys.A @ v0 + h1.T @ sys.a
    constraints += [AlInv * L + Au2Inv @ u2 + bv2Inv <= 0]
    # Flow Constraints on v2
    AlFlo = xi.T @ sys.A @ alpha0
    Au2Flo = xi.T @ sys.B
    bv2Flo = xi.T @ sys.A @ v0 + xi.T @ sys.a
    constraints += [AlFlo * L + Au2Flo @ u2 + bv2Flo >= 0]
    # Constraints on u
    # umax = 6
    # u_max = np.matrix([[6], [6]])
    # M = np.kron( np . matrix ([[1] ,[ -1]]) , np . eye (m))
    # p = np.ones([m, 1]) * umax
    # #constraints += [M @ u2 <= p, M @ u3 <= p]

    # Objective function calculations:
    xi_n = nr.normal_2(xi)
    # Max flow condition on v1
    Av1Mflo = xi_n.T @ sys.B
    bv1Mflo = xi_n.T @ sys.A @ v1 + xi_n.T @ sys.a
    # Max flow condition on v2
    AlMflo = xi_n.T @ sys.A @ alpha0
    Au2Mflo = xi_n.T @ sys.B
    bv2Mflo = xi_n.T @ sys.A @ v0 + xi_n.T @ sys.a
    # Objective Function
    obj =  cvx.norm(Lmax-L) + cvx.norm(Av1Mflo @ u1 + bv1Mflo) + cvx.norm(AlMflo * L + Au2Mflo @ u2 + bv2Mflo)

    # CVX problem
    prob = cvx.Problem(cvx.Minimize(obj), constraints)
    result = prob.solve()

    # u matrix
    uMat = np.zeros([n+1, m])
    uMat[0, :] = np.reshape(u0, [1, m])
    uMat[1, :] = np.reshape(u1.value, [1, m])
    uMat[2, :] = np.reshape(u2.value, [1, m])

    # v Matrix
    v2 = v0 + L.value[0]*alpha0
    vMat = np.zeros([n+1, n])
    vMat[0, :] = np.reshape(v0, [1, n])
    vMat[1, :] = np.reshape(v1, [1, n])
    vMat[2, :] = np.reshape(v2, [1, n])

    S = rcpSimplex2(vMat, uMat, sys)
    return S, result


if __name__=="__main__":
    import system as sys
    import pypoman as p
    import matplotlib.pyplot as plt
    ##############################################################################################
    F = np.matrix([[0, 0],
                   [0, 1]])
    u0 = np.matrix([[1.2], [0]])
    Lmax = 1
    xi = np.matrix([[1], [0]])
    Sim, opt = rcp_simgen(F, u0, sys.lsys, xi, Lmax)
    xi1 = np.matrix([[1], [0.5]])
    F1 = np.matrix([Sim.vMat[2, :], Sim.vMat[1,:]])
    u1 = np.reshape(Sim.uMat[2,:], [2, 1])
    Sim1, opt1 = rcp_simgen(F1, u1, sys.lsys, xi1, Lmax)
    xi2 = np.matrix([[1], [-0.5]])
    F2 = np.matrix([Sim1.vMat[1, :], Sim1.vMat[2,:]])
    u2 = np.reshape(Sim1.uMat[1,:], [2, 1])
    Sim2, opt2 = rcp_simgen(F2, u2, sys.lsys, xi2, Lmax)
    plt.figure()
    pp.plot_polygon(pp.duality.compute_polytope_vertices(Sim.A, Sim.b))
    for i in range(3):
        plt.plot([Sim.vMat[i, 0], 0.2*(Sim.alphaMat[i, 0]/np.linalg.norm(Sim.alphaMat[i,:]))+Sim.vMat[i, 0]], [Sim.vMat[i, 1], 0.2*(Sim.alphaMat[i, 1]/np.linalg.norm(Sim.alphaMat[i,:]))+Sim.vMat[i, 1]])
        #plt.plot([Sim.vMat[i, 0], 0.5*(xi[0, 0])+Sim.vMat[i, 0]], [Sim.vMat[i, 1], 0.5*(xi[1, 0])+Sim.vMat[i, 1]], "--k")
    pp.plot_polygon(pp.duality.compute_polytope_vertices(Sim1.A, Sim1.b))
    for i in range(3):
        plt.plot([Sim1.vMat[i, 0], 0.2*(Sim1.alphaMat[i, 0]/np.linalg.norm(Sim1.alphaMat[i,:]))+Sim1.vMat[i, 0]], [Sim1.vMat[i, 1], 0.2*(Sim1.alphaMat[i, 1]/np.linalg.norm(Sim1.alphaMat[i,:]))+Sim1.vMat[i, 1]])
        plt.plot([Sim1.vMat[i, 0], 0.5*(xi1[0, 0])+Sim1.vMat[i, 0]], [Sim1.vMat[i, 1], 0.5*(xi1[1, 0])+Sim1.vMat[i, 1]], "--k")
    pp.plot_polygon(pp.duality.compute_polytope_vertices(Sim2.A, Sim2.b))
    for i in range(3):
        plt.plot([Sim2.vMat[i, 0], 0.2*(Sim2.alphaMat[i, 0]/np.linalg.norm(Sim2.alphaMat[i,:]))+Sim2.vMat[i, 0]], [Sim2.vMat[i, 1], 0.2*(Sim2.alphaMat[i, 1]/np.linalg.norm(Sim2.alphaMat[i,:]))+Sim2.vMat[i, 1]])
        plt.plot([Sim2.vMat[i, 0], 0.5*(xi2[0, 0])+Sim2.vMat[i, 0]], [Sim2.vMat[i, 1], 0.5*(xi2[1, 0])+Sim2.vMat[i, 1]], "--r")
    ###############################################################################################
    # from space import W, I
    # F = I
    # u0 = np.matrix([[1.2], [0]])
    # Lmax = 1
    # sims = []
    # xis = []
    # for i in range(4):
    #     xi = np.reshape(W[i+1,:]- W[i,:], [2, 1])
    #     xi = xi/np.linalg.norm(xi)
    #     S, opt = rcp_simgen(F, u0, sys.lsys, xi, Lmax)
    #     sims.append(S)
    #     F = np.matrix([S.vMat[1, :], S.vMat[2,:]])
    #     u0 = np.reshape(S.uMat[1,:], [2, 1])
    #     xis.append(xi)
    # plt.figure()
    # for Sim in sims:
    #     pp.plot_polygon(pp.duality.compute_polytope_vertices(Sim.A, Sim.b))
    #     for i in range(3):
    #         plt.plot([Sim.vMat[i, 0], 0.2*(Sim.alphaMat[i, 0]/np.linalg.norm(Sim.alphaMat[i,:]))+Sim.vMat[i, 0]], [Sim.vMat[i, 1], 0.2*(Sim.alphaMat[i, 1]/np.linalg.norm(Sim.alphaMat[i,:]))+Sim.vMat[i, 1]])
    #         plt.plot([Sim.vMat[i, 0], 0.5*(xis[i][0, 0])+Sim.vMat[i, 0]], [Sim.vMat[i, 1], 0.5*(xis[i][1, 0])+Sim.vMat[i, 1]], "--k")
    plt.show()
