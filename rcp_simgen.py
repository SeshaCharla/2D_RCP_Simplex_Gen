from rcpSimplex import rcpSimplex2
import numpy as np
import cvxpy as cvx
import pypoman as pp
import normals as nr



def rcp_simgen(F, u0, sys, xi, Lmax):
    """Returns an RCP simplex with the proper control inputs (column vector) and velocity vectors"""
    v0 = F[0, :].T
    v1 = F[1, :].T
    alpha = sys.A @ v0 + sys.B @ u0 + sys.a
    alpha0 = alpha/np.linalg.norm(alpha)
    n, m =  np.shape(sys.B)

    # Finding the required normals
    h2 = nr.normal_2(v0-v1)
    if (h2.T @ alpha0) > 0:
        h2 = -h2
    h1 = nr.normal_2(alpha0)
    if (h1.T @ (v1-v0)) > 0:
        h1 = -h1

    # Optimization Problem
    u1 = cvx.Variable([m, 1])
    u2 = cvx.Variable([m, 1])
    L = cvx.Variable(1)

    constraints = [L>0, L<=Lmax]
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
    umax = 6
    u_max = np.matrix([[6], [6]])
    M = np.kron( np . matrix ([[1] ,[ -1]]) , np . eye (m))
    p = np.ones([m, 1]) * umax
    constraints += [M @ u2 <= p, M @ u3 <= p]

    # Cost function:
    v1max = (Av1Flo @ u_max + bv1Flo)[0,0]
    v2max = (AlFlo * Lmax + Au2Flo @ u_max + bv2Flo)[0,0]
    obj = -(L/Lmax)**2 + (1 - (AlFlo * L + Au2Flo @ u2 + bv2Flo)/v2max) + (1- (Av1Flo @ u1 + bv1Flo)/v1max)

    # CVX problem
    prob = cvx.Problem(cvx.Minimize(obj), constraints)
    result = prob.solve()

    # u matrix
    uMat = np.zeros([n+1, m])
    uMat[0, :] = np.reshape(u0, [1, m])
    uMat[1, :] = np.reshape(u1.value, [1, m])
    uMat[2, :] = np.reshape(u2.value, [1, m])

    # v Matrix
    v2 = v0 + L.value*alpha0
    vMat = np.zeros([n+1, n])
    vMat[0, :] = np.reshape(v0, [1, n])
    vMat[1, :] = np.reshape(v1, [1, n])
    vMat[2, :] = np.reshape(v2, [1, n])

    S = rcpSimplex2(vMat, uMat, sys)
    return S, result


if __name__=="__main__":
    import system as sys
    F = np.matrix([[0, 0],
                   [0, 1]])
    u0 = np.matrix([[1.2], [1.1]])
    Lmax = 1
    xi = np.matrix([[1/2**0.5], [1/2**0.5]])
    Sim, opt = rcp_simgen(F, u0, sys.lsys, xi, Lmax)
