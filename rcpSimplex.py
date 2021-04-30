import numpy as np
import pypoman as pp
from numpy import reshape as rs
import space as spc


class rcpSimplex2():
    """RCP simplex Class for 2D"""
    def __init__(self, vMat, uMat, lsys):
        # Given
        self.n = (np.shape(vMat))[1]
        self.m = (np.shape(uMat))[1]
        self.sys = lsys
        self.vMat = vMat
        self.uMat = uMat

        # Vertex Flow Vectors
        self.alphaMat = np.zeros([self.n+1, self.n])
        for i in range(self.n+1):
            self.alphaMat[i,:] = rs((self.sys.A @ rs((self.vMat[i,:]), [2,1]) + self.sys.B @ rs((self.uMat[i,:]), [2,1]) + self.sys.a), [1, 2])

        # Getting the affine feedback matrices
        augV = np.append(self.vMat, np.ones([self.n+1, 1]), axis=1)
        kg = [np.linalg.solve(augV, self.uMat[:, i]) for i in range(self.m)]
        self.K = np.zeros([self.m, self.n])
        self.g = np.zeros([self.m, 1])
        for i in range(self.m):
            self.K[i] = rs(kg[i][0:self.n], [1, self.n])
            self.g[i] = kg[i][-1]

        # Half Space Represintation
        self.v_list = [rs(vRow, [1, self.n]) for vRow in vMat]
        self.A, self.b = pp.duality.compute_polytope_halfspaces(self.v_list)

    def in_simplex(self, x):
        """x is np array"""
        y = rs(x, [self.n, 1])
        z = (self.A @ y).T - self.b
        if z.all() <= 0:
            return True
        else:
            return False

    def get_u(self, x):
        """ x in an array """
        y = rs(x, [self.n, 1])
        u = self.K @ y + self.g
        return u


    def set_xi(self, xi):
        """Records the flow vector"""
        self.xi = xi


if __name__=="__main__":
    import system
    vMat = np.matrix([[0, 0], [1, 0], [0, 1]])
    uMat = np.matrix([[1, 1], [2, 1], [3, 1]])
    rsp = rcpSimplex2(vMat, uMat, system.lsys)
    print(rsp.in_simplex([0.5, 0.5]))
    print(rsp.get_u([0.5,0.5]))
