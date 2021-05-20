from os import umask
import numpy as np
from numpy.linalg import LinAlgError
import pypoman as pp
from numpy import delete, reshape as rs
import space as spc
import cvxpy as cvx
import normals as nr


class rcpSimplex2():
    """RCP simplex Class for 2D"""
    def __init__(self, n, asys, vMat, uMat, phi):
        """n - Dimension; asys - affine linear system;
           vMat - Vertex Matrix; uMat - Control input matrix,
           phi - support way-point set"""
        # Given
        self.n = n
        self.m = np.shape(uMat)[1]    # Input size
        self.asys = asys
        self.vMat = vMat
        self.uMat = uMat
        self.phi = phi
        # Sanity Checks
        if (n+1) != np.shape(vMat)[1] or np.shape(vMat)[1] != np.shape(uMat)[1] or n != np.shape(vMat)[0]:
            raise(ValueError("The dimensions don't match!"))
        # Half Space Represintation
        self.A, self.b = pp.duality.compute_polytope_halfspaces(np.array(self.vMat))
        self.calc_exit_flow()
        self.optimize_inputs()
        self.calc_affine_feedback()
        self.calc_vertex_flows()

    def calc_exit_flow(self):
        """Calculate the exit facet intersection and the flow vector"""
        Fo = vMat[1:, :]    # Matrix containing the exit facet vertices
        p, *_ = np.shape(self.phi)
        Ld = np.empty([self.n+2, 1])
        for i in range(p-1):
            # Constructing A and b matrices
            M = np.append(Fo, np.matrix([[-1*self.phi[i,:]],[-1*self.phi[i+1, :]]]), axis=0)
            A = M.T
            b = np.zeros([np.shape(A)[0], 1])
            b[-1, 0] = 1
            b[-2, ] = 1
            try:
                Ld = np.linalg.solve(A, b)
            except LinAlgError:
                continue
            if all(Ld) >= 0:
                break
        if any(np.isnan(Ld)):
            raise(ValueError("The Facet seems not to intersect the support curve!"))
        else:
            self.seg = self.phi[i:i+2, :]
        delt = Ld[-2:, 0]
        self.so = (delt.T @ self.seg).T
        if (delt[2, 0] >= 0.2):
            xi_vec = (self.phi[i+1, :] - self.phi[i, :]).T
            self.xi = xi_vec / (np.linalg.norm(xi_vec))
        else:
            xi_vec = (self.phi[i+2, :] - self.phi[i+1, :]).T
            self.xi = xi_vec / (np.linalg.norm(xi_vec))

    def optimize_inputs(self):
        """Runs a new optimization problem to update inputs"""


    def calc_affine_feedback(self):
        """Getting the affine feedback matrices"""
        augV = np.append(self.vMat, np.ones([self.n+1, 1]), axis=1)
        kg = [np.linalg.solve(augV, self.uMat[:, i]) for i in range(self.m)]
        self.K = np.zeros([self.m, self.n])
        self.g = np.zeros([self.m, 1])
        for i in range(self.m):
            self.K[i] = rs(kg[i][0:self.n], [1, self.n])
            self.g[i] = kg[i][-1]

    def calc_vertex_flows(self):
        """ Vertex Flow Vectors """
        self.alphaMat = np.zeros([self.n+1, self.n])
        for i in range(self.n+1):
            self.alphaMat[i,:] = rs((self.sys.A @ rs((self.vMat[i,:]), [2,1]) + self.sys.B @ rs((self.uMat[i,:]), [2,1]) + self.sys.a), [1, 2])

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



if __name__=="__main__":
    import system
    vMat = np.matrix([[0, 0], [1, 0], [0, 1]])
    uMat = np.matrix([[1, 1], [2, 1], [3, 1]])
    rsp = rcpSimplex2(vMat, uMat, system.lsys, spc.W)
    print(rsp.in_simplex([0.5, 0.5]))
    print(rsp.get_u([0.5,0.5]))
