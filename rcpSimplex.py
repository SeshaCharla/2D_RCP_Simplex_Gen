import numpy as np
import pypoman as pp
from numpy import reshape as rs
import space as spc
import cvxpy as cvx
import normals as nr
from itertools import combinations


class Simplex():
    """Just Geometric aspects of the simplex"""
    def __init__(self, n, vMat):
        """n - Dimension; asys - affine linear system;
           vMat - Vertex Matrix; uMat - Control input matrix,
           phi - support way-point set"""
        # Given
        self.n = n
        self.vMat = vMat
        # Sanity Checks
        if (n+1) != np.shape(self.vMat)[0] or n != np.shape(self.vMat)[1] :
            raise(ValueError("The dimensions don't match!"))
        # # Half Space Represintation
        # self.A, self.b = pp.duality.compute_polytope_halfspaces(np.array(self.vMat))
        self.calc_ourward_normals()

    def calc_ourward_normals(self):
        """ Calculates the matrix of outward normals of all the facets"""
        self.F = []    # Facets
        self.h = np.zeros([self.n+1, self.n])
        for i in range(self.n+1):
            I = list(np.arange(0, self.n+1))
            j = I.pop(i)    # Facet index set
            fMat = np.zeros([self.n, self.n]) # Facet vertex Matrix
            for i in range(self.n):
                fMat[i, :] = self.vMat[I[i], :]
            self.F.append(fMat)
            vecMat = np.zeros([self.n-1, self.n])
            for i in range(self.n-1):
                vecMat[i, :] = fMat[i+1, :] - fMat[0, :]
            h_n = nr.normal(vecMat, self.n)
            edge = rs(self.vMat[j,:] - fMat[0,:], [self.n, 1])
            edge_n = edge/np.linalg.norm(edge)
            if (h_n.T @ edge_n) < 0 :
                h_n = -h_n
            self.h[i, :] = rs(h_n, [1, self.n])


class rcpSimplex():
    """RCP simplex Class for n-D"""
    def __init__(self, n, asys, vMat, uMat, phi, xi_gen, u_lims):
        """n - Dimension; asys - affine linear system;
           vMat - Vertex Matrix; uMat - Control input matrix,
           phi - support way-point set"""
        # Given
        self.n = n
        self.asys = asys
        self.m = np.shape(self.asys.B)[1]    # Input size
        self.vMat = vMat
        self.uMat = uMat
        self.phi = phi
        self.xi_gen = xi_gen
        self.u_lims = u_lims    # u_lims=[u1_max, u2_max, ..., -u1_min, -u2_min, ...]
        # Sanity Checks
        if (n+1) != np.shape(self.vMat)[0] or np.shape(self.vMat)[0] != np.shape(self.uMat)[0] or n != np.shape(self.vMat)[1] or \
            self.m != np.shape(self.uMat)[1]:
            raise(ValueError("The dimensions don't match!"))
        # Half Space Represintation
        self.A, self.b = pp.duality.compute_polytope_halfspaces(np.array(self.vMat))
        self.calc_vertex_flows()
        self.calc_exit_flow()
        #if (0 < 1- np.abs(self.xi_gen.T @ self.xi) < 1e-3) :
        self.optimize_inputs()
        self.calc_affine_feedback()
        self.calc_vertex_flows()
        self.set_quality()

    def calc_exit_flow(self):
        """Calculate the exit facet intersection and the flow vector"""
        Fo = self.vMat[1:, :]    # Matrix containing the exit facet vertices
        p, *_ = np.shape(self.phi)  # Total number of segments
        ld = np.nan * np.ones([self.n+2, 1])   # \lambda1, ..., -\delta1, -\delta2,...
        for i in range(p-1):
            # Constructing A and b matrices
            M_vert = np.append(Fo, -1*self.phi[i:i+2,:], axis=0)
            l_sum = np.append(np.ones([self.n, 1]), np.zeros([2, 1]), axis = 0)
            d_sum = np.append(np.zeros([self.n, 1]), np.ones([2, 1]), axis = 0)
            M = np.append(M_vert, np.append(l_sum, d_sum, axis = 1), axis=1)
            A = M.T
            b = np.zeros([np.shape(A)[0], 1])
            b[-1, 0] = 1
            b[-2, 0] = 1
            try:
                ld = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            if all(ld) >= 0:
                break
        if any(np.isnan(ld)):
            raise(ValueError("The Facet seems not to intersect the support curve!"))
        else:
            self.seg = self.phi[i:i+2, :]
        self.l_int = ld[0:-2, 0]
        d_int = ld[-2:, 0]
        self.so = (d_int @ self.seg).T
        if (d_int[0] <= 0.2) and i < p-2:       # When the point is close to the end point of segment
            xi_vec = (self.phi[i+2, :] - self.phi[i+1, :]).T
        else:
            xi_vec = (self.phi[i+1, :] - self.phi[i, :]).T
        self.xi = xi_vec / (np.linalg.norm(xi_vec))

    def optimize_inputs(self):
        """Runs a new optimization problem to update inputs"""
        eps = 1e-3
        M = np.kron(np.matrix([[1] ,[ -1]]) , np.eye(self.m))
        #vr = rs(self.vMat[0,:], [self.n, 1])
        #alpha_0 = rs(self.alphaMat[0, :], [self.n, 1])
        self.calc_ourward_normals()
        # Optimization problem
        u = [cvx.Variable((self.m, 1)) for i in range(1, self.n+1)]
        constraints = []
        obj = 0
        for i in range(1, self.n+1):
            obj += self.xi.T @ (self.asys.A @ rs(self.vMat[i, :], [self.n, 1]) + self.asys.B @ u[i-1] + self.asys.a)
            # Flow constraints
            constraints += [self.xi.T @ (self.asys.A @ rs(self.vMat[i, :], [self.n, 1]) + self.asys.B @ u[i-1] + self.asys.a) >= eps]
            # Invariance Constraints
            I = list(np.arange(1, self.n+1))    # Index Set
            _ = I.pop(i-1)                      # Pop the index opposite to current face
            for j in I:
                hj = rs(self.h[j, :], [self.n, 1])
                constraints += [hj.T @ (self.asys.A @ rs(self.vMat[i, :], [self.n, 1]) + self.asys.B @ u[i-1] + self.asys.a) <= -eps]
            # input constraints
            constraints += [M @ u[i-1] <= self.u_lims]
        prob = cvx.Problem(cvx.Maximize(obj), constraints=constraints)
        if not prob.is_dcp():
            raise(ValueError("The problem doesn't follow DCP rules!!"))
        prob.solve()
        print(prob.status)
        print(u[1].value)
        for i in range(1, self.n+1):
            self.uMat[i, :] = rs(u[i-1].value, [1, self.m])

    def calc_ourward_normals(self):
        """ Calculates the matrix of outward normals of all the facets"""
        self.F = []    # Facets
        self.h = np.zeros([self.n+1, self.n])
        for i in range(self.n+1):
            I = list(np.arange(0, self.n+1))
            j = I.pop(i)    # Facet index set
            fMat = np.zeros([self.n, self.n]) # Facet vertex Matrix
            for i in range(self.n):
                fMat[i, :] = self.vMat[I[i], :]
            self.F.append(fMat)
            vecMat = np.zeros([self.n-1, self.n])
            for i in range(self.n-1):
                vecMat[i, :] = fMat[i+1, :] - fMat[0, :]
            h_n = nr.normal(vecMat, self.n)
            edge = rs(self.vMat[j,:] - fMat[0,:], [self.n, 1])
            edge_n = edge/np.linalg.norm(edge)
            if (h_n.T @ edge_n) < 0 :
                h_n = -h_n
            self.h[i, :] = rs(h_n, [1, self.n])

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
            alpha_i = (self.asys.A @ rs((self.vMat[i,:]), [self.n,1]) + self.asys.B @ rs((self.uMat[i,:]), [self.n,1]) + self.asys.a)
            alpha_n = alpha_i/np.linalg.norm(alpha_i)
            self.alphaMat[i,:] = rs(alpha_n, [1, self.n])

    def in_simplex(self, x):
        """x is np array"""
        y = rs(x, [self.n, 1])
        z = (self.A @ y).T - self.b
        if np.all(z <= 0):
            return True
        else:
            return False

    def get_u(self, x):
        """ x in an array """
        y = rs(x, [self.n, 1])
        u = self.K @ y + self.g
        return u

    def set_quality(self):
        """Get the quality of simplex"""
        self.qlty = np.linalg.norm(self.lmbas- (1/self.n)*np.ones(self.n))


if __name__=="__main__":
    import system
    # Showing for 3D case
    A = np.eye(3)
    B = np.eye(3)
    a = np.zeros([3, 1])
    lsys = system.asys(A, B, a)
    vMat = np.matrix([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    uMat = np.matrix([[1, 1, 1], [2, 1, 1], [3, 1, 1], [1, 1, 1]])
    xi = np.matrix([[0], [1], [1]])
    umax = 12
    u_max = np.matrix([[6], [6], [6]])
    M = np.kron( np . matrix ([[1] ,[ -1]]) , np . eye (3))
    u_lims = np.ones([2*3, 1]) * umax
    W = np.matrix([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4,4,4], [5, 5, 5]])
    rsp = rcpSimplex(3, lsys, vMat, uMat, W, xi, u_lims)
    print(rsp.in_simplex([0.3, 0.3, 0.3]))
    print(rsp.get_u([0.3,0.3, 0.3]))
