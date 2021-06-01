import numpy as np
import pypoman as pp
from numpy import reshape as rs
import cvxpy as cvx
import normals as nr
import intersection as intsc
import support_vecs as svc



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

        self.calc_ourward_normals()

    def calc_ourward_normals(self):
        """ Calculates the matrix of outward normals of all the facets"""
        self.F = []    # Facets
        self.h = np.zeros([self.n+1, self.n])
        for i in range(self.n+1):
            I = list(np.arange(0, self.n+1))
            j = I.pop(i)    # Facet index set
            fMat = np.zeros([self.n, self.n]) # Facet vertex Matrix
            for k in range(self.n):
                fMat[k, :] = self.vMat[I[k], :]
            self.F.append(fMat)
            vecMat = np.zeros([self.n-1, self.n])
            for l in range(self.n-1):
                vecMat[l, :] = fMat[l+1, :] - fMat[0, :]
            h_n = nr.normal(vecMat, self.n)
            edge = rs(self.vMat[j,:] - fMat[0,:], [self.n, 1]) # drawing normal from the the facet point to the opposite point
            edge_n = edge/np.linalg.norm(edge)
            if (h_n.T @ edge_n) > 0 :               # Normal at the point should be opposite to the edge
                h_n = -h_n
            self.h[i, :] = rs(h_n, [1, self.n])


class rcpSimplex(Simplex):
    """RCP simplex Class for n-D"""
    def __init__(self, n, asys, vMat, uMat, phi, xi_gen, u_max, u_min):
        """n - Dimension; asys - affine linear system;
           vMat - Vertex Matrix; uMat - Control input matrix,
           phi - support way-point set"""
        # Given
        self.n = n
        self.asys = asys
        self.m = np.shape(self.asys.B)[1]    # Input size
        self.vMat = np.matrix(vMat)
        self.uMat = uMat
        self.phi = phi
        self.xi_gen = xi_gen
        self.u_max = u_max
        self.u_min = u_min
        # Vertices for plotting
        self.vertices = [self.vMat[i, :].A1 for i in range(n+1)]
        # Sanity Checks
        if (n+1) != np.shape(self.vMat)[0] or np.shape(self.vMat)[0] != np.shape(self.uMat)[0] or n != np.shape(self.vMat)[1] or \
            self.m != np.shape(self.uMat)[1]:
            raise(ValueError("The dimensions don't match!"))
        # Half Space Represintation
        self.A, self.b = pp.duality.compute_polytope_halfspaces(np.array(self.vMat))
        self.calc_vertex_flows()
        self.calc_exit_flow()
        self.optimize_inputs()
        self.calc_affine_feedback()
        self.calc_vertex_flows()
        self.calc_affine_feedback()
        self.calc_next_vr()
        self.calc_centering_err()

    def calc_exit_flow(self):
        """Calculate the exit facet intersection and the flow vector"""
        Fo = self.vMat[1:, :]    # Matrix containing the exit facet
        p, _ = np.shape(self.phi)
        int_lst = intsc.seg_facet(self.n, Fo, self.phi)
        if int_lst:
            int_tu = int_lst[0]
            k = int_tu[1]
            self.seg = self.phi[k:k+2, :]
            self.l_int = rs(int_tu[0], [1, self.n])
            d_int = rs(int_tu[2], [1, 2])
            self.so = (d_int @ self.seg).T   # intersection of exit facet with the support curvex
            sn, self.xi = svc.chain_flow(self.n, self.so, self.phi)
        else:
            raise(ValueError("The facet is not intersecting the curve!!"))

    # def calc_ourward_normals(self):
    #     """ Calculates the matrix of outward normals of all the facets"""
    #     self.F = []    # Facets
    #     self.h = np.zeros([self.n+1, self.n])
    #     for i in range(self.n+1):
    #         I = list(np.arange(0, self.n+1))
    #         j = I.pop(i)    # Facet index set
    #         fMat = np.zeros([self.n, self.n]) # Facet vertex Matrix
    #         for k in range(self.n):
    #             fMat[k, :] = self.vMat[I[k], :]
    #         self.F.append(fMat)
    #         vecMat = np.zeros([self.n-1, self.n])
    #         for l in range(self.n-1):
    #             vecMat[l, :] = fMat[l+1, :] - fMat[0, :]
    #         h_n = nr.normal(vecMat, self.n)
    #         edge = rs(self.vMat[j,:] - fMat[0,:], [self.n, 1]) # drawing normal from the the facet point to the opposite point
    #         edge_n = edge/np.linalg.norm(edge)
    #         if (h_n.T @ edge_n) > 0 :      # Normal at the point should be opposite to the edge
    #             h_n = -h_n
    #         self.h[i, :] = rs(h_n, [1, self.n])

    def optimize_inputs(self):
        """Runs a new optimization problem to update inputs"""
        eps = 1e-6
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
            constraints += [u[i-1] <= self.u_max, u[i-1]>= self.u_min]
        prob = cvx.Problem(cvx.Maximize(obj), constraints=constraints)
        if not prob.is_dcp():
            raise(ValueError("The problem doesn't follow DCP rules!!"))
        prob.solve()
        if prob.status in ["infeasible", "unbounded"]:
            raise(ValueError("The optimization problem is {}.\nCheck control input Limits!!".format(prob.status)))
        for i in range(1, self.n+1):
            self.uMat[i, :] = rs(u[i-1].value, [1, self.m])

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

    def calc_next_vr(self):
        """Finds the next restricted vertex among the vectors and puts it in F0[0,:]"""
        Fo = self.vMat[1:, :]
        F_aug = np.append(Fo, Fo, axis=0)
        alpha_o = self.alphaMat[1:, :]
        align_vecs = alpha_o @ self.xi
        # j = np.argmax(align_vecs)          # alpha_0 is removed
        # self.F_next = F_aug[j:j+self.n, :] #v0 is removed
        # self.u0_next = rs(self.uMat[j+1, :], [self.m, 1])      # u0 is not removed
        # self.alpha0_next = rs(self.alphaMat[j+1, :], [self.n, 1])
        self.F_next_list = []
        self.u_next_list = []
        self.alpha_next_list = []
        vec_max = np.max(align_vecs)
        for i in range(self.n):
            if abs((align_vecs[i, 0]-vec_max)/vec_max) <=0.1:
                self.F_next_list.append(F_aug[i:i+self.n, :])     #v0 is removed
                self.u_next_list.append(rs(self.uMat[i+1, :], [self.m, 1]))        # u0 is not removed
                self.alpha_next_list.append(rs(self.alphaMat[i+1,:], [self.n, 1])) # alpha_0 is not removed
        self.next_list = zip(self.F_next_list, self.u_next_list, self.alpha_next_list)


    def in_simplex(self, x):
        """x is np array"""
        y = rs(x, [self.n, 1])
        z = (self.A @ y).T - self.b
        if np.all(z <= 1e-6):
            return True
        else:
            return False

    def get_u(self, x):
        """ x in an array """
        y = rs(x, [self.n, 1])
        u = self.K @ y + self.g
        return u

    def calc_centering_err(self):
        """Centering error"""
        self.c_err = np.linalg.norm(self.l_int - (1/self.n) * np.ones(np.shape(self.l_int)))


class terminalSimplex(rcpSimplex):
    """The terminal simplex class"""
    def __init__(self, n, asys, F, phi, u_max, u_min):
        self.asys = asys
        self.phi = phi
        self.vMat = np.append(F, self.phi[-1,:], axis=0)
        # Vertices for plotting
        self.vertices = [self.vMat[i, :].A1 for i in range(n+1)]
        self.n = n
        self.m = np.shape(asys.B)[1]
        self.u_min = u_min
        self.u_max = u_max
        self.uMat = np.zeros([self.n+1, self.m])
        self.optimize_inputs()
        self.calc_affine_feedback()
        self.calc_vertex_flows()
        # Half Space Represintation
        self.A, self.b = pp.duality.compute_polytope_halfspaces(np.array(self.vMat))

    def optimize_inputs(self):
        """Terminate Simplex by solving the all invariance conditions based RCP"""
        eps = 1e-6
        self.calc_ourward_normals()
        # Optimization problem
        u = [cvx.Variable((self.m, 1)) for i in range(0, self.n+1)]
        constraints = []
        obj = 0
        for i in range(0, self.n+1):
            obj += 0
            # Invariance Constraints
            I = list(np.arange(0, self.n+1))    # Index Set
            _ = I.pop(i)                      # Pop the index opposite to current face
            for j in I:
                hj = rs(self.h[j, :], [self.n, 1])
                constraints += [hj.T @ (self.asys.A @ rs(self.vMat[i, :], [self.n, 1]) + self.asys.B @ u[i] + self.asys.a) <= -eps]
            # input constraints
            constraints += [u[i] <= self.u_max, u[i]>= self.u_min]
        prob = cvx.Problem(cvx.Maximize(obj), constraints=constraints)
        if not prob.is_dcp():
            raise(ValueError("The problem doesn't follow DCP rules!!"))
        prob.solve()
        if prob.status in ["infeasible", "unbounded"]:
            raise(ValueError("The optimization problem is {}.\nCheck control input Limits!!".format(prob.status)))
        for i in range(0, self.n+1):
            self.uMat[i, :] = rs(u[i].value, [1, self.m])
