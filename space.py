import numpy as np
import pypoman as pp
from numpy import reshape as rs


class ptope():
    """Convex Polytope class"""
    def __init__(self, vMat):
        n, m = np.shape(vMat)
        self.vMat = vMat
        self.vertices = [self.vMat[i, :].A1 for i in range(n)]
        self.A, self.b = pp.duality.compute_polytope_halfspaces(self.vertices)




"""General way of representing vertex sets using matrices: rows -- vertics"""

vobs_vMat = np.matrix([[0, 1],
                           [0, -3],
                           [-1, -3],
                           [-1, 1]])
hobs_vMat = np.matrix([[0, 1],
                           [0, 0],
                           [3, 0],
                           [3, 1]])
rgn_vMat = np.matrix([[-5, -6],
                          [-5, 5],
                          [5, 5],
                          [5, -6]])

I = np.matrix([[0.5, -1.5],
               [1.5, -0.5]])
E = np.matrix([[3.5, 4.5],
               [4.5, 3.5]])
W = np.matrix([[1, -1],
               [1.25, -2],
               [1.25, -3],
               [0.75, -3.75],
               [-0, -4],
               [-1, -4],
               [-2, -3.5],
               [-3.75, -1],
               [-4, 0],
               [-4, 1],
               [-4, 2],
               [-2, 3],
               [-1, 3.25],
               [1, 3.5],
               [2, 3.75],
               [3, 4],
               [4, 3.75]])

vobs = ptope(vobs_vMat)
hobs = ptope(hobs_vMat)
rgn  = ptope(rgn_vMat)


def which_seg(s):
    """ Finds the end points of the segments and returns the row index of the starting segment:
    s_k, s_kp1, k
    """
    n = 2
    m, _ = np.shape(W)
    s = rs(s, [2, 1])
    for i in range(m-1):
        s_k = rs(W[i, :], [n, 1])
        s_kp1 = rs(W[i+1, :], [n, 1])
        c = s - s_k
        c1 = s_kp1 - s_k
        M = np.column_stack([c1, c])
        if np.linalg.matrix_rank(M) < n:
            return i
    raise(ValueError("Not in the segments"))


if __name__=="__main__":
    import matplotlib.pyplot as plt
    plt.figure()
    pp.plot_polygon(vobs.vertices)
    pp.plot_polygon(hobs.vertices)
    pp.plot_polygon(rgn.vertices)
    plt.plot(I[:, 0], I[:, 1])
    plt.plot(E[:, 0], E[:, 1])
    plt.plot(W[:, 0], W[:, 1])
    plt.xticks(np.arange(-5, 7))
    plt.yticks(np.arange(-6, 7))
    plt.grid()
    plt.show()
    print(which_seg([1.125, -1.5]))
