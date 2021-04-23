""" Normals to n-dimensional vectors
Ref: https://ef.gy/linear-algebra:normal-vectors-in-higher-dimensional-spaces
"""

import numpy as np


def normal_2(vec):
    """2 Normal of the vector"""
    N = np.matrix([[0, 1],
                   [-1, 0]])
    x = np.reshape(vec, [2, 1])
    h = N @ x
    n = h/np.linalg.norm(h)
    return n


# def normal_n(vecMat):
#     """ Normal vector of the given list of vectors
#         Rows of the vecMat are vectors
#     """
#     m, n = np.shape(vecMat)    # m = number of vectors, n = dimension
#     if m != n-1:
#         raise(ValueError("Not enough vectors"))
#     d = vecMat.T   # columns are the vectors
#     n = np.zeros(n)
#     j = np.append(np.arange(0, n), np.arange(n-2, -1, -1))
#     M = np.zeros([n-1, n-1])    # Minor matrix
#     for i in range(n):
#         for p in range(n-1):
#             for q in range(n-1):
#                 M[p, q] = d[j[i+p], j[q]


if __name__=="__main__":
    print(normal_2(np.array([0, 1])))
