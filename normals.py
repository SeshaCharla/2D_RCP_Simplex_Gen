""" Normals to n-dimensional vectors
Ref: https://ef.gy/linear-algebra:normal-vectors-in-higher-dimensional-spaces
"""

import numpy as np
from numpy import reshape as rs


def normal(vecMat, n):
    """function that calculates the normal in given dimension
    -- Normal vector of the given list of vectors Rows of the vecMat are vectors
    -- Normal
    """
    p, q = np.shape(vecMat)
    if n-1 != p or n != q :
        raise(ValueError("The dimensions don't match!"))
    if n<2:
        raise(ValueError("You can't find the normal for less than 2 dimensions!!"))
    elif n==2:
        return normal_2(vecMat)
    else:
        return normal_n(vecMat, n)


def normal_2(vec):
    """2 Normal of the vector"""
    N = np.matrix([[0, 1],
                   [-1, 0]])
    x = rs(vec, [2, 1])
    h = N @ x
    n = h/np.linalg.norm(h)
    return n


def normal_n(vecMat, n):
    """ Normal vector of the given list of vectors
        Rows of the vecMat are vectors
    """
    m, p = np.shape(vecMat)    # m = number of vectors, n = dimension
    if n != p:
        raise(ValueError("Given dimensions don't match!"))
    if m != n-1:
        raise(ValueError("Not enough vectors!!"))
    nMat_half = vecMat.T   # columns are the vectors
    nMat = np.append(nMat_half, nMat_half, axis=0)
    baseVecs = np.zeros([n, n])
    for i in range(n):
        baseVecs[i, i] = 1
    nVec = np.zeros([n, 1])
    for i in range(n):
        nVec += (-1)**i * rs(baseVecs[:, i], [n, 1]) * np.linalg.det(nMat[i+1:i+n, :])
    h = nVec/np.linalg.norm(nVec)
    return h


if __name__=="__main__":
    print(normal(np.matrix([[1, 0]]), 2))
