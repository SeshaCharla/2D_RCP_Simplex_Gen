import numpy as np
from numpy import reshape as rs


"""Module for functions that do segments to ray intersection and segments to plane intersection"""


def seg_ray(n, v, vec, segs):
    """Segment ray intersection
    returns \lambda, which_seg and corresponding deltas if there is an intersection
    intersections: list of above tuples
    If no intersection returns an empty list"""

    #sanity checks
    if n != np.shape(v)[0] or n!= np.shape(vec)[0]:
        raise(ValueError("Dimension Mismatch!!"))

    m, *_ = np.shape(segs)
    intsct = []
    lst_col = np.append(np.ones([n, 1]), np.zeros([1,1]), axis=0)
    b = np.append(v, np.ones([1, 1]), axis=0)
    for i in range(m-1):
        s_matrix = segs[i:i+2, :]
        M_vert = np.append(s_matrix, -rs(vec, [1, n]),  axis=0)
        M = np.append(M_vert, lst_col, axis=1)
        A = M.T
        if np.linalg.matrix_rank(A) == n+1:
            lds = np.linalg.solve(A, b)
            s = lds[0:2, 0]
            l  = lds[-1, 0]
            sum_s = np.sum(s)
            if any(s > 0) and all(s >= 0) and all(s <= 1) and np.round(sum_s, 2) == 1.0 and l>0 :
                intsct.append((l, i, s))
    return intsct


def seg_facet(n, F, segs):
    """Segment facet intersection
    returns \lambda's, which_seg and corresponding \delta's if there is an intersection
    intersections: list of above tuples
    If no intersection returns an empty list"""

    # sanity checks
    if n != np.shape(F)[0] or n != np.shape(F)[1] or n != np.shape(segs)[1]:
        raise(ValueError("Dimension Mismatch!!"))

    m, *_ = np.shape(segs)
    intsct = []
    lst_cols = np.append(np.append(np.ones([n, 1]), np.zeros([n, 1]), axis=1),
                         np.append(np.zeros([2,1]), np.ones([2, 1]), axis=1), axis=0)
    b = np.append(np.zeros([n, 1]), np.ones([2, 1]), axis=0)
    for i in range(m-1):
        s_matrix = segs[i:i+2, :]
        M_vert = np.append(F, -1*s_matrix,  axis=0)
        M = np.append(M_vert, lst_cols, axis=1)
        A = M.T
        if np.linalg.matrix_rank(A) == n+2:
            lds = np.linalg.solve(A, b)
            l = lds[0:n, 0]
            s  = lds[-2:, 0]
            sum_l = np.sum(l)
            sum_s = np.sum(s)
            l_conds = any(l > 0) and all(l >= 0) and all(l <= 1) and np.round(sum_l, 1) == 1.0
            s_conds = any(s > 0) and all(s >= 0) and all(s <= 1) and np.round(sum_s, 1) == 1.0
            if l_conds and s_conds:
                intsct.append((l, i, s))
    return intsct
