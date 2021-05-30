import numpy as np
from numpy import reshape as rs


def which_seg(n, s, phi):
    """ Finds the end points of the segments and returns the row index of the starting segment:
    s_k, s_kp1, k
    """
    # sanity checks
    if n != np.shape(s)[0]:
        raise(ValueError("Dimensional Mismatch!!"))

    m, _ = np.shape(phi)

    for i in range(m-1):
        s_k = rs(phi[i, :], [n, 1])
        s_kp1 = rs(phi[i+1, :], [n, 1])
        c = s - s_k
        c1 = s_kp1 - s_k
        M = np.column_stack([c1, c])
        if np.linalg.matrix_rank(M, tol=1e-2) < n:
            return i
    raise(ValueError("Not in the segments"))


def chain_flow(n, s_in, phi):
    """ Returns xi and s_o vectors."""
    # sanity checks
    if n != np.shape(s_in)[0]:
        raise(ValueError("Dimensional Mismatch!!"))
    m, _ = np.shape(phi)
    k = which_seg(n, s_in, phi)
    s_k = rs(phi[k, :], [n, 1])
    s_kp1 = rs(phi[k+1, :], [n, 1])
    seg_length= np.linalg.norm((s_kp1-s_k))
    diff_vec = s_kp1 - s_in
    diff_norm = np.linalg.norm(diff_vec)
    if (k+1 < m-1) and ((diff_norm/seg_length) <= 0.25):     # The segment is not the last segment
        s_kp2 = rs(phi[k+2, :], [n, 1])
        s_c = 0.5*(s_kp1) + 0.5*(s_kp2)
        diff_vec_2 = s_c - s_in
        diff_norm_2 = np.linalg.norm(diff_vec_2)
        xi = diff_vec_2/diff_norm_2
        s_o = s_c
    else:
        xi = diff_vec/diff_norm
        s_o = s_kp1
    return s_o, xi
