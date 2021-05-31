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
        c_ = s_kp1 - s
        ck = s_kp1 - s_k
        if np.all(ck == 0):
            raise(ValueError("The segment is a point!!"))
        M = np.column_stack([ck, c])
        if np.linalg.matrix_rank(M, 1e-6) == 1:   # This implies that the
            mag_c = np.linalg.norm(c)
            mag_c_ = np.linalg.norm(c_)
            mag_ck = np.linalg.norm(ck)
            if mag_c <= mag_ck and mag_c_ <= mag_ck:
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
