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
    s = rs(s, [n, 1])
    for i in range(m-1):
        s_k = rs(phi[i, :], [n, 1])
        s_kp1 = rs(phi[i+1, :], [n, 1])
        c = s - s_k
        c1 = s_kp1 - s_k
        M = np.column_stack([c1, c])
        if np.linalg.matrix_rank(M) < n:
            return i
    raise(ValueError("Not in the segments"))


def chain_flow(n, s_in, del_s, phi):
    """ Returns xi and s_o vectors."""
    # sanity checks
    if n != np.shape(s_in)[0]:
        raise(ValueError("Dimensional Mismatch!!"))
    m, _ = np.shape(phi)
    k = which_seg(n, s_in, phi)
    s_kp1 = rs(phi[k+1, :], [n, 1])
    diff_vec = s_kp1 - s_in
    diff_norm = np.linalg.norm(diff_vec)
    if (diff_norm <= del_s):
        xi = diff_vec/diff_norm
        s_o = s_in + del_s * xi
    else:
        s_kp2 = rs(phi[k+2, :], [n, 1])
        diff_vec_2 = s_kp2 - s_kp1
        diff_norm_2 = np.linalg.norm(diff_vec_2)
        if diff_norm + diff_norm_2 - del_s >=0:
            s_o = s_kp1 + (del_s-diff_norm) * (diff_vec_2/diff_norm_2)    # this moves the s_o slowly near the edges.
            vec = s_o - s_in
            xi = vec/np.linalg.norm(vec)
        else:
            print("Making del_s/2")
            del_s = del_s/2
            chain_flow(n, s_in, del_s, phi)
            #raise(ValueError("Too big \\delta s"))
    return s_o, xi


if __name__=="__main__":
    import space as spc
    s0, xi = chain_flow(2, np.matrix([[1], [-1]]), 0.1, spc.W)
    print(s0)
    print(xi)
