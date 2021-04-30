import numpy as np
import space as spc
from numpy import reshape as rs


def chain_sup(s_in, del_s):
    """ Returns xi and s_o vectors."""
    n = np.shape(s_in)[0]
    k = spc.which_seg(s_in)
    s_kp1 = rs(spc.W[k+1, :], [n, 1])
    diff_vec = s_kp1 - s_in
    diff_norm = np.linalg.norm(diff_vec)
    if diff_norm <= del_s:
        xi = diff_vec/diff_norm
        s_o = s_in + del_s * xi
    else:
        s_kp2 = rs(spc.W[k+2, :], [n, 1])
        diff_vec_2 = s_kp2 - s_kp1
        diff_norm_2 = np.linalg.norm(diff_vec_2)
        if diff_norm + diff_norm_2 - del_s >=0:
            s_o = s_kp1 + (del_s-diff_norm) * (diff_vec_2/diff_norm_2)    # this moves the s_o slowly near the edges.
            vec = s_o - s_in
            xi = vec/np.linalg.norm(vec)
        else:
            raise(ValueError("Too big \\delta s"))
    return s_o, xi



if __name__=="__main__":
    s0, xi = chain_sup(np.matrix([[1], [-1]]), 0.1)
    print(s0)
    print(xi)
