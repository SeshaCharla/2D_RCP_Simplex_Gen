import numpy as np
from numpy import reshape as rs
import intersection as intsc

def lambda_hit(n, v0, alpha0, polytope):
    """Find the lambda P
    Always use normalized alpha
    Note: Each row of the matrix A, b represent a half space."""
    A = polytope.A
    b = rs(polytope.b, [np.shape(A)[0], 1])
    c = b - A @ v0
    d = A @ alpha0
    ld = []
    for i in range(np.shape(c)[0]):
        try:
            if d[i, 0]!= 0:
                ldi = c[i, 0]/d[i, 0]
                if ldi>=0 and np.all(A @ (v0 + ldi*alpha0) - b <= 0):
                    ld.append(ldi)
                else:
                    ld.append(np.Inf)
            else:
                ld.append(np.Inf)
        except:
            ld.append(np.Inf)
    return min(ld)


def lambda_ptope(n, v0, alpha0, ptope_list):
    """Find the min lambda inside the polytopic sets"""
    #sanity checks
    if n != np.shape(v0)[0] or n!= np.shape(alpha0)[0]:
        raise(ValueError("Dimension Mismatch!!"))
    ld_list = [lambda_hit(n, v0, alpha0, ptope) for ptope in ptope_list]
    return min(ld_list)


def lambda_int(n, v0, alpha0, phi):
    """Intersection with support curve"""
    int_lst = intsc.seg_ray(n, v0, alpha0, phi)
    if int_lst:
        ld = [int_tu[0] for int_tu in int_lst]
        return min(ld)
    else:
        return np.Inf


def lambda_max(n, v0, alpha0, phi, ptope_list, dels_max):
    """Find the lambda_max allowable for construction of simplex
    dels_max = np.linalg.norm(so-sin)"""
    ld_p = lambda_ptope(n, v0, alpha0, ptope_list)
    ld_int = lambda_int(n, v0, alpha0, phi)
    ld_max = min([0.6*ld_p, 0.6*ld_int])
    return min([ld_max, dels_max])
