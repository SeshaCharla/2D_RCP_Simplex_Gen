import numpy as np
from numpy import reshape as rs


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
    #sanity checks
    if n != np.shape(v0)[0] or n!= np.shape(alpha0)[0]:
        raise(ValueError("Dimension Mismatch!!"))
    m, *_ = np.shape(phi)
    ld = []
    for i in range(m-1):
        sk = phi[i,:]
        skp1 = phi[i+1, :]
        s_matrix = np.append(sk, skp1, axis=0)
        lst_col = np.matrix([[0], [1], [1]])
        f_col = np.append(rs(alpha0, [1, n]), s_matrix, axis=0)
        A = np.append(f_col, lst_col, axis=1)
        b = np.append(rs(v0, [1, n]), np.matrix([[1]]), axis=1)
        try:
            lds  = np.linalg.solve(A, b)
            if (0<=lds[1, 0]<=1) and (0<=lds[2, 0]<=1) and (lds[0, 0] >= 0):
                ld.append(lds[0,0])
            else:
                ld.append(np.Inf)
        except:
            ld.append(np.Inf)
    return min(ld)


def lambda_max(n, v0, alpha0, phi, ptope_list, dels_max):
    """Find the lambda_max allowable for construction of simplex
    dels_max = np.linalg.norm(so-sin)"""
    ld_p = lambda_ptope(n, v0, alpha0, ptope_list)
    ld_int = lambda_int(n, v0, alpha0, phi)
    ld_max = min([0.5*ld_p, 0.5*ld_int])
    return min([ld_max, 2*dels_max])
