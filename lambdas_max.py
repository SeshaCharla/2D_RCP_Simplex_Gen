from math import exp
from system import A
import numpy as np
import space as spc
from numpy import reshape as rs


def lambda_hit(v0, alpha0, polytope):
    """Find the lambda P
    Always use normalized alpha"""
    alpha0n = alpha0/np.linalg.norm(alpha0)
    A = polytope.A
    b = rs(polytope.b, [np.shape(spc.rgn.A)[0], 1])
    c = b - A @ v0
    d = A @ alpha0n
    ld = []
    for i in range(np.shape(c)[0]):
        try:
            ldi = c[i, 0]/d[i, 0]
            if ldi>=0:
                ld.append(ldi)
            else:
                ld.append(np.Inf)
        except:
            ld.append(np.Inf)
    return min(ld)


def lambda_ints(v0, alpha0):
    """Intersection with support curve"""
    m, n = np.shape(spc.W)
    ld = []
    for i in range(m-1):
        sk = rs(spc.W[i,:], [n, 1])
        skp1 = rs(spc.W[i+1, :], [n, 1])
        try:
            A = np.column_stack([alpha0, -(skp1-sk)])
            b = sk-v0
            lds  = np.linalg.solve(A, b)
            if (0<=lds[1, 0]<=1) and (lds[0, 0] >= 0):
                ld.append(lds[0,0])
            else:
                ld.append(np.Inf)
        except:
            ld.append(np.Inf)
    return min(ld)


def lambda_intc(v0, alpha0, s_in, s_out):
    """Intersection with support curve"""
    m, n = np.shape(spc.W)
    ld = []
    try:
        A = np.column_stack([alpha0, -(s_out-s_in)])
        b = s_in-v0
        lds  = np.linalg.solve(A, b)
        if (lds[1, 0]>=0) and (lds[0, 0] >= 0):
            ld.append(lds[0,0])
        else:
            ld.append(np.Inf)
    except:
        ld.append(np.Inf)
    return min(ld)


def lambda_max(v0, alpha0, s_in, s_out):
    """Find the lambda_max allowable for construction of simplex"""
    ld_p = lambda_hit(v0, alpha0, spc.rgn)
    ld_ov = lambda_hit(v0, alpha0, spc.vobs)
    ld_oh = lambda_hit(v0, alpha0, spc.hobs)
    ld_ints = lambda_ints(v0, alpha0)
    ld_intc = lambda_intc(v0, alpha0, s_in, s_out)
    ld_max = min([0.75*ld_p, 0.75*ld_oh, 0.75*ld_ov, ld_intc, 0.5*ld_ints])
    return ld_max


if __name__=="__main__":
    print(lambda_hit(np.matrix([[0.5], [-1.5]]), np.matrix([[0], [-1]]), spc.rgn))
    print(lambda_hit(np.matrix([[0.5], [-1.5]]), np.matrix([[0], [-1]]), spc.vobs))
    print(lambda_hit(np.matrix([[0.5], [-1.5]]), np.matrix([[0], [-1]]), spc.hobs))
    print(lambda_ints(np.matrix([[0.5], [-1.5]]), np.matrix([[0], [-1]])))
    print(lambda_intc(np.matrix([[0.5], [-1.5]]), np.matrix([[0], [-1]]), np.matrix([[1.25], [-2]]), np.matrix([[1], [-3]])))
    print(lambda_max(np.matrix([[0.5], [-1.5]]), np.matrix([[0], [-1]]), np.matrix([[1.25], [-2]]), np.matrix([[1], [-3]])))
