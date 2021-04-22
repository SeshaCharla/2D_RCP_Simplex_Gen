import numpy as np



class asys():
    """Linear affine system class"""
    def __init__(self, A, B, a):
        self.A = A
        self.B = B
        self.a = a


A = np.matrix([[-0.1, 0.2],[-1, 0.4]])
B = np.matrix([[1, 0],[0, 1]])
a = np.matrix([[-0.2], [-0.1]])

lsys = asys(A, B, a)

def f(t, x, u):
    """ The system"""
    xv = np.matrix([[x[0]], [x[1]]])
    xd = lsys.A @ xv + lsys.B*u + lsys.a
    return np.array([xd[0, 0], xd[1, 0]])


if __name__ == "__main__":
    from scipy.integrate import odeint
    import matplotlib.pyplot as plt
    t = np.linspace(0, 10, 100)
    u = 2
    x0 = np.array([0, 0])
    sol = odeint(f, x0, t, args=(u,), tfirst=True)
    plt.plot(t, sol[:, 1])
    plt.show()
