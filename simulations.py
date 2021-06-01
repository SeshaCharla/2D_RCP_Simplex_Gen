from scipy.integrate.odepack import odeint
from create_chain import chain
from numpy import reshape as rs
import numpy as np
import system as syst
import pypoman as pp

def odefun(t, x):
    """ode system"""
    A = syst.lsys.A
    B = syst.lsys.B
    a = syst.lsys.a
    splxs = []
    for i in range(len(chain)):
        if chain[i].in_simplex(x):
            splxs.append(i)
    u = chain[max(splxs)].get_u(x)
    y = rs(x, [2, 1])
    yd = A @ y + B @ u + a
    return np.array([yd[0, 0], yd[1, 0]])

def simulate(tf, x0):
    """Run odesimulation"""
    return odeint(odefun, x0, np.arange(0, tf, 0.01), tfirst=True)

if __name__=="__main__":
    import space as spc
    import matplotlib.pyplot as plt
    import plot2D
    tf = 10
    ld = np.linspace(0, 1, 15)

    x0 = [(ldm * spc.I[0,:] + (1-ldm)*spc.I[1,:]).A1 for ldm in ld]
    x0.pop(0)
    x0.pop(-1)
    x0.pop(-1)
    x0.pop(-1)
    x = [simulate(tf, xo) for xo in x0]

    plt.figure()
    plt.figure()
    pp.plot_polygon(spc.vobs.vertices)
    pp.plot_polygon(spc.hobs.vertices)
    pp.plot_polygon(spc.rgn.vertices)
    plt.plot(spc.I[:, 0], spc.I[:, 1])
    plt.plot(spc.E[:, 0], spc.E[:, 1])
    plt.plot(spc.W[:, 0], spc.W[:, 1])
    plt.xticks(np.arange(-5, 7))
    plt.yticks(np.arange(-6, 7))
    plt.grid()
    for sim in chain:
        plot2D.plot2d_spx(sim)
    for xi in x:
        plt.plot(xi[:, 0], xi[:, 1])
    plt.show()
