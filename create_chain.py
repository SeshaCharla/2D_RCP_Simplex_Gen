import space as spc
import system as ss
import chain_funcs as cf
import numpy as np
from numpy import reshape as rs
import support_vecs as svc
import matplotlib.pyplot as plt
import pypoman as pp
import plot2D


chain = []
n = 2
F = spc.I
s_in = np.matrix([[1], [-1]])
del_s = 1
u_max = 6*np.ones([2, 1])
u_min = -6*np.ones([2, 1])
Sim = cf.init_chain(2, ss.lsys, F, s_in,  u_max, u_min, spc.W, spc.ptope_list)
Sim2 = cf.prop_chain(2, ss.lsys, Sim, u_max, u_min, spc.W, spc.ptope_list)
chain.append(Sim)
chain.append(Sim2)
# Plot
plt.figure()
pp.plot_polygon(spc.vobs.vertices)
pp.plot_polygon(spc.hobs.vertices)
pp.plot_polygon(spc.rgn.vertices)
plt.plot(spc.I[:, 0], spc.I[:, 1])
#plt.plot(spc.E[:, 0], spc.E[:, 1])
plt.plot(spc.W[:, 0], spc.W[:, 1])
plt.xticks(np.arange(-5, 7))
plt.yticks(np.arange(-6, 7))
plt.grid()
plot2D.plot2D_rcpSpx(chain[0])
plot2D.plot2D_rcpSpx(chain[1])

j = 0
old_spx = Sim2
while (svc.which_seg(n, s_in, spc.W) != (np.shape(spc.W))[0] -1) and j<3:
    Sim = cf.prop_chain(n, ss.lsys, old_spx, u_max,  u_min, spc.W, spc.ptope_list)
    s_in = Sim.so
    chain.append(Sim)
    plot2D.plot2D_rcpSpx(Sim)
    old_spx = Sim
    j = j + 1
    print(j)

plt.show()
