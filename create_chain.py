import space as spc
import system as ss
import chain_funcs as cf
import numpy as np
from numpy import reshape as rs
import support_vecs as svc
import matplotlib.pyplot as plt
import pypoman as pp


chain = []
n = 2
F = spc.I
s_in = np.matrix([[1], [-1]])
print(svc.which_seg(2, s_in, spc.W))
del_s = 0.1
u_max = 6*np.ones([n, 1])
u_min = -6*np.ones([n, 1])
chain.append(cf.init_chain(n, ss.lsys, F, s_in, del_s, u_max, u_min, spc.W, spc.ptope_list))

# Plot
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


j = 0
while (svc.which_seg(n, s_in, spc.W) != (np.shape(spc.W))[0] -1) and j<3:
    Sim = cf.prop_chain(n, ss.lsys, chain[-1], del_s, u_max,  u_min, spc.W, spc.ptope_list)
    s_in = Sim.so
    chain.append(Sim)
    pp.plot_polygon(Sim.vertices)
    j = j + 1
    print(j)
for i in range(3):
    plt.plot([Sim.vMat[i, 0], 0.2*(Sim.alphaMat[i, 0]/np.linalg.norm(Sim.alphaMat[i,:]))+Sim.vMat[i, 0]], [Sim.vMat[i, 1], 0.2*(Sim.alphaMat[i, 1]/np.linalg.norm(Sim.alphaMat[i,:]))+Sim.vMat[i, 1]])
    plt.plot([Sim.vMat[i, 0], 0.5*(Sim.xi[0, 0])+Sim.vMat[i, 0]], [Sim.vMat[i, 1], 0.5*(Sim.xi[1, 0])+Sim.vMat[i, 1]], "--k")

plt.show()
