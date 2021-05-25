from support_vecs import calc_sin, chain_sup
import space as spc
import system as ss
import chain_funcs as cf
import numpy as np
from numpy import reshape as rs
import support_vecs as svc
import matplotlib.pyplot as plt
import pypoman as pp


chain = []
F = spc.I
s_in = calc_sin(F)
del_s = 2.8
s_o, xi = svc.chain_sup(s_in, del_s)
uMat = cf.init_chain(F, xi, ss.lsys)

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
while (spc.which_seg(s_in) != (np.shape(spc.W))[0] -1) and j<50:
    Sim = cf.prop_chain(F, uMat, ss.lsys, s_in, del_s)
    F = Sim.vMat[1:,:]
    uMat = Sim.uMat[1:,:]
    s_in = svc.calc_sin(F)
    chain.append(Sim)
    pp.plot_polygon(Sim.vMat)
    j = j + 1
    print(j)
for i in range(3):
    plt.plot([Sim.vMat[i, 0], 0.2*(Sim.alphaMat[i, 0]/np.linalg.norm(Sim.alphaMat[i,:]))+Sim.vMat[i, 0]], [Sim.vMat[i, 1], 0.2*(Sim.alphaMat[i, 1]/np.linalg.norm(Sim.alphaMat[i,:]))+Sim.vMat[i, 1]])
    plt.plot([Sim.vMat[i, 0], 0.5*(Sim.xi[0, 0])+Sim.vMat[i, 0]], [Sim.vMat[i, 1], 0.5*(Sim.xi[1, 0])+Sim.vMat[i, 1]], "--k")

plt.show()
