import matplotlib.pyplot as plt
import numpy as np


def plot2D_flow(spx):
    """plots the 2D flow vectors of the simplex"""
    l = np.linalg.norm(spx.vMat[2,:]-spx.vMat[1,:])
    for i in range(3):
        plt.plot([spx.vMat[i, 0], 0.2*l*(spx.alphaMat[i, 0]/np.linalg.norm(spx.alphaMat[i,:]))+spx.vMat[i, 0]], [spx.vMat[i, 1], 0.2*l*(spx.alphaMat[i, 1]/np.linalg.norm(spx.alphaMat[i,:]))+spx.vMat[i, 1]], "--r", label="Closed-loop Vector Field")
        plt.plot([spx.vMat[i, 0], 0.3*l*(spx.xi[0, 0])+spx.vMat[i, 0]], [spx.vMat[i, 1], 0.3*l*(spx.xi[1, 0])+spx.vMat[i, 1]], "--k", label="Flow Vector")
