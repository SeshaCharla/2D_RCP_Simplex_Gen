import matplotlib.pyplot as plt
import numpy as np
import pypoman as pp


def plot2D_flow(spx):
    """plots the 2D flow vectors of the simplex"""
    l = np.linalg.norm(spx.vMat[2,:]-spx.vMat[1,:])
    cen = np.mean(spx.vMat, axis=0)
    for i in range(3):
        plt.plot([spx.vMat[i, 0], 0.2*l*(spx.alphaMat[i, 0]/np.linalg.norm(spx.alphaMat[i,:]))+spx.vMat[i, 0]], [spx.vMat[i, 1], 0.2*l*(spx.alphaMat[i, 1]/np.linalg.norm(spx.alphaMat[i,:]))+spx.vMat[i, 1]], "--r", label="Closed-loop Vector Field")

    plt.plot([cen[0, 0], 0.3*l*(spx.xi[0, 0])+cen[0, 0]], [cen[0, 1], 0.3*l*(spx.xi[1, 0])+cen[0,1]], "--k", label="Flow Vector")

def plot2D_normals(spx):
    """plots the 2D normals of the simplices"""
    l = np.linalg.norm(spx.vMat[2,:]-spx.vMat[1,:])
    for i in range(3):
        v_list = list(spx.vertices).copy()
        vi = v_list.pop(i)
        c = (np.mean(np.array(v_list), axis=0))
        p = c + 0.05*l*spx.h[i,:]
        plt.plot([c[0], p[0]], [c[1], p[1]], "--g", label="Normals")

def plot2D_rcpSpx(spx):
    """Plot the entire rcpSimplex"""
    pp.plot_polygon(spx.vertices)
    plot2D_flow(spx)
    plot2D_normals(spx)
    plt.grid()
    _ = plt.axis("equal")

def plot2D_terminal_flow(spx):
    """plots the 2D flow vectors of the simplex"""
    l = np.linalg.norm(spx.vMat[2,:]-spx.vMat[1,:])
    for i in range(3):
        plt.plot([spx.vMat[i, 0], 0.2*l*(spx.alphaMat[i, 0]/np.linalg.norm(spx.alphaMat[i,:]))+spx.vMat[i, 0]], [spx.vMat[i, 1], 0.2*l*(spx.alphaMat[i, 1]/np.linalg.norm(spx.alphaMat[i,:]))+spx.vMat[i, 1]], "--r", label="Closed-loop Vector Field")

def plot2D_term_spx(spx):
    """plot the terminal simplex"""
    """Plot the entire rcpSimplex"""
    pp.plot_polygon(spx.vertices)
    plot2D_terminal_flow(spx)
    plot2D_normals(spx)
    plt.grid()
    _ = plt.axis("equal")
