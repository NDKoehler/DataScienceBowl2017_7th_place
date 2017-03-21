from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import numpy as np
from numpy import linalg
from random import random
import code

def getMinVolEllipse(P=None, tolerance=0.01, v_center_px=[], v_diam_px=[]):
    """ Find the minimum volume ellipsoid which holds all the points"""
    (N, d) = np.shape(P)
    d = float(d)

    # Q is the working array
    Q = np.vstack([np.copy(P.T), np.ones(N)])
    QT = Q.T

    # initializations
    err = 1.0 + tolerance
    u = (1.0 / N) * np.ones(N)

    # Khachiyan Algorithm
    while err > tolerance:
        V = np.dot(Q, np.dot(np.diag(u), QT))
        try:
            M    = np.diag(np.dot(QT , np.dot(linalg.inv(V), Q)))    # M the diagonal vector of an NxN matrix
        except linalg.linalg.LinAlgError:
            center = [int(x) for x in v_center_px]
            radii  = np.array(v_diam_px)/5.

            rotation = [[0,0,0],[0,0,0],[0,0,0]]

            return (center, radii, rotation) 

        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u = (1.0 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u

    # center of the ellipse
    center = np.dot(P.T, u)

    # the A matrix for the ellipse
    A = linalg.inv(
                   np.dot(P.T, np.dot(np.diag(u), P)) -
                   np.array([[a * b for b in center] for a in center])
                   ) / d

    # Get the values to return
    U, s, rotation = linalg.svd(A)
    radii = 1.0/np.sqrt(s)

    return (center, radii, rotation)

def getEllipsoidVolume(radii):
    """Calculate the volume of the blob"""
    return 4./3.*np.pi*radii[0]*radii[1]*radii[2]

def plotEllipsoid(center, radii, rotation, ax=None, plotAxes=False, cageColor='b', cageAlpha=0.2):
    """Plot an ellipsoid"""
    make_ax = ax == None
    if make_ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)

    # cartesian coordinates that correspond to the spherical angles:
    # x,y,z in shape (100x100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v)) #all possible x for these angles
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center
            [x[i,j],y[i,j],z[i,j]] = [round(x[i,j]),round(y[i,j]),round(z[i,j])]
    # code.interact(local=dict(globals(), **locals()))
    ax.plot_wireframe(x, y, z,  rstride=10, cstride=10, color=cageColor, alpha=cageAlpha)

    if make_ax:
        plt.show()
def plot__ellipse(P):
    # find the ellipsoid
    (center, radii, rotation) = getMinVolEllipse(P, .01)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot points
    ax.scatter(P[:,0], P[:,1], P[:,2], color='g', marker='o', s=100)
    # plot ellipsoid
    plotEllipsoid(center, radii, rotation, ax=ax, plotAxes=True)

    plt.show()

def plot__scatter(P):
    print("plotting scatter")
    # find the ellipsoid
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P[:,0], P[:,1], P[:,2], color='b', marker='o', s=100)
    plt.show()


def plot__both_scatters(new, old):
    print("plotting scatter")
    # find the ellipsoid
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(new[:,0], new[:,1], new[:,2], color='b', marker='o', s=100)
    ax.scatter(old[:,0], old[:,1], old[:,2], color='g', marker='o', s=100)

    plt.show()
