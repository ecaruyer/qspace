################################################
# Author: Emmanuel Caruyer <caruyer@gmail.com> #
#                                              #
# Code to generate multiple-shell sampling     #
# schemes, with optimal angular coverage. This #
# implements the method described in Caruyer   #
# et al., MRM 69(6), pp. 1534-1540, 2013.      #
# This software comes with no warranty, etc.   #
################################################
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, art3d
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Create a colormap for the b-values
cm_dict = {'red':  ((0.0, 0.0, 0.0),
                    (1.0, 1.0, 1.0)),
          'green': ((0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),
          'blue':  ((0.0, 1.0, 1.0),
                    (1.0, 0.0, 0.0))}
blue_red = LinearSegmentedColormap('blue_red', cm_dict)


_epsi = 1.0e-9


def rotation_matrix(u, v):
    """
    returns a rotation matrix R s.t. Ru = v
    """
    # the axis is given by the product u x v
    u = u / np.sqrt((u ** 2).sum())
    v = v / np.sqrt((v ** 2).sum())
    w = np.asarray([u[1] * v[2] - u[2] * v[1], 
                    u[2] * v[0] - u[0] * v[2], 
                    u[0] * v[1] - u[1] * v[0]])
    if (w ** 2).sum() < _epsi:
        #The vectors u and v are collinear
        return np.eye(3)

    # computes sine and cosine
    c = np.dot(u, v)
    s = np.sqrt((w ** 2).sum())

    w = w / s
    P = np.outer(w, w)
    Q = np.asarray([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    R = P + c * (np.eye(3) - P) + s * Q
    return R


def draw_shell(r, ax, color='0.4'):
    """
    draw a sphere outline in matplotlib.
    """
    theta, phi = np.pi * (90 - ax.elev) / 180., np.pi * ax.azim / 180.
    view_axis = np.asarray([np.sin(theta)*np.cos(phi),
                            np.sin(theta)*np.sin(phi),
                            np.cos(theta)])
    R = rotation_matrix(np.asarray([1, 0, 0]), view_axis)

    # a circle centered at [1, 0, 0] with radius r
    M = 50
    t = np.linspace(0, 2 * np.pi, M)
    circleX = np.zeros((M, 3))
    circleX[:, 1] = r * np.cos(t)
    circleX[:, 2] = r * np.sin(t)

    circle = np.dot(circleX, R.T)
    ax.plot(circle[:, 0], circle[:, 1], circle[:, 2], color='k')


def draw_circles(positions, r=0.04):
    """
    draw circular patches (lying on a sphere) at given positions.
    """
    # a circle centered at [1, 0, 0] with radius r
    M = 20
    t = np.linspace(0, 2 * np.pi, M)
    circleX = np.zeros((20, 3))
    circleX[:, 1] = r * np.cos(t)
    circleX[:, 2] = r * np.sin(t)
    
    nbPoints = positions.shape[0]
    circles = np.zeros((nbPoints, M, 3))
    for i in range(positions.shape[0]):
        norm = np.sqrt((positions[i] ** 2).sum())
        point = positions[i] / norm 
        R1 = rotation_matrix(np.asarray([1, 0, 0]), point)
        circles[i] = positions[i] + np.dot(R1, circleX.T).T
    return circles


def draw_points_reprojected(vects, S, Ks, rs, ax):
    """
    Draw the vectors on a unit sphere. 

    Parameters
    ----------
    vects : array-like shape (N, 3) 
            Contains unit vectors. Shells should be stored consecutively.
    S : number of shells
    KS : array-lke, shpae (S,)
         list of integers, corresponding to number of points per shell.
    rs : array-like shape (S,)
         shell radii. This is simply used for color mapping.
    ax : the matplolib axes instance to plot in.
    """
    indices = np.cumsum(Ks).tolist()
    indices.insert(0, 0)

    v = np.asarray([0., 0., 1.])
    elev = 90
    azim = 0  
    ax.view_init(azim=azim, elev=elev)

    # First plot sphere outline
    draw_shell(0.99, ax)

    vects = np.copy(vects)
    vects[vects[:, 2] < 0] *= -1

    for s, r in zip(range(S), rs):
        color = blue_red(r / np.max(rs))
        positions = vects[indices[s]:indices[s + 1]]
        circles = draw_circles(positions)
        ax.add_collection(art3d.Poly3DCollection(circles, facecolors=color, 
            linewidth=0))

    max_val = 0.6
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(-max_val, max_val)
    ax.axis("off")


def draw_points_sphere(vects, S, Ks, rs, fig):
    """
    Draw the vectors on multi-shell. 

    Parameters
    ----------
    vects : array-like shape (N, 3) 
            Contains unit vectors. Shells should be stored consecutively.
    S : number of shells
    KS : array-lke, shpae (S,)
         list of integers, corresponding to number of points per shell.
    rs : array-like shape (S,)
         Shell radii.
    fig : the matplolib figure instance to plot in.
    """
    indices = np.cumsum(Ks).tolist()
    indices.insert(0, 0)
    
    for s in range(S):
        ax = fig.add_subplot(1, S, s + 1, projection='3d')
        draw_points(vects[indices[s]:indices[s+1]],
                    1, [Ks[s]], [rs[s]], ax, np.max(rs))
    return


def draw_points(vects, S, Ks, rs, ax, r_max=None):
    """
    Draw the vectors on multi-shell. 

    Parameters
    ----------
    vects : array-like shape (N, 3) 
            Contains unit vectors. Shells should be stored consecutively.
    S : number of shells
    KS : array-lke, shpae (S,)
         list of integers, corresponding to number of points per shell.
    rs : array-like shape (S,)
         Shell radii.
    ax : the matplolib axes instance to plot in.
    """
    v = np.asarray([0., 0., 1.])
    elev = 90
    azim = 0  
    ax.view_init(azim=azim, elev=elev)

    # Plot spheres outlines
    for r in rs:
        draw_shell(0.99 * r, ax)

    vects = np.copy(vects)
    vects[vects[:, 2] < 0] *= -1

    if rs is None:
        rs = np.linspace(0., 1., S + 1)[1:]

    indices = np.cumsum(Ks).tolist()
    indices.insert(0, 0)
    
    for s, r in zip(range(S), rs):
        vects[indices[s]:indices[s + 1]] *= r
   
    if r_max is None:
        r_max = np.max(rs)
 
    for s, r in zip(range(S), rs):
        dots_radius = np.sqrt(r / r_max) * 0.04
        color = blue_red(r / r_max)
        positions = vects[indices[s]:indices[s + 1]]
        circles = draw_circles(positions, dots_radius)
        ax.add_collection(art3d.Poly3DCollection(circles, facecolors=color, 
            linewidth=0))

    max_val = 0.6 * r_max
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(-max_val, max_val)
    ax.axis("off")
