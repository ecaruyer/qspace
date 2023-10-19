import numpy as np
import os
from qspace.sampling import multishell as ms

__location__ = os.path.dirname(__file__)


def jones(K):
    """Returns a set of K points minimizing DK Jones' electrostatic repulsion
    energy.
    """
    if K > 256:
        nb_shells = 1
        nb_points_per_shell = [K]
        weights = np.ones((1, 1))
        return ms.optimize(nb_shells, nb_points_per_shell, weights)
    filename = os.path.join(__location__, "data/jones_%03d.txt" % K)
    dirs = np.loadtxt(filename)
    return dirs


def koay(K):
    """Returns a set of K points in the upper hemisphere, generated 
    analytically following CG Koay's method _[1].

    References
    ----------
    1. Koay, Cheng Guan. "A simple scheme for generating nearly uniform
    distribution of antipodally symmetric points on the unit sphere." Journal
    of computational science 2.4 (2011): 377-381.
    """
    nb_latitudes = int(np.sqrt(K * np.pi / 8))
    ks = np.zeros(nb_latitudes, dtype=int)
    latitudes = (0.5 + np.arange(nb_latitudes)) * np.pi / (2 * nb_latitudes)
    ratios = 2 * np.sin(latitudes) * np.sin(np.pi / (4 * nb_latitudes))
    thetas = np.zeros(K)
    phis = np.zeros(K)
    current_index = 0
    for n in range(nb_latitudes):
        if n < nb_latitudes - 1:
            ks[n] = int(np.rint(ratios[n] * K))
        else:
            ks[n] = K - np.sum(ks)
        thetas[current_index:current_index + ks[n]] = latitudes[n]
        phis[current_index:current_index + ks[n]] = (0.5 + np.arange(ks[n])) \
                                                  * 2 * np.pi / ks[n]
        current_index += ks[n]
    return to_cartesian(thetas, phis)


def to_cartesian(theta, phi):
    """Returns points on the unit sphere provided their polar and azimuthal 
    coordinates.

    Parameters
    ----------
    theta : array-like, shape (K, )
        The polar angles.
    phi : array-like, shape (K, )
        The azimuthal angles.

    Returns
    -------
    points : array-like, shape (K, 3)
        The Cartesian coordinates corresponding to the points passed as input.
    """
    K = theta.shape[0]
    points = np.zeros((K, 3))
    points[:, 0] = np.sin(theta) * np.cos(phi)
    points[:, 1] = np.sin(theta) * np.sin(phi)
    points[:, 2] = np.cos(theta)
    return points


def to_spherical(points):
    """Returns points on the unit sphere provided their polar and azimuthal 
    coordinates.

    Parameters
    ----------
    points : array-like, shape (K, 3)
    
    Returns
    -------
    theta : array-like, shape (K, )
        The polar angles.
    phi : array-like, shape (K, )
        The azimuthal angles.
    """
    norms = np.sqrt(np.sum(points**2, 1))
    if not np.allclose(norms, 1.0):
        raise(ValueError("spherical_to_cartesian expects unit vectors."))
    theta = np.arccos(points[:, 2])
    phi = np.arctan2(points[:, 1], points[:, 0])
    return theta, phi


def random_uniform(K):
    """Creates a set of K pseudo-random unit vectors, following a uniform 
    distribution on the half sphere.

    Parameters
    ----------
    K : int
        Number of directions to generate.

    Returns
    -------
    points : array-like, shape (K, 3)
    """
    phi = 2 * np.pi * np.random.rand(K)

    r = 2 * np.sqrt(np.random.rand(K))
    theta = 2 * np.arcsin(r / 2)
    
    points = np.zeros((K, 3))
    points[:, 0] = np.sin(theta) * np.cos(phi)
    points[:, 1] = np.sin(theta) * np.sin(phi)
    points[:, 2] = np.cos(theta)

    return points


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    K = 200
    points = koay(K)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*points.T)
    plt.axis("off")
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.set_box_aspect([1,1,1])
    plt.show()
