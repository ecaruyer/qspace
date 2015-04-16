import numpy as np
import os


__location__ = os.path.dirname(__file__)


def jones(K):
    """Returns a set of K points minimizing DK Jones' electrostatic repulsion
    energy.
    """
    if K > 256:
        raise(ValueError("Spherical point sets available for up to K=256."))
    filename = os.path.join(__location__, "data/jones_%03d.txt" % K)
    dirs = np.loadtxt(filename)
    return dirs


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



