import numpy as np
import os
import sphere, multishell


def to_cartesian(r, theta, phi):
    """Returns points on the unit sphere provided their polar and azimuthal 
    coordinates.

    Parameters
    ----------
    r : array-like, shape (K, )
        The radii.
    theta : array-like, shape (K, )
        The polar angles.
    phi : array-like, shape (K, )
        The azimuthal angles.

    Returns
    -------
    points : array-like, shape (K, 3)
        The Cartesian coordinates corresponding to the points passed as input.
    """
    K = r.shape[0]
    points = np.zeros((K, 3))
    points[:, 0] = r * np.sin(theta) * np.cos(phi)
    points[:, 1] = r * np.sin(theta) * np.sin(phi)
    points[:, 2] = r * np.cos(theta)
    return points


def to_spherical(points):
    """Returns points on the unit sphere provided their polar and azimuthal 
    coordinates.

    Parameters
    ----------
    points : array-like, shape (K, 3)
    
    Returns
    -------
    r : array-like, shape (K, )
        The radii.
    theta : array-like, shape (K, )
        The polar angles.
    phi : array-like, shape (K, )
        The azimuthal angles.
    """
    r = np.sqrt(np.sum(points**2, 1))
    unit_vectors = points / r[:, np.newaxis]
    unit_vectors[r == 0] = 0
    theta, phi = sphere.to_spherical(unit_vectors)
    return r, theta, phi



