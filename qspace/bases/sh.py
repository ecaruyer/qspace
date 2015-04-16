###################
# qspace - shm.py #
###################
# This file contains the implementation of the spherical harmonic basis. #
##########################################################################

import numpy as np
from scipy.special import lpmv, gamma, hyp1f1, legendre
from scipy.special.orthogonal import genlaguerre
from scipy.misc import factorial
from sampling import *


_default_rank = 4


class SphericalHarmonics:
    """This class describes a real, antipodally symmetric spherical function by 
    its spherical harmonics coefficients. It also contains a set of static 
    methods related to the definition and manipulation of spherical harmonics.

    Parameters
    ----------
    coefficients : array-like, shape (R, )
        A 1d array of coefficients representing the function.
    """
    
    def __init__(self, coefficients):
        self._create_from_coefficients(coefficients)

    
    def _create_from_coefficients(self, coefficients):
        rank = 2
        while True:
            dim_sh = dimension(rank)
            if len(coefficients) == dim_sh:
                self.rank = rank
                self.coefficients = coefficients
                return
            elif len(coefficients) < dim_sh:
                raise ValueError("Invalid dimension for SH coefficients.")
            rank += 2


    def get_rank(self):
        return self._rank

    def set_rank(self, value):
        if value % 2 != 0:
            raise ValueError("'rank' only accepts even values.")
        self._rank = value

    rank = property(get_rank, set_rank)


    def get_coefficients(self):
        return self._coefficients

    def set_coefficients(self, value):
        if value.shape[0] != dimension(self.rank):
            raise ValueError("Coefficients shape and rank mismatch.")
        self._coefficients = value

    coefficients = property(get_coefficients, set_coefficients)


    def angular_function(self, theta, phi):
        """Computes the function at angles theta, phi.

        Parameters
        ----------
        theta : array-like
            Polar angles, using the physics convention.
        phi : array-like
            Azimuthal angle, using the physics convention.
        """
        coefs = self.coefficients
        result = 0
        rank = self.rank
        for l in range(0, rank+1, 2):
            for m in range(-l, l+1):
                j = index_j(l, m)
                if coefs[j] != 0.0:
                    if m < 0:
                        result += coefs[j] * sqrt(2)         \
                        * sqrt((2*l + 1) * factorial(l + m)  \
                        / (4 * pi * factorial(l - m)))       \
                        * (-1) ** (-m)                       \
                        * lpmv(-m, l, cos(theta)) * cos(m * phi)  
                    if m == 0:
                        result += coefs[j]                   \
                        * sqrt((2*l + 1) * factorial(l - m)  \
                        / (4 * pi * factorial(l + m)))       \
                        * lpmv(m, l, cos(theta))
                    if m > 0:
                        result += coefs[j] * sqrt(2)         \
                        * sqrt((2*l + 1) * factorial(l - m)  \
                        / (4 * pi * factorial(l + m)))       \
                        * lpmv(m, l, cos(theta)) * sin(m * phi)
        return result


def dimension(rank):
    """Returns the dimension of the spherical harmonics basis for a given 
    rank.
    """
    return (rank + 1) * (rank + 2) / 2


def index_j(l, m):
    "Returns the flattened index j of spherical harmonics."
    # l is between 0 and rankSH, m is btw -l and l
    if np.abs(m) > l:
        raise NameError('SphericalHarmonics.j: m must lie in [-l, l]')
    return int(l + m + (2 * array(range(0, l, 2)) + 1).sum())


def index_l(j):
    "Returns the degree l of SH associated to index j"
    l = 0
    while dimension(l) - 1 < j:
        l += 2
    return l


def index_m(j):
    "Returns the order m of SH associated to index j"
    l = index_l(j)
    return j - dimension(l) + l + 1


def matrix(theta, phi, rank=_default_rank):
    """Returns the spherical harmonics observation matrix for a given set
    of directions represented by their polar and azimuthal angles.

    Parameters
    ----------
    theta : array-like, shape (K, )
        Polar angles of the direction set.
    phi : array-like, shape (K, )
        Azimuthal angles of the direction set.
    rank : int
        The truncation rank of the SH basis.
    
    Returns
    -------
    H : array-like, shape (K, R)
        The observation matrix corresponding to the direction set passed as
        input.
    """
    dim_sh = dimension(rank)
    sh = SphericalHarmonics(np.zeros(dim_sh))
    N = theta.shape[0]
    H = np.zeros((N, dim_sh))
    for j in range(dim_sh):
        sh.coefficients[:] = 0
        sh.coefficients[j] = 1.0
        H[:, j] = sh.angular_function(theta, phi)
    return H


def L(rank=_default_rank):
    """Returns Laplace-Beltrami regularization matrix.

    Parameters
    ----------
    rank : int
        The truncation rank of the SH basis.
    """

    dim_sh = dimension(rank)
    L = np.zeros((dimSH, dimSH))
    for j in range(dimSH):
        l =  index_l(j)
        L[j, j] = - (l * (l + 1))
    return L


def P(rank=_default_rank):
    "returns the Funk-Radon operator matrix"
    dim_sh = dimension(rank)
    P = np.zeros((dim_sh, dim_sh))
    for j in range(dim_sh):
        l =  index_l(j)
        P[j, j] = 2 * np.pi * legendre(l)(0)
    return P

