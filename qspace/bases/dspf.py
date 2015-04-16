####################
# qspace - dspf.py #
####################
# This file contains the implementation of the spherical harmonic basis. #
##########################################################################

import numpy as np
from scipy.special import lpmv, gamma, hyp1f1, legendre
from scipy.special.orthogonal import genlaguerre
from scipy.misc import factorial
import sh, spf, utils


# default parameters values
_default_radial_order = spf._default_radial_order
_default_angular_rank = sh._default_rank
_default_zeta = spf._default_zeta


class DualSphericalPolarFourier:
    """This class implements the dual SPF basis, as defined by Cheng et al. 
    for the reconstruction of the Fourier transform of a function epressed in
    the SPF basis.

    Parameters
    ----------
    radial_order : int
        The radial truncation order of the mSPF basis.
    angular_rank : int
        The truncation rank of the angular part of the mSPF basis.
    zeta : float
        The scale parameter of the mSPF basis.
    """

    def __init__(self, radial_order=_default_radial_order, 
                 angular_rank=_default_angular_rank, zeta=_default_zeta):
        self.radial_order = radial_order
        self.angular_rank = angular_rank
        self.zeta = zeta
        self.coefficients = np.zeros((self.radial_order,
                                      sh.dimension(self.angular_rank)))


    def get_angular_rank(self):
        return self._angular_rank

    def set_angular_rank(self, value):
        if value % 2 != 0:
            raise ValueError("'angular_rank' only accepts even values.")
        self._angular_rank = value

    angular_rank = property(get_angular_rank, set_angular_rank)


    def spherical_function(self, r, theta, phi):
        """The 3d function represented by the DualSphericalPolarFourier object.

        Parameters
        ----------
        r : array-like, shape (K, )
            The radii of the points in q-space where to compute the spherical 
            function.
        theta : array-like, shape (K, )
            The polar angles of the points in q-space where to compute the 
            spherical function.
        phi : array-like, shape (K, )
            The azimuthal angles of the points in q-space where to compute the 
            spherical function.

        Returns
        -------
        f : array-like, shape (K, )
            The function computed at the points provided as input.
        """
        result = 0.0
        Y = sh.matrix(theta, phi, rank=self.angular_rank)
        F = np.zeros(Y.shape)
        for n in range(self.radial_order):
            if (np.abs(self.coefficients[n])).max() > 0.0:
                for l in range(0, self.angular_rank + 1, 2):
                    f_n_l = radial_function(r, n, l, self.zeta)
                    for m in range(-l, l + 1):
                        j = sh.j(l, m)
                        F[:, j] += self.coefficients[n, j] * f_n_l
        return (F * Y).sum(1)


def matrix(r, theta, phi, radial_order=_default_radial_order, 
           angular_rank=_default_angular_rank, zeta=_default_zeta):
    """Returns the dual spherical polar Fourier observation matrix for a given 
    set of points represented by their spherical coordinates.

    Parameters
    ----------
    r : array-like, shape (K, )
        The radii of the points in q-space where to compute the spherical 
        function.
    theta : array-like, shape (K, )
        The polar angles of the points in q-space where to compute the 
        spherical function.
    phi : array-like, shape (K, )
        The azimuthal angles of the points in q-space where to compute the 
        spherical function.
    radial_order : int
        The radial truncation order of the SPF basis.
    angular_rank : int
        The truncation rank of the angular part of the SPF basis.
    zeta : float
        The scale parameter of the mSPF basis.
    
    Returns
    -------
    H : array-like, shape (K, R)
        The observation matrix corresponding to the point set passed as input.
    """
    K = r.shape[0]
    H = np.zeros((K, radial_order, SphericalHarmonics.dimension(angular_rank)))
    b_n_j = dSPF(radial_order=radial_order, angular_rank=angular_rank, zeta=zeta)
    for n in range(H.shape[1]):
        for j in range(H.shape[2]):
            b_n_j.coefficients[:] = 0
            b_n_j.coefficients[n, j] = 1.0
            H[:, n, j] = b_n_j.sphericalFunction(r, theta, phi)
    return H.reshape(K, dimension(radial_order, angular_rank))


def dimension(radial_order, angular_rank):
    return radial_order * sh.dimension(angular_rank)


index_i = spf.index_i
index_n = spf.index_n
index_l = spf.index_l
index_m = spf.index_m


def radial_function(r, n, l, zeta):
    result = 0.0
    for i in range(n + 1):
        result += \
          (- 1)**i * utils.binomial(n + 0.5, n - i) / \
          factorial(i) * 2**(0.5 * l + i - 0.5) * \
          special.gamma(0.5 * l + i + 1.5) * \
          hyp1f1((2 * i + l + 3) * 0.5, l + 1.5, - 2 * np.pi**2 * r**2 * zeta)
    result = result * 4 * (-1)**(0.5 * l) * zeta**(0.5 * l + 1.5) * \
      np.pi**(l + 1.5) * r**l / special.gamma(l + 1.5) * spf.kappa(zeta, n)
    return result


