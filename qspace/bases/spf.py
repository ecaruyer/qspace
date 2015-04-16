###################
# qspace - spf.py #
###################
# This file contains the implementation of the spherical harmonic basis. #
##########################################################################

import numpy as np
import sh, utils
from scipy.special import lpmv, gamma, hyp1f1, legendre
from scipy.special.orthogonal import genlaguerre
from scipy.misc import factorial


# default parameters values
_default_radial_order = 3
_default_angular_rank = sh._default_rank
_default_zeta = 700.0


class SphericalPolarFourier:
    """A SphericalPolarFourier object represents a function expressed as a 
    linear combination of the truncated SPF basis elements.

    Parameters
    ----------
    radial_order : int
        The radial truncation order of the SPF basis.
    angular_rank : int
        The truncation rank of the angular part of the SPF basis.
    zeta : float
        The scale parameter of the SPF basis.
    """

    def __init__(self, radial_order=_default_radial_order, 
                 angular_rank=_default_angular_rank, zeta=_default_zeta):
        self.radial_order = radial_order
        self.angular_rank = angular_rank
        self.zeta = zeta
        self.coefficients = np.zeros((self.radial_order,
                                      sh.dimension(self.angular_rank)))


    def spherical_function(self, r, theta, phi):
        """The 3d function represented by the SPF object.

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
        for n in range(self.radial_order):
            if abs(self.coefficients[n]).max() > 0.0:
                sh_coefs = self.coefficients[n]
                spherical_harm = sh.SphericalHarmonics(sh_coefs)  
                result += \
                    spherical_harm.angular_function(theta, phi) * \
                    radial_function(r, n, self.zeta)
        return result


    def get_angular_rank(self):
        return self._angular_rank

    def set_angular_rank(self, value):
        if value % 2 != 0:
            raise ValueError("'angular_rank' only accepts even values.")
        self._angular_rank = value

    angular_rank = property(get_angular_rank, set_angular_rank)


    def odf_tuch(self):
        """Computes the Tuch ODF from the q-space signal attenuation expressed
        in the SPF basis, following [cheng-ghosh-etal:10].

        Returns
        -------
        spherical_harmonics : sh.SphericalHarmonics instance.
        """
        dim_sh = sh.dimension(self.angular_rank)
        sh_coefs = np.zeros(dim_sh)
        for j in range(dim_sh):
            l = sh.index_l(j)
            for n in range(self.radial_order):
                partial_sum = 0.0
                for i in range(n):
                    partial_sum += utils.binomial(i - 0.5, i) * (-1)**(n - i)
                sh_coefs[j] += partial_sum * self.coefficients[n, j] \
                  * kappa(zeta, n)
            sh_coefs[j] = sh_coefs[j] * legendre(l)(0)
        return sh.SphericalHarmonics(sh_coefs)


    def odf_marginal(self):
        """Computes the marginal ODF from the q-space signal attenuation 
        expressed in the SPF basis, following [cheng-ghosh-etal:10].

        Returns
        -------
        spherical_harmonics : sh.SphericalHarmonics instance.
        """
        dim_sh = sh.dimension(self.angular_rank)
    
        sh_coefs = np.zeros(dim_sh)
        sh_coefs[0] = 1 / np.sqrt(4 * np.pi)
    
        for l in range(2, self.angular_rank + 1, 2):
            for m in range(-l, l + 1):
                j = sh.index_j(l, m)
                for n in range(1, self.radial_order):
                    partial_sum = 0.0
                    for i in range(1, n + 1):
                        partial_sum += (-1)**i * \
                          utils.binomial(n + 0.5, n - i) * 2**i / i
                    sh_coefs[j] += partial_sum * kappa(self.zeta, n) * \
                      self.coefficients[n, j] * \
                      legendre(l)(0) * l * (l + 1) / (8 * np.pi)
        return sh.SphericalHarmonics(sh_coefs)


def matrix(r, theta, phi, radial_order=_default_radial_order, 
           angular_rank=_default_angular_rank, zeta=_default_zeta):
    """Returns the spherical polar Fourier observation matrix for a given set
    of points represented by their spherical coordinates.

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
    
    Returns
    -------
    H : array-like, shape (K, R)
        The observation matrix corresponding to the point set passed as input.
    """
    K = r.shape[0]
    H = np.zeros((K, radial_order, sh.dimension(angular_rank)))
    b_n_j = SphericalPolarFourier(radial_order, angular_rank, zeta)
    for n in range(H.shape[1]):
        for j in range(H.shape[2]):
            b_n_j.coefficients[:] = 0
            b_n_j.coefficients[n, j] = 1.0
            H[:, n, j] = b_n_j.spherical_function(r, theta, phi)
    return H.reshape(K, dimension(radial_order, angular_rank))


def dimension(radial_order, angular_rank):
    "Returns the dimension of the truncated SPF basis."
    return radial_order * sh.dimension(angular_rank)


def index_i(n, l, m, radial_order, angular_rank):
    """Returns flattened index i based on radial rank, the angular degree l and
    order m.
    """
    dim_sh = sh.dimension(angular_rank)
    j = sh.index_j(l, m)
    return n * dim_sh + j


def index_n(i, radial_order, angular_rank):
    "Returns radial rank n corresponding to flattened index i."
    dim_sh = sh.dimension(angular_rank)
    return i // dim_sh


def index_l(i, radial_order, angular_rank):
    "Returns angular degree l corresponding to flattened index i."
    dim_sh = sh.dimension(angular_rank)
    j = i % dim_sh
    return sh.index_l(j)


def index_m(i, radial_order, angular_rank):
    "Returns angular order m corresponding to flattened index i."
    dim_sh = sh.dimension(angular_rank)
    j = i % dim_sh
    return sh.index_m(j)


def L(radial_order, angular_rank):
    "Returns the angular regularization matrix as introduced by Assemlal."
    dim_sh = sh.dimension(angular_rank)
    diag_L = np.zeros((radial_order, dim_sh))
    for j in range(dim_sh):
        l =  sh.l(j)
        diag_L[:, j] = (l * (l + 1)) ** 2
    dim_spf = dimension(radial_order, angular_rank)
    return np.diag(diag_L.reshape(dim_spf))


def N(radial_order, angular_rank):
    "Returns the radial regularisation matrix as introduced by Assemlal."
    dim_sh = sh.dimension(angular_rank)
    diag_N = np.zeros((radial_order, dim_sh))
    for n in range(radial_order):
        diag_N[n, :] = (n * (n + 1)) ** 2
    dim_spf = dimension(radial_order, angular_rank)
    return np.diag(diag_N.reshape(dim_spf))


def kappa(zeta, n):
    "Returns the normalization constant of the SPF basis."
    return np.sqrt(2 / zeta**1.5 * factorial(n) / gamma(n + 1.5)) 


def radial_function(r, n, zeta):
    "Computes the radial part of the SPF basis."
    return genlaguerre(n, 0.5)(r**2 / zeta) * \
        np.exp(- r**2 / (2 * zeta)) * \
        kappa(zeta, n)


