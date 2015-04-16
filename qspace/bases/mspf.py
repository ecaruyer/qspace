####################
# qspace - mspf.py #
####################
# This file contains the implementation of the spherical harmonic basis. #
##########################################################################

import numpy as np
from scipy.special import lpmv, gamma, hyp1f1, legendre
from scipy.special.orthogonal import genlaguerre
from scipy.misc import factorial
import sh, spf


# default parameters values
_default_radial_order = spf._default_radial_order
_default_angular_rank = sh._default_rank
_default_zeta = spf._default_zeta


class ModifiedSphericalPolarFourier:
    """This class implements the modified SPF basis, for the reconstruction of 
    a continuous function.

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
        """The 3d function represented by the mSPF object.

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
        for n in range(self.radial_order - 1):
            if np.abs(self.coefficients[n]).max() > 0.0:
                sh_coefs = self.coefficients[n]
                spherical_harm = sh.SphericalHarmonics(sh_coefs)
                result += spherical_harm.angular_function(theta, phi) * \
                    radial_function(r, n, self.zeta)
        return result


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
    zeta : float
        The scale parameter of the mSPF basis.
    
    Returns
    -------
    H : array-like, shape (K, R)
        The observation matrix corresponding to the point set passed as input.
    """
    K = r.shape[0]
    H = np.zeros((K, radial_order - 1, sh.dimension(angular_rank)))
    b_n_j = ModifiedSphericalPolarFourier(radial_order, angular_rank, zeta)
    for n in range(H.shape[1]):
        for j in range(H.shape[2]):
            b_n_j.coefficients[:] = 0
            b_n_j.coefficients[n, j] = 1.0
            H[:, n, j] = b_n_j.spherical_function(r, theta, phi)
    return H.reshape(K, dimension(radial_order, angular_rank))


def to_spf_matrix(radial_order=_default_radial_order, 
                  angular_rank=_default_angular_rank, zeta=_default_zeta):
    "Computes the transition matrix from modified SPF basis to SPF basis."
    M = np.zeros((spf.dimension(radial_order, angular_rank), 
               dimension(radial_order, angular_rank)))
    for i in range(M.shape[0]):
        n_i = spf.index_n(i, radial_order, angular_rank)
        l_i = spf.index_l(i, radial_order, angular_rank)
        m_i = spf.index_m(i, radial_order, angular_rank)
        kappa_ni = spf.kappa(zeta, n_i)
        for j in range(M.shape[1]):
            n_j = index_n(j, radial_order, angular_rank)
            l_j = index_l(j, radial_order, angular_rank)
            m_j = index_m(j, radial_order, angular_rank)
            chi_nj = chi(zeta, n_j)
            if (l_i == l_j and m_i == m_j):
                if n_i <= n_j:
                    M[i, j] = 3 * chi_nj / (2 * kappa_ni)
                else:
                    if n_i == n_j + 1:
                        M[i, j] = - (n_j + 1) * chi_nj / kappa_ni
    return M


def dimension(radial_order, angular_rank):
    "Returns the dimension of the truncated mSPF basis."
    return (radial_order - 1) * sh.dimension(angular_rank)


index_i = spf.index_i
index_n = spf.index_n
index_l = spf.index_l
index_m = spf.index_m


def chi(zeta, n):
    "Returns the normalization constant of the mSPF basis."
    return np.sqrt(2 / zeta**1.5 * factorial(n) / gamma(n + 3.5)) 


def radial_function(r, n, zeta):
    "Computes the radial part of the mSPF basis."
    return genlaguerre(n, 2.5)(r**2 / zeta) * \
        r**2 / zeta * np.exp(- r**2 / (2 * zeta)) * chi(zeta, n)


def Lambda(radial_order, angular_rank, zeta=_default_zeta):
    """The Laplace regularization is computed by matrix multiplication 
    (x-x0)^T Lambda (x-x0).
    """
    max_degree = 2 * (radial_order + 1)
    gammas = gamma(np.arange(max_degree) + 0.5)

    dim = dimension(radial_order, angular_rank)
    L = np.zeros((dim, dim))
    dim_sh = sh.dimension(angular_rank)
    for n1 in range(radial_order - 1):
        chi1 = chi(zeta, n1)
        for n2 in range(radial_order - 1):
            chi2 = chi(zeta, n2)
            for j1 in range(dim_sh):
                l1 = sh.index_l(j1)
                coeffs = __Tcoeffs(n1, n2, l1)
                degree = coeffs.shape[0]
                matrix_entry = chi1 * chi2 / (2 * np.sqrt(zeta)) * \
                    np.dot(coeffs, gammas[range(degree-1, -1, -1)])
                for j2 in range(dim_sh):
                    l2 = sh.index_l(j2)
                    if j1 == j2:
                        L[n1 * dim_sh + j1, n2 * dim_sh + j2] = matrix_entry
    return L


def v(radial_order, angular_rank, zeta=_default_zeta):
    "The vector x0 for Laplace regularization is -Lambda^-1 v."
    max_degree = 2 * (radial_order + 1)
    gammas = gamma(np.arange(max_degree) + 0.5)

    dim = dimension(radial_order, angular_rank)
    v = np.zeros(dim)
    dim_sh = sh.dimension(angular_rank)

    for n in range(radial_order - 1):
        chi1 = chi(zeta, n)
        coeffs = __Tcoeffs(n, -1, 0)
        degree = coeffs.shape[0]
        v[n * dim_sh] = chi1 / (2 * np.sqrt(zeta)) \
            * np.dot(coeffs, gammas[range(degree-1, -1, -1)])
    return v


def __F_n(n):
    """F_n(q) = \chi_n \exp(-q^2 / 2\zeta) P_n(q)
    and P_n(q) = q^2 / zeta * L_n^{5/2}(q^2 / zeta)"""
    if n == -1:
        return np.poly1d([1.0])
    else:
        a = np.poly1d([1, 0.0, 0.0])
        return a * genlaguerre(n, 2.5)(a)


def __diffFn(p):
    """F_n'(q) = \chi_n \exp(-q^2 / 2\zeta) * 
    (-q / \zeta * P_n(q) + P_n'(q))"""
    a = np.poly1d([-1, 0.0])
    return a * p + p.deriv()


def __h_i_poly(n, l):
    """h_i(q) =  \chi_n \exp(-q^2 / 2\zeta) * h_i_poly(q)"""
    F0n = __F_n(n)
    F1n = __diffFn(F0n)
    F2n = __diffFn(F1n)
    a = np.poly1d([1.0, 0.0])
    b = (F0n / a)[0]         # Polynomial euclidian division
    return a * F2n + 2 * F1n - l * (l + 1) * b


def __Tcoeffs(ni, nj, l):
    """The entry (i, j) of laplace matrix is 
    $\chi_{n(i)}\chi_{n(j)} 
    \int_0^\infty \exp(-q^2/\zeta) T_{i,j}(q^2/\zeta)\,\mathrm{d}q$.
    This function returns the coefficients of T."""
    Tij = __h_i_poly(ni, l) * __h_i_poly(nj, l)
    degree = Tij.coeffs.shape[0]
    coeffs = Tij.coeffs[range(0, degree, 2)]
    return coeffs

