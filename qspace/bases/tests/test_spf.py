import numpy as np
import numpy.testing as npt
from qspace.bases import spf
from qspace.sampling import sphere, space
from numpy.testing import (assert_, assert_equal, assert_almost_equal,
                           assert_array_almost_equal, run_module_suite,
                           assert_array_equal)


def test_spherical_polar_fourier():
    radial_order = 3
    angular_rank = 4
    zeta = 60.0
    spherical_polar_fourier = spf.SphericalPolarFourier(radial_order, 
        angular_rank, zeta)
    assert_equal(spherical_polar_fourier.radial_order, radial_order)
    assert_equal(spherical_polar_fourier.angular_rank, angular_rank)
    assert_equal(spherical_polar_fourier.zeta, zeta)

    r = 1.0
    theta = np.pi * np.random.rand()
    phi = 2 * np.pi * np.random.rand()
    assert_almost_equal(
         spherical_polar_fourier.spherical_function(r, theta, phi), 0)


def test_dimension():
    radial_order = 3
    angular_rank = 4
    dim_spf = 45
    assert_equal(dim_spf, spf.dimension(radial_order, angular_rank))


def test_indices():
    radial_order = 4
    angular_rank = 6
    dim_spf = spf.dimension(radial_order, angular_rank)
    for i in range(dim_spf):
        n = spf.index_n(i, radial_order, angular_rank)
        l = spf.index_l(i, radial_order, angular_rank)
        m = spf.index_m(i, radial_order, angular_rank)
        assert_equal(i, spf.index_i(n, l, m, radial_order, angular_rank))
        assert_equal(l % 2, 0)
        assert_(np.abs(m) <= l)
        assert_(n < radial_order)


def test_matrix():
    shell_radii = [1.0, 2.0, 3.0, 4.0, 5.0]
    nb_shells = len(shell_radii)
    K_s = 64
    shell = sphere.jones(K_s)
    points = np.vstack([radius * shell for radius in shell_radii])
    r, theta, phi = space.to_spherical(points)

    radial_order = 4
    angular_rank = 4
    zeta = 1.0
    H = spf.matrix(r, theta, phi, radial_order, angular_rank, zeta)

    y = np.exp(-r**2 / 2) * points[:, 2]**2
    x = np.dot(np.linalg.pinv(H), y)
    dim_spf = spf.dimension(radial_order, angular_rank)
    for i in range(dim_spf):
        n = spf.index_n(i, radial_order, angular_rank)
        l = spf.index_l(i, radial_order, angular_rank)
        m = spf.index_m(i, radial_order, angular_rank)
        if m != 0 or l > 2 or n > 1:
            assert_almost_equal(x[i], 0)
    assert_almost_equal(y, np.dot(H, x))


if __name__ == '__main__':
    run_module_suite()

