import numpy as np
import numpy.testing as npt
from qspace.bases import spf, dspf
from qspace.sampling import sphere, space
from numpy.testing import (assert_, assert_equal, assert_almost_equal,
                           assert_array_almost_equal, run_module_suite,
                           assert_array_equal)


def test_dual_spherical_polar_fourier():
    radial_order = 3
    angular_rank = 4
    zeta = 60.0
    dual_spherical_polar_fourier = dspf.DualSphericalPolarFourier(radial_order, 
        angular_rank, zeta)
    assert_equal(dual_spherical_polar_fourier.radial_order, radial_order)
    assert_equal(dual_spherical_polar_fourier.angular_rank, angular_rank)
    assert_equal(dual_spherical_polar_fourier.zeta, zeta)

    r = 1.0
    theta = np.pi * np.random.rand(1)
    phi = 2 * np.pi * np.random.rand(1)
    assert_array_almost_equal(
         dual_spherical_polar_fourier.spherical_function(r, theta, phi), 0)


def test_dimension():
    radial_order = 3
    angular_rank = 4
    dim_dspf = 45
    assert_equal(dim_dspf, dspf.dimension(radial_order, angular_rank))


def test_indices():
    radial_order = 4
    angular_rank = 6
    dim_dspf = dspf.dimension(radial_order, angular_rank)
    for i in range(dim_dspf):
        n = dspf.index_n(i, radial_order, angular_rank)
        l = dspf.index_l(i, radial_order, angular_rank)
        m = dspf.index_m(i, radial_order, angular_rank)
        assert_equal(i, dspf.index_i(n, l, m, radial_order, angular_rank))
        assert_equal(l % 2, 0)
        assert_(np.abs(m) <= l)
        assert_(n < radial_order)


if __name__ == '__main__':
    run_module_suite()

