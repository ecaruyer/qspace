import numpy as np
import numpy.testing as npt
from qspace.bases import sh
from qspace.sampling import sphere
from numpy.testing import (assert_, assert_equal, assert_almost_equal,
                           assert_array_almost_equal, run_module_suite,
                           assert_array_equal)


def test_spherical_harmonics():
    rank = 4
    dim_sh = sh.dimension(rank)
    coefs = np.zeros(dim_sh)
    spherical_harm = sh.SphericalHarmonics(coefs)
    assert_equal(spherical_harm.rank, rank)

    theta = np.pi * np.random.rand(1)
    phi = 2 * np.pi * np.random.rand(1)
    assert_array_almost_equal(spherical_harm.angular_function(theta, phi), 0)


def test_dimension():
    rank = 4
    dim_sh = 15
    assert_equal(dim_sh, sh.dimension(rank))


def test_indices():
    for j in range(100):
        l = sh.index_l(j)
        m = sh.index_m(j)
        assert_equal(j, sh.index_j(l, m))
        assert_equal(l % 2, 0)
        assert_(np.abs(m) <= l)


def test_matrix():
    K = 64
    rank = 6
    dim_sh = sh.dimension(rank)
    points = sphere.jones(K)
    theta, phi = sphere.to_spherical(points)
    H = sh.matrix(theta, phi, rank)
    y = points[:, 2]**2
    x = np.dot(np.linalg.pinv(H), y)
    for j in range(dim_sh):
        l =sh.index_l(j)
        m = sh.index_m(j)
        if m != 0 or l > 2:
            assert_almost_equal(x[j], 0)
    assert_array_almost_equal(y, np.dot(H, x))


if __name__ == '__main__':
    run_module_suite()

