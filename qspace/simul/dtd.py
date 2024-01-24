#!/usr/bin/env python
"""This module contains utility functions for the diffusion tensor
distribution. This implements methods in [1].

References
----------
.. [1]  Westin, C. F., Knutsson, H., Pasternak, O., Szczepankiewicz, F.,
   Özarslan, E., van Westen, D., ... & Nilsson, M. (2016). Q-space trajectory
   imaging for multidimensional diffusion MRI of the human brain. Neuroimage,
   135, 345-362.
"""
import numpy as np
from abc import ABC, abstractmethod
from qspace.sampling import multishell as ms
from qspace.sampling import sphere
import functools


# Let's define the isotropic 4th order tensor, the bulk and shear modulus
_e_iso = np.eye(6) / 3
_e_bulk = np.array([[1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]]) / 9
_e_shear = _e_iso - _e_bulk


class DTD(ABC):
    """Abstract class defining the interface for the diffusion tensor
    distribution.
    """
    @abstractmethod
    def first_moment(self):
        """Computes the first moment (average tensor) of the distribution.

        Returns
        -------
        D : array-like, shape (3, 3)
        """
        pass
 

    @abstractmethod
    def second_moment(self):
        """Computes the second moment (the average of D^2) of the distribution.
  
        Returns
        -------
        M : array-like, shape (6, 6)
        """
        pass


    @abstractmethod
    def signal(self, b_tensor):
        """Computes the signal that corresponds to a B-tensor.

        Parameters
        ----------
        b_tensor : array-like, shape (3, 3)
        """
        pass


    def micro_fa(self):
        """Computes the microscopic fractional anisotropy of the diffusion
        tensor distribution."""
        s = self.second_moment()
        c_mu = 3 * np.sum(s * _e_shear) / (2 * np.sum(s * _e_iso))
        return np.sqrt(c_mu)


    def orientation_parameter(self):
        """Computes the orientation parameter of the diffusion tensor 
        distribution."""
        s = self.second_moment()
        d = voigt_6(self.first_moment())
        op_squared = np.sum(np.outer(d, d) * _e_shear) 
        op_squared /= np.sum(s * _e_shear)
        return np.sqrt(op_squared)


    def size_variance(self):
        """Computes the orientation parameter of the diffusion tensor 
        distribution."""
        s = self.second_moment()
        d = voigt_6(self.first_moment())
        c = s - np.outer(d, d)
        c_md = np.sum(c * _e_bulk) / np.sum(s * _e_bulk)
        return c_md



class DiscreteDTD(DTD):
    """This represents a discrete diffusion tensor distribution; this
    implements the abstract class DTD.
    """
    def __init__(self, tensors, weights):
        """Constructor.

        Parameters
        ----------
        tensors : array-like, shape (nb_tensors, 3, 3)
        weights : array-like, shape (nb_tensors, )
        """
        nb_tensors = tensors.shape[0]
        if weights.shape[0] != nb_tensors:
            raise ValueError("Shape of `weights` and `tensors` mismatch.")
        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError("The sum of `weights` should be approximately 1.")
        self.tensors = tensors
        self.weights = weights

    def first_moment(self):
        return np.einsum("i,ijk->jk", self.weights, self.tensors)

    def second_moment(self):
        voigt = self.tensors[:, [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
        voigt[:, 3:] *= np.sqrt(2)
        return np.einsum("i,ij,ik->jk", self.weights, voigt, voigt)

    def signal(self, b_tensor):
        tenso   


@functools.lru_cache()
def _random_orientations(nb_orientations):
    directions = ms.optimize(1, [nb_orientations], np.array([[1.0]]))
    return directions


def dtd_from_op(orientation_parameter, nb_tensors, micro_fa=1.0, trace=2.0e-3,
                size_variance=0):
    """Creates a (discrete) tensor distribution with prescribed orientation 
    parameter, and (optionally) microscopic fractional anisotropy and trace. 
    There is no variance in shape (i.e. all tensors in the distribution have 
    the same triplet of eigenvlaues), the only variance is in orientation.

    Parameters
    ----------
    orientation_parameter : double
    nb_tensors : int
    micro_fa : double
    trace : double
        Mean trace of the individual tensors in the distribution.
    size_variance : double
        Corresponds to the C_MD parameter in Westin et al. This is a normalized
        size variance, between 0 (no variance in size) and 1.
    """
    lambda1, lambda2 = _fa_to_evals(micro_fa, trace)
    if size_variance > 0:
        shape = (1 - size_variance) / size_variance
        scale = 1.0 / shape
        normalized_mds = np.random.gamma(shape, scale, size=nb_tensors)
    else:
        normalized_mds = np.ones(nb_tensors)
    print(np.mean(normalized_mds))
    directions = _random_orientations(nb_tensors)
    thetas, phis = sphere.to_spherical(directions)
    a = np.sqrt(1 - orientation_parameter)
    new_thetas = np.arcsin(a * np.sin(thetas))
    new_directions = sphere.to_cartesian(new_thetas, phis)
    rank1_tensors = np.einsum("ij,ik->ijk", new_directions, new_directions)
    identities = np.repeat(np.eye(3)[np.newaxis, ...], nb_tensors, axis=0)
    tensors = lambda2 * identities + (lambda1 - lambda2) * rank1_tensors
    tensors *= normalized_mds[:, np.newaxis, np.newaxis]
    weights = np.ones(nb_tensors) / nb_tensors
    return DiscreteDTD(tensors, weights)
   

def voigt_6(D):
    """Converts a 3x3 matrix into its 6 dimensional Voigt notation (compatible
    with the notations in Westin et al.)"""
    d = np.zeros(6)
    d[0] = D[0][0]
    d[1] = D[1][1]
    d[2] = D[2][2]

    d[3] = np.sqrt(2) * D[1][2]
    d[4] = np.sqrt(2) * D[0][2]
    d[5] = np.sqrt(2) * D[0][1]
    return np.transpose(d)


def reverse_voigt_6(d):
    """Converts a 6 dimensional vector written in the Voigt convention back to 
    its 3x3 matrix notation (compatible with the notations in Westin et al.)"""
    d = np.transpose(d)
    D = np.zeros((3,3))
    D[0][0] = d[0]
    D[1][1] = d[1]
    D[2][2] = d[2]
    D[1][2] = D[2][1] = d[3] / np.sqrt(2)
    D[0][2] = D[2][0] = d[4] / np.sqrt(2)
    D[0][1] = D[1][0] = d[5] / np.sqrt(2)
    return D


def _fa_to_evals(fa, trace):
    """This is a helper function : provided a target fa and trace, computes
    lambda1 and lambda2 = lambda3 so that the corresponding tensor has the
    right fa and trace.
    """
    # we write lambda2 = espilon * lambda1; let's solve for epsilon first
    # we need to find epsilon s.t.(2f - 1)\epsilon^2 + 2\epsilon + f - 1 = 0
    # (where f = FA^2)
    a, b, c = 2 * fa**2 - 1, 2, fa**2 - 1
    delta = b**2 - 4 * a * c
    epsilon1 =  (-b + np.sqrt(delta)) / (2 * a)
    epsilon2 =  (-b - np.sqrt(delta)) / (2 * a)
    if epsilon1 >= 0 and epsilon2 >= 0:
        epsilon = min(epsilon1, epsilon2)
    else:
        epsilon = max(epsilon1, epsilon2)
    lambda1 = trace / (1 + 2 * epsilon)
    lambda2 = epsilon * lambda1
    return lambda1, lambda2
    

def isotropic_dtd(micro_fa, nb_tensors, trace=2.0e-3):
    """Creates a (discrete) tensor distribution that is macroscopically
    isotropic, but with prescribed microscopic fractional anisotropy. There is
    no variance in shape (i.e. all tensors in the distribution have the same 
    triplet of eigenvlaues), the only variance is in orientation.

    Parameters
    ----------
    micro_fa : double
        Prescribed microscopic FA.
    nb_tensors : int
    trace : double
        Trace of the individual tensors in the distribution.
    """
    lambda1, lambda2 = _fa_to_evals(micro_fa, trace)
    directions = _random_orientations(nb_tensors)
    rank1_tensors = np.einsum("ij,ik->ijk", directions, directions)
    identities = np.repeat(np.eye(3)[np.newaxis, ...], nb_tensors, axis=0)
    tensors = lambda2 * identities + (lambda1 - lambda2) * rank1_tensors
    weights = np.ones(nb_tensors) / nb_tensors
    return DiscreteDTD(tensors, weights)
   


if __name__ == "__main__":
    micro_fa = 0.5
    nb_tensors = 250
    dtd1 = isotropic_dtd(micro_fa, nb_tensors)

    estimated_c_mds = []
    op = 0.5
    gt_c_mds = np.linspace(0, 0.25, 10)
    for gt_c_md in gt_c_mds:
        dtd2 = dtd_from_op(op, nb_tensors, micro_fa, size_variance=gt_c_md)
        estimated_c_mds.append(dtd2.size_variance())

    from matplotlib import pyplot as plt
    plt.plot(gt_c_mds, estimated_c_mds)
    plt.plot(gt_c_mds, gt_c_mds, "k--")
    plt.xlabel("Ground truth $C_\mathrm{MD}$")
    plt.ylabel("Estimated $C_\mathrm{MD}$")
    plt.show()

    exit()
    b1 = np.diag([2000, 500, 500])
    b2 = np.diag([500, 2000, 500])
    b3 = np.diag([500, 500, 2000])

    print(dtd2.micro_fa())
    print(dtd2.orientation_parameter())
    print(dtd2.signal(b1), dtd2.signal(b2), dtd2.signal(b3))
    print(dtd2.first_moment())
