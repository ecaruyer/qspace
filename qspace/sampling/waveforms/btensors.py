#!/usr/bin/env python
import numpy as np
from scipy.integrate import trapezoid, cumulative_trapezoid


gamma = 267.513e6
def compute_b(waveform, dt):
    """Compute the B-tensor provided a discretized gradient trajectory.

    Parameters
    ----------
    waveform : array-like, shape (nb_steps, 3)
        Gradient waveforms [T/m].
    dt : float
        Timestep [s].

    Result
    ------
    B : array-like, shape (3, 3)
        B-matrix [s/m^2]
    """
    qs = gamma * cumulative_trapezoid(waveform, dx=dt, axis=0, initial=0)
    qqt = np.einsum("ij,ik->ijk", qs, qs)
    B = trapezoid(qqt, dx=dt, axis=0)        
    return B


def transform_waveform(source_b_tensor, target_b_tensor):
    """Provided a source and a target b-tensors, computes the linear 
    transform to be applied to the trajectory to obtain the target b_tensor,
    following [1]_. Raises a ValueError if the source b-tensor has a rank that
    is lower than the target b-tensor (e.g. you want to transform a linear 
    tensor into a spherical tensor).
    
    Parameters
    ----------
    source_b_tensor : array-like, shape (3, 3)
        Source b-tensor [s/m^2].
    target_b_tensor : array-like, shape (3, 3)
        Target b-tensor [s/m^2].

    References
    ----------
    .. [1] Westin, Carl-Fredrik, et al. "Q-space trajectory imaging for
       multidimensional diffusion MRI of the human brain." NeuroImage 135
       (2016): 345-362.
    """
    # We rename the source and target b-tensors as b1 and b2 for clarity
    b1 = source_b_tensor
    b2 = target_b_tensor

    # We compute eigen decomposition of b1 and the target b-tensor
    w1, v1 = np.linalg.eigh(b1)
    w2, v2 = np.linalg.eigh(b2)

    # We make sure the rank of b1 is at least equal to the rank of b2
    epsilon = 1.0e-6
    if np.sum(w1 > epsilon) < np.sum(w2 > epsilon):
        raise ValueError("Trying to map a b-matrix to higher dimension.")

    # We compute the transform
    ratio = np.where(np.logical_or(w1 < epsilon, w2 < epsilon), 0, 
                     np.sqrt(w2 / w1))
    if np.any(np.isnan(ratio)):
        print(w1, w2, ratio)
        raise ValueError("We have Nan values...")
    return np.dot(v2, np.dot(np.diag(ratio), v1.T))


if __name__ == "__main__":
    # We try the transform_waveform snippet.3
    import os
    __location__ = os.path.dirname(__file__)
    
    filename = os.path.join(__location__, "data/template_now.txt")
    template_waveform = np.loadtxt(filename)
    te = 80e-3
    nb_steps = template_waveform.shape[0]
    ts = np.linspace(0, te, nb_steps)
    dt = ts[1] - ts[0]
    gmax = 80e-3
    source_b_tensor = compute_b(gmax * template_waveform, dt)
    print(source_b_tensor)
