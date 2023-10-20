"""This module contains functions required for reading/writing diffusion
encoding gradient tables. Currently supported formats include FSL bval/bvec,
MRTrix btable as well as Siemens dvs."""
import numpy as np
from qspace.sampling import space


def write(bvals, bvecs, filename, format="siemens"):
    """Writes a gradient table, provided as an array of b-values and an array
    of b-vectors, to a file. Supported formats are either "fsl", "mrtrix" or 
    "siemens".

    Parameters
    ----------
    bvals : array-like, shape (nb_points, )
    bvecs : array-like, shape (nb_points, 3)
    filename : string
    format : string, either "fsl", "mrtrix" or "siemens"
    """
    if bvecs.shape[1] != 3:
        raise ValueError("The b-vectors should contain 3 coordinates.")
    if bvals.shape[0] != bvecs.shape[0]:
        raise ValueError("Number of b-values and b-vectors do not match.")
    nb_points = bvals.shape[0]
    if format == "siemens":
        epsilon = 1.0e-6
        b_norms = np.sqrt(bvals / (epsilon + np.max(bvals)))
        b_table = b_norms[:, np.newaxis] * bvecs
        with open(filename, "r") as dvs_file:
            dvs_file.write("[directions=%d]\n" % nb_acquisitions)
            dvs_file.write("CoordinateSystem = xyz\n")
            dvs_file.write("Normalisation = none\n")
            for i in range(nb_points):
                dvs_file.write("Vector[%d] = ( %.6f, %.6f, %.6f )\n" % \
                  (i, b_table[i, 0], b_table[i, 1], b_table[i, 2]))
            dvs_file.close()
