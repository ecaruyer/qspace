#!/usr/bin/env python3
from abc import ABC
import numpy as np

class Substrate(ABC):
    name = "Substrate"
    pass


class Free(Substrate):


    def __init__(self):
        self.camino_options = {"-substrate":  "empty"}

    def __str__(self):
        return "free"


class CylindersLattice(Substrate):
    def __init__(self, radius, packing="hexagonal", separation=None, 
                 density=None, **kwargs):
        self.camino_options = {}
        self.camino_options["-geometry"] = "cylinder"
        if packing.lower() not in ["hexagonal", "square"]:
            raise(ValueError("packing is either 'hexagonal' or 'square'"))
        self.radius = radius
        self.packing = packing
        if separation is not None:
            if density is not None:
                raise(ValueError("provide either separation or density."))
            self.separation = separation
            self.density = _cylinder_density_from_separation(radius, 
              separation)
        else:
            if density is None:
                raise(ValueError("provide either separation or density."))
            self.separation = _cylinder_separation_from_density(radius, 
              density, packing)
            self.density = density
        if self.separation < 2 * self.radius:
            raise(ValueError("cylinders separation should be >= 2*radius."))
        if packing == "hexagonal":
            self.camino_options["-packing"] = "hex"
        elif packing == "square":
            self.camino_options["-packing"] = "square"
        self.camino_options["-cylinderrad"] = "%.3e" % self.radius
        self.camino_options["-cylindersep"] = "%.3e" % self.separation


    def __str__(self):
        return (f"CylindersLattice_{self.packing}_r={self.radius:.3e}_"
                f"d={self.density:.2f}")

   
def _cylinder_separation_from_density(radius, density, packing="hexagonal"):
    """Computes the separation between closest cylinder axes for a target 
    density.
    """
    if packing == "hexagonal":
        return np.sqrt(2 * np.pi * radius**2 / (np.sqrt(3) * density))
    elif packing == "square":
        return np.sqrt(np.pi * radius**2 / (density))
    else:
        raise(ValueError("Packing %s not implemented." % packing))

   
def _cylinder_density_from_separation(radius, separation, packing="hexagonal"):
    """Computes the density of a given cylinders packing."""
    if packing == "hexagonal":
        return (2 * np.pi * radius**2) / (np.sqrt(3) * separation**2)
    elif packing == "square":
        return np.pi * radius**2 / separation**2
    else:
        raise(ValueError("Packing %s not implemented." % packing))


def substrate(**kwargs):
    """Initialize a Substrate instance, picking the subclass corresponding to
    the `name` passed in the parameters.
    """
    if parameters["name"] == "Free":
        return Free()
    if parameters["name"] == "CylindersLattice":
        return CylindersLattice(kwargs["radius"], **kwargs)


if __name__=="__main__":
    radius = 1.0e-6
    density = 0.9
    substrate = CylindersLattice(radius, density=density)
    print(substrate)
