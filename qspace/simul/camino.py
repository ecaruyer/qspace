#!/usr/bin/env python3
import numpy as np
import subprocess
import struct
from tempfile import mkstemp
import os
from .substrates import CylindersLattice


class CaminoSimulation():
    def __init__(self, substrate, trajfile, 
                 nb_walkers=1000, nb_steps=1000, duration=0.1, seed=1, 
                 initial="uniform", permeability=0.0, diffusivity=2.0e-9):
        """A CaminoSimulation object encapsulates all the necessary information
        to launch a simulation.

        Parameters
        ----------
        substrate : Substrate instance
        trajfile : str
        nb_walkers : int
        nb_steps : int
        duration : double
            Experiment duration [s].
        seed : int
        initial : str
            Either "uniform", "spike", "intra" or "extra".
        permeability : double
        diffusivity : double
            Free diffusion coefficient [m^2/s]
        """        
        self.substrate = substrate
        self.trajfile = trajfile
        self.nb_walkers = nb_walkers
        self.nb_steps = nb_steps
        self.duration = duration 
        self.seed = seed
        self.initial = initial
        self.permeability = permeability
        self.diffusivity = diffusivity
        

    def get_options(self):
        return {
            "-walkers"     : "%d" % self.nb_walkers,
            "-tmax"        : "%d" % (self.nb_steps - 1),
            "-duration"    : "%f" % self.duration, 
            "-trajfile"    : self.trajfile,
            "-seed"        : "%d" % self.seed,
            "-initial"     : self.initial,
            "-p"           : "%f" % self.permeability,
            "-diffusivity" : "%.2e" % self.diffusivity,
        }


    def run(self):
        command = ["datasynth", 
                   "-voxels", "1"]
        for key, value in self.get_options().items():
            command.extend([key, value])
        for key, value in self.substrate.camino_options.items():
            command.extend([key, value])
        subprocess.call(command, stdout=subprocess.DEVNULL, 
                                 stderr=subprocess.STDOUT)


    def __str__(self):
        return "_".join([f"{key}={value}" \
                         for key, value in self.get_options().items()])

    
def _read_header(fhandle):
    """
    Structure of the trajfile (per Camino documentation)

    Header:
      dynamics duration (double)
      number of spins (double)
      tmax (double)
    """
    sformat = ">ddd"
    size = struct.calcsize(sformat)
    duration, nb_spins, tmax = struct.unpack(">ddd", fhandle.read(size))
    return duration, int(nb_spins), int(tmax) + 1


def _read_update(fhandle):
    """
    Structure of the trajfile (per Camino documentation)

    Data (one line per spin per update):
      time (double)
      spin index (int)
      spin x (double)
      spin y (double)
      spin z (double)
    """
    sformat = ">diddd"
    size = struct.calcsize(sformat)
    time, spin, x, y, z = struct.unpack(sformat, fhandle.read(size))
    return time, spin, x, y, z


def parse_trajfile(filename):
    """
    Parses a Camino trajfile, and returns a numpy array of trajectories.
    """
    trajfile = open(filename, "rb")
    duration, nb_spins, nb_steps = _read_header(trajfile)
    trajectories = [[] for i in range(int(nb_spins))]
    indices = np.zeros(nb_spins, dtype=int)
    trajectories = np.zeros((nb_spins, nb_steps, 3))
    sformat = ">diddd"
    for t, i, x, y, z in struct.iter_unpack(sformat, trajfile.read()):
        trajectories[i, indices[i]] = (x, y, z)
        indices[i] += 1
    assert(np.all(indices == nb_steps))
    return trajectories


if __name__=="__main__":
    import argparse
    description = "Generates trajectories with Camino's Monte-Carlo simulator."
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", dest="nb_walkers", type=int, default=100,
        help="Number of walkers.")
    parser.add_argument("-m", dest="nb_steps", type=int, default=1000,
        help="Number of timesteps.")
    parser.add_argument("-t", dest="duration", type=float, default=0.1,
        help="Duration of the simulation [s].") 
    parser.add_argument("-r", dest="radius", type=float, default=1e-6,
        help="Cylinders radii [m].")
    parser.add_argument("-d", dest="density", type=float, default=0.50,
        help="Cylinder density (intra-axonal volume fraction).")
    parser.add_argument("-o", dest="output", default="trajectories.traj",
        help="Ouput filename (either Camino .traj or numpy .npy)")
    args = parser.parse_args()

    print("Preparing and running simulation with Camino.")
    _, filename = mkstemp(suffix=".traj")
    substrate = CylindersLattice(args.radius, density=args.density)
    simulation = CaminoSimulation(substrate, filename, 
      nb_walkers=args.nb_walkers, nb_steps=args.nb_steps, 
      duration=args.duration)
    simulation.run()
    
    
    if args.output.endswith(".traj"):
        os.rename(filename, args.output)
    elif args.output.endswith(".npy"):
        print("Parsing and converting trajfile to npy.")
        trajectories = parse_trajfile(filename)
        np.save(args.output, trajectories)
