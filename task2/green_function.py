"""
Parallel Monte Carlo Green's function solver for the 2D Poisson equation.

Computes both the Laplace (boundary) and charge Green's functions at a
specified interior grid point, with standard error estimates.

Copyright (c) 2026 Jack MacQuarrie

This code is released under the MIT License. See the LICENSE file in the
repository for details.

Python Version: 3.9.21
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from mpi4py import MPI


@dataclass
class GreenFunctionMC:
    """
    Monte Carlo estimator for the Green's functions of the 2D Poisson equation.

    A set of random walkers is launched from a chosen interior grid point.
    Each walker performs an unbiased random walk until it reaches the boundary.
    By collecting statistics over many walks, both the boundary and
    charge Green's functions can be estimated as well as their uncertainties.

    Parameters
    ----------
    grid_size : int
        Number of grid points along each side of the square domain.
    length : float
        Physical side length of the domain (metres).
    n_walkers : int
        Total number of walkers used across all MPI ranks.
    seed : int
        Base random seed. Each rank generates an independent stream.
    """

    grid_size: int = 100
    length: float = 1.0
    n_walkers: int = 100_000
    seed: int = 42

    _h: float = field(init=False, repr=False)
    _comm: object = field(init=False, repr=False)
    _rank: int = field(init=False, repr=False)
    _size: int = field(init=False, repr=False)

    def __post_init__(self):
        """Initialise grid spacing and MPI communicator."""
        self._h = self.length / (self.grid_size - 1)

        # Standard MPI setup
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()
