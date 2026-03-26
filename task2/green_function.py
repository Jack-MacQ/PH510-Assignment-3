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

    # ------------------------------------------------------------------
    # Grid / boundary helpers
    # ------------------------------------------------------------------

    def coord_to_index(self, x_metres: float) -> int:
        """
        Convert a physical coordinate to the nearest grid index.

        Parameters
        ----------
        x_metres : float
            Position in metres.

        Returns
        -------
        int
            Closest grid index.
        """
        return int(round(x_metres / self._h))

    def _boundary_to_linear(self, i: int, j: int) -> int:
        """
        Map a boundary grid point (i, j) to a 1D index.

        The boundary is traversed once counter-clockwise without
        double-counting the corners. This allows the boundary
        probabilities to be stored in a flat array.

        Returns
        -------
        int
            Index in the range [0, 4*(N-1)).
        """
        n = self.grid_size

        if j == 0:                  # bottom edge
            return i
        if i == n - 1:             # right edge
            return (n - 1) + j
        if j == n - 1:             # top edge (right to left)
            return 2 * (n - 1) + (n - 1 - i)
        # left edge (top to bottom)
        return 3 * (n - 1) + (n - 1 - j)

    def linear_to_boundary(self, idx: int) -> Tuple[int, int]:
        """
        Inverse mapping from linear boundary index to (i, j).
        """
        n = self.grid_size

        if idx < n - 1:
            return idx, 0
        if idx < 2 * (n - 1):
            return n - 1, idx - (n - 1)
        if idx < 3 * (n - 1):
            return n - 1 - (idx - 2 * (n - 1)), n - 1
        return 0, n - 1 - (idx - 3 * (n - 1))

    # ------------------------------------------------------------------
    # Single walk
    # ------------------------------------------------------------------

    def _single_walk(
        self,
        start_i: int,
        start_j: int,
        rng: np.random.Generator,
    ) -> Tuple[int, np.ndarray]:
        """
        Perform a single random walk until the boundary is reached.

        The walker moves to one of its four nearest neighbours with equal
        probability. The number of visits to each grid point is recorded.

        Returns
        -------
        boundary_index : int
            Linear index of the boundary site where the walker exits.
        visit_counts : ndarray
            Number of visits to each grid point during the walk.
        """
        n = self.grid_size

        # Step directions: right, left, up, down
        di = np.array([1, -1, 0, 0], dtype=np.int32)
        dj = np.array([0, 0, 1, -1], dtype=np.int32)

        visit_counts = np.zeros((n, n), dtype=np.int32)
        i, j = start_i, start_j

        # Generate steps in batches to reduce Python overhead
        batch = 2048

        while True:
            directions = rng.integers(0, 4, size=batch)

            for d in directions:
                visit_counts[i, j] += 1

                # Move walker
                i += di[d]
                j += dj[d]

                # Stop once boundary is reached
                if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                    return self._boundary_to_linear(i, j), visit_counts
