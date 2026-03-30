"""
Parallel Monte Carlo Green's function solver for the 2D Poisson equation.

Computes both the Laplace (boundary) and charge Green's functions at a
specified interior grid point, with standard error estimates.

Copyright (c) 2026 Jack MacQuarrie

This code is released under the MIT License. See the LICENSE file in the
repository for details.

Python Version: 3.9.21
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

# pylint: disable=no-name-in-module
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

    _comm: object = field(init=False, repr=False)
    _rank: int = field(init=False, repr=False)
    _size: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialise the MPI communicator."""
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
        return int(round(x_metres / self.grid_spacing))

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

        if j == 0:  # bottom edge
            return i
        if i == n - 1:  # right edge
            return (n - 1) + j
        if j == n - 1:  # top edge (right to left)
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

        di = np.array([1, -1, 0, 0], dtype=np.int32)
        dj = np.array([0, 0, 1, -1], dtype=np.int32)

        visit_counts = np.zeros((n, n), dtype=np.int32)
        i, j = start_i, start_j
        batch = 2048

        while True:
            directions = rng.integers(0, 4, size=batch)

            for direction in directions:
                visit_counts[i, j] += 1
                i += di[direction]
                j += dj[direction]

                if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                    return self._boundary_to_linear(i, j), visit_counts

    # ------------------------------------------------------------------
    # Main Green's function computation
    # ------------------------------------------------------------------

    def _make_rng(self) -> np.random.Generator:
        """Create an independent random number generator for this MPI rank."""
        seed_sequence = np.random.SeedSequence(self.seed)
        rank_sequence = seed_sequence.spawn(self._size)[self._rank]
        return np.random.default_rng(rank_sequence)

    def _local_walker_count(self) -> int:
        """Return the number of walkers assigned to this MPI rank."""
        base, remainder = divmod(self.n_walkers, self._size)
        return base + (1 if self._rank < remainder else 0)

    def _run_local_walkers(
        self,
        start_i: int,
        start_j: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Run the walkers assigned to this MPI rank and accumulate statistics.
        """
        n = self.grid_size
        n_boundary = self.n_boundary_points
        local_n = self._local_walker_count()

        local_hits = np.zeros(n_boundary, dtype=np.int64)
        local_visit_sum = np.zeros((n, n))
        local_visit_sum2 = np.zeros((n, n))

        for _ in range(local_n):
            boundary_index, visits = self._single_walk(start_i, start_j, rng)
            local_hits[boundary_index] += 1

            visits_float = visits.astype(float)
            local_visit_sum += visits_float
            local_visit_sum2 += visits_float * visits_float

        return local_hits, local_visit_sum, local_visit_sum2, local_n

    def _reduce_statistics(
        self,
        local_hits: np.ndarray,
        local_visit_sum: np.ndarray,
        local_visit_sum2: np.ndarray,
        local_n: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Reduce local statistics to rank 0.
        """
        global_hits = np.zeros_like(local_hits)
        global_sum = np.zeros_like(local_visit_sum)
        global_sum2 = np.zeros_like(local_visit_sum2)

        self._comm.Reduce(local_hits, global_hits, op=MPI.SUM, root=0)
        self._comm.Reduce(local_visit_sum, global_sum, op=MPI.SUM, root=0)
        self._comm.Reduce(local_visit_sum2, global_sum2, op=MPI.SUM, root=0)
        n_total = self._comm.reduce(local_n, op=MPI.SUM, root=0)

        return global_hits, global_sum, global_sum2, n_total

    def _finalise_green_functions(
        self,
        global_hits: np.ndarray,
        global_sum: np.ndarray,
        global_sum2: np.ndarray,
        n_total: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct Green's functions and standard errors on rank 0.
        """
        n_total_float = float(n_total)

        g_laplace = global_hits / n_total_float
        g_laplace_err = np.sqrt(
            np.maximum(g_laplace * (1.0 - g_laplace), 0.0) / n_total_float
        )

        mean_visits = global_sum / n_total_float
        g_charge = (self.grid_spacing ** 2) * mean_visits

        var = (global_sum2 / n_total_float) - mean_visits**2
        var *= n_total_float / (n_total_float - 1.0)
        g_charge_err = (
            (self.grid_spacing ** 2)
            * np.sqrt(np.maximum(var, 0.0) / n_total_float)
        )

        return g_laplace, g_laplace_err, g_charge, g_charge_err

    def compute_green_function(
        self,
        start_i: int,
        start_j: int,
    ) -> Tuple[object, object, object, object]:
        """
        Estimate the Green's functions for a given starting point.

        Walkers are distributed across MPI ranks. Each rank performs its
        subset of walks, and the results are combined on rank 0.

        Returns
        -------
        tuple
            On rank 0:
                (g_laplace, g_laplace_err, g_charge, g_charge_err)
            On other ranks:
                (None, None, None, None)
        """
        n = self.grid_size

        if not (0 < start_i < n - 1 and 0 < start_j < n - 1):
            raise ValueError("Starting point must be inside the domain")

        rng = self._make_rng()
        local_stats = self._run_local_walkers(start_i, start_j, rng)
        global_stats = self._reduce_statistics(*local_stats)

        if self._rank != 0:
            return None, None, None, None

        return self._finalise_green_functions(*global_stats)

    # ------------------------------------------------------------------
    # Potential reconstruction
    # ------------------------------------------------------------------

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def potential_from_green(
        self,
        g_laplace,
        g_charge,
        boundary_phi,
        charge_density,
        g_laplace_err=None,
        g_charge_err=None,
    ):
        """
        Reconstruct the potential at the source point using Green's functions.

        The potential is obtained as the sum of the boundary contribution
        and the charge contribution.
        """
        phi = 0.0
        phi_err_sq = 0.0

        for idx, boundary_weight in enumerate(g_laplace):
            i, j = self.linear_to_boundary(idx)
            phi += boundary_weight * boundary_phi[i, j]

            if g_laplace_err is not None:
                phi_err_sq += (g_laplace_err[idx] * boundary_phi[i, j]) ** 2

        phi += np.sum(g_charge * charge_density)

        if g_charge_err is not None:
            phi_err_sq += np.sum((g_charge_err * charge_density) ** 2)

        return phi, np.sqrt(phi_err_sq)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def grid_spacing(self) -> float:
        """Grid spacing h."""
        return self.length / (self.grid_size - 1)

    @property
    def n_boundary_points(self) -> int:
        """Number of boundary points."""
        return 4 * (self.grid_size - 1)

    @property
    def rank(self) -> int:
        """MPI rank."""
        return self._rank
