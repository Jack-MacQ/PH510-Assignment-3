"""
Numerical solution of the two-dimensional Poisson equation on a square domain
using successive over-relaxation (SOR).

The solver works on a uniform N x N grid with Dirichlet boundary conditions,
so the potential is fixed on all four edges of the domain. The source term can
either be zero, giving Laplace's equation, or non-zero for the Poisson case.

This implementation was written as a deterministic reference solver for comparison with the random-walk method.

Copyright (c) 2026 Jack MacQuarrie

This code is released under the MIT License. See the LICENSE file in the
repository for details.

Python Version: 3.9.21
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PoissonSOR:
    """
    Solve the 2D Poisson equation on a square grid using SOR.

    Parameters
    ----------
    grid_size :
        Number of grid points along each side of the square domain.
    length :
        Physical side length of the domain in metres.
    tolerance :
        Convergence tolerance based on the maximum absolute update
        in a full iteration sweep.
    max_iterations :
        Maximum number of SOR sweeps before stopping.
    omega :
        Relaxation parameter. If not provided, a standard near-optimal
        value for a square grid is used.

    Notes
    -----
    The potential is stored as phi[i, j], where i corresponds to the x
    direction and j corresponds to the y direction. With this convention:

    - phi[0, :]   is the left boundary
    - phi[-1, :]  is the right boundary
    - phi[:, 0]   is the bottom boundary
    - phi[:, -1]  is the top boundary
    """

    grid_size: int = 101
    length: float = 1.0
    tolerance: float = 1e-8
    max_iterations: int = 100_000
    omega: Optional[float] = None

    _phi: np.ndarray = field(init=False, repr=False)
    _f: np.ndarray = field(init=False, repr=False)
    _h: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Set up the grid spacing and initialise the potential and source arrays."""
        if self.grid_size < 3:
            raise ValueError("grid_size must be at least 3")

        self._h = self.length / (self.grid_size - 1)
        self._phi = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
        self._f = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)

        # Use a standard near-optimal SOR parameter for a square grid (unless it is provided specifically)
        if self.omega is None:
            self.omega = 2.0 / (1.0 + math.sin(math.pi / self.grid_size))

    def set_boundary(
        self,
        *,
        top: float = 0.0,
        bottom: float = 0.0,
        left: float = 0.0,
        right: float = 0.0,
    ) -> None:
        """
        Set constant Dirichlet boundary conditions on the four edges.

        Parameters
        ----------
        top :
            Potential on the top edge, y = L.
        bottom :
            Potential on the bottom edge, y = 0.
        left :
            Potential on the left edge, x = 0.
        right :
            Potential on the right edge, x = L.
        """
        self._phi[0, :] = left
        self._phi[-1, :] = right
        self._phi[:, 0] = bottom
        self._phi[:, -1] = top

        # Each corner belongs to two boundaries so using the average is an
        # easy way to assign the corner values.
        self._phi[0, 0] = 0.5 * (left + bottom)
        self._phi[0, -1] = 0.5 * (left + top)
        self._phi[-1, 0] = 0.5 * (right + bottom)
        self._phi[-1, -1] = 0.5 * (right + top)

    def set_boundary_array(self, boundary_phi: np.ndarray) -> None:
        """
        Set the boundary values from a full grid array.

        Only the edge values are used; interior values in the supplied array
        are ignored.

        Parameters
        ----------
        boundary_phi :
            Array whose outer edge defines the boundary conditions.
        """
        expected_shape = (self.grid_size, self.grid_size)
        if boundary_phi.shape != expected_shape:
            raise ValueError(
                f"boundary_phi must have shape {expected_shape}, "
                f"got {boundary_phi.shape}"
            )

        self._phi[0, :] = boundary_phi[0, :]
        self._phi[-1, :] = boundary_phi[-1, :]
        self._phi[:, 0] = boundary_phi[:, 0]
        self._phi[:, -1] = boundary_phi[:, -1]

    def set_charge(self, f_array: np.ndarray) -> None:
        """
        Set the source term on the grid.

        Parameters
        ----------
        f_array :
            N x N array containing the source term values.
        """
        expected_shape = (self.grid_size, self.grid_size)
        if f_array.shape != expected_shape:
            raise ValueError(
                f"f_array must have shape {expected_shape}, got {f_array.shape}"
            )

        self._f = f_array.copy()

    def solve(self, verbose: bool = True) -> np.ndarray:
        """
        Solve the Poisson equation using successive over-relaxation.

        The method performs repeated Gauss-Seidel sweeps with an SOR update
        until the maximum absolute change in the grid falls below the chosen
        tolerance, or until the maximum number of iterations is reached.

        Parameters
        ----------
        verbose :
            If True, print convergence information.

        Returns
        -------
        numpy.ndarray
            Converged potential array.
        """
        phi = self._phi.copy()
        f = self._f
        h2 = self._h * self._h
        omega = float(self.omega)

        max_change = float("inf")

        for iteration in range(1, self.max_iterations + 1):
            max_change = 0.0

            # Update only the interior points. Boundary values remain fixed
            # throughout the iteration.
            for i in range(1, self.grid_size - 1):
                for j in range(1, self.grid_size - 1):
                    old_value = phi[i, j]

                    # Gauss-Seidel update for the discrete Poisson equation.
                    gs_value = 0.25 * (
                        phi[i + 1, j]
                        + phi[i - 1, j]
                        + phi[i, j + 1]
                        + phi[i, j - 1]
                        + h2 * f[i, j]
                    )

                    # SOR improves convergence by taking a weighted step between
                    # the previous value and the Gauss-Seidel update.
                    new_value = (1.0 - omega) * old_value + omega * gs_value
                    phi[i, j] = new_value

                    change = abs(new_value - old_value)
                    if change > max_change:
                        max_change = change

            if max_change < self.tolerance:
                if verbose:
                    print(f"Converged after {iteration} iterations")
                    print(f"Final max absolute change: {max_change:.3e}")
                self._phi = phi
                return phi.copy()

        if verbose:
            print("WARNING: Solver did not reach the requested tolerance")
            print(f"Final max absolute change: {max_change:.3e}")

        self._phi = phi
        return phi.copy()

    @property
    def phi(self) -> np.ndarray:
        """Return a copy of the current potential array."""
        return self._phi.copy()

    @property
    def grid_spacing(self) -> float:
        """Return the uniform grid spacing."""
        return self._h

    @property
    def x_coords(self) -> np.ndarray:
        """Return the x coordinates of the grid points."""
        return np.linspace(0.0, self.length, self.grid_size)

    @property
    def y_coords(self) -> np.ndarray:
        """Return the y coordinates of the grid points."""
        return np.linspace(0.0, self.length, self.grid_size)

    def potential_at(self, x: float, y: float) -> float:
        """
        Return the potential at the nearest grid point to the coordinate (x, y).

        Parameters
        ----------
        x :
            x coordinate in metres.
        y :
            y coordinate in metres.
        """
        i = int(round(x / self._h))
        j = int(round(y / self._h))

        # Clip indices so that coordinates slightly outside the domain are
        # mapped to the nearest valid grid point.
        i = max(0, min(i, self.grid_size - 1))
        j = max(0, min(j, self.grid_size - 1))

        return float(self._phi[i, j])


def uniform_charge(grid_size: int, value: float) -> np.ndarray:
    """Return a source term with the same value at every grid point."""
    return np.full((grid_size, grid_size), value, dtype=np.float64)


def linear_y_charge(grid_size: int, y_min: float, y_max: float) -> np.ndarray:
    """
    Return a source term that varies linearly in the y direction.

    The value is y_min at y = 0 and y_max at y = L.
    """
    y_values = np.linspace(y_min, y_max, grid_size, dtype=np.float64)
    return np.tile(y_values, (grid_size, 1))


def exponential_central_charge(grid_size: int, length: float) -> np.ndarray:
    """
    Return a centrally peaked exponential source term of the form exp(-10 r),
    where r is the distance from the centre of the square.
    """
    x = np.linspace(0.0, length, grid_size)
    y = np.linspace(0.0, length, grid_size)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    x0 = 0.5 * length
    y0 = 0.5 * length
    r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)

    return np.exp(-10.0 * r)


def plot_potential(
    phi: np.ndarray,
    length: float = 1.0,
    title: str = "Potential"
) -> None:
    """
    Plot the potential as a filled contour map over the square domain.
    """
    n = phi.shape[0]
    x = np.linspace(0.0, length, n)
    y = np.linspace(0.0, length, n)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(xx, yy, phi, levels=50)
    cbar = fig.colorbar(contour, ax=ax, label="Potential (V)")
    cbar.set_ticks(np.linspace(-100, 100, 9))

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


def print_assignment_points(solver: PoissonSOR) -> None:
    """
    Print the potential at the three sample points used later in the assignment.
    """
    points = {
        "Centre (0.50, 0.50)": (0.50, 0.50),
        "Near corner (0.02, 0.02)": (0.02, 0.02),
        "Mid-face (0.02, 0.50)": (0.02, 0.50),
    }

    print("\nSelected potentials:")
    print(f"{'Point':<28} {'phi (V)':>12}")
    print("-" * 42)

    for label, (x, y) in points.items():
        value = solver.potential_at(x, y)
        print(f"{label:<28} {value:>12.6f}")


def main() -> None:
    """Run a simple example with fixed boundary values and zero source term."""
    solver = PoissonSOR(
        grid_size=101,
        length=1.0,
        tolerance=1e-8,
    )

    # Example Laplace case: top and bottom held at +100 V, left and right at -100 V.
    solver.set_boundary(
        top=100.0,
        bottom=100.0,
        left=-100.0,
        right=-100.0,
    )

    # Zero source term gives Laplace's equation rather than the full Poisson case.
    solver.set_charge(uniform_charge(solver.grid_size, 0.0))

    phi = solver.solve(verbose=True)

    print_assignment_points(solver)
    plot_potential(phi, length=solver.length, title="SOR solution for Task 1")


if __name__ == "__main__":
    main()
