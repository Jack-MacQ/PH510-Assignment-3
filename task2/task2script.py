"""
Driver script for evaluating Monte Carlo Green's functions at one of the
sample points used in the assignment.

This script calls the parallel random-walk solver in green_function.py and
evaluates the boundary and charge Green's functions at the required points,
and saves the resulting arrays and plots for later analysis.

Copyright (c) 2026 Jack MacQuarrie

This code is released under the MIT License. See the LICENSE file in the
repository for details.

Python Version: 3.9.21
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI  # pylint: disable=no-name-in-module

from green_function import GreenFunctionMC


def make_boundary_array(
    n: int,
    top: float,
    bottom: float,
    left: float,
    right: float,
) -> np.ndarray:
    """
    Construct an NxN array with constant boundary values on each edge.
    """
    phi = np.zeros((n, n), dtype=np.float64)
    phi[:, -1] = top
    phi[:, 0] = bottom
    phi[0, :] = left
    phi[-1, :] = right
    return phi


# pylint: disable=too-many-locals
def plot_green_function_laplace(
    g_laplace: np.ndarray,
    g_laplace_err: np.ndarray,
    solver: GreenFunctionMC,
    label: str,
) -> None:
    """
    Plot the boundary Green's function and its standard error.

    The 1D boundary array is mapped back onto the square grid so that the
    spatial distribution of the boundary-hitting probabilities can be shown
    more clearly.
    """
    n = solver.grid_size
    n_b = solver.n_boundary_points
    phi_map = np.zeros((n, n))
    err_map = np.zeros((n, n))

    # Put the boundary values back onto a 2D grid for plotting
    for b_idx in range(n_b):
        bi, bj = solver.linear_to_boundary(b_idx)
        phi_map[bi, bj] = g_laplace[b_idx]
        err_map[bi, bj] = g_laplace_err[b_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, title in zip(
        axes,
        [phi_map, err_map],
        [f"G_laplace - {label}", f"G_laplace standard error - {label}"],
    ):
        im = ax.imshow(data.T, origin="lower", cmap="viridis")
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("i (grid index)")
        ax.set_ylabel("j (grid index)")

    plt.tight_layout()
    filename = f"plots/green_laplace_{label.replace(' ', '_').replace(',', '')}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")


def plot_green_function_charge(
    g_charge: np.ndarray,
    g_charge_err: np.ndarray,
    label: str,
) -> None:
    """
    Plot the charge Green's function and its standard error as heatmaps.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, title in zip(
        axes,
        [g_charge, g_charge_err],
        [f"G_charge - {label}", f"G_charge standard error - {label}"],
    ):
        im = ax.imshow(data.T, origin="lower", cmap="plasma")
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("i (grid index)")
        ax.set_ylabel("j (grid index)")

    plt.tight_layout()
    filename = f"plots/green_charge_{label.replace(' ', '_').replace(',', '')}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")


# pylint: disable=too-many-locals
def main():
    """
    Evaluate and save Green's functions at the three points specified for
    the later stages of the assignment.
    """
    os.makedirs("plots", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Grid and Monte Carlo setup
    n = 101
    length = 1.0
    n_walkers = 500_000

    solver = GreenFunctionMC(
        grid_size=n,
        length=length,
        n_walkers=n_walkers,
        seed=12345,
    )
    h = solver.grid_spacing

    if rank == 0:
        print(
            f"Grid: {n}x{n}, h = {h * 100:.1f} cm, "
            f"{n_walkers} walkers on {size} ranks"
        )
        print("-" * 60)

    # The centre point
    centre_point = {
        "centre_50_50": (0.50, 0.50)
    }

    results = {}

    for label, (x, y) in centre_point.items():
        si = solver.coord_to_index(x)
        sj = solver.coord_to_index(y)

        if rank == 0:
            print(
                f"\nComputing Green's function at ({x:.2f} m, {y:.2f} m) "
                f"-> grid ({si}, {sj})"
            )

        t0 = time.perf_counter()
        g_laplace, g_laplace_err, g_charge, g_charge_err = solver.compute_green_function(si, sj)
        t1 = time.perf_counter()

        if rank == 0:
            wall_time = t1 - t0

            # Basic checks on the output
            print(f"  Wall time: {wall_time:.2f} s")
            print(f"  G_laplace sum: {g_laplace.sum():.6f}  (expect 1.0)")
            print(f"  G_laplace max standard error: {g_laplace_err.max():.2e}")
            print(f"  G_charge max value: {g_charge.max():.4e}")
            print(f"  G_charge max standard error: {g_charge_err.max():.2e}")

            # Save arrays for later use in potential reconstruction
            np.save(f"data/G_laplace_{label}.npy", g_laplace)
            np.save(f"data/G_laplace_err_{label}.npy", g_laplace_err)
            np.save(f"data/G_charge_{label}.npy", g_charge)
            np.save(f"data/G_charge_err_{label}.npy", g_charge_err)

            # Save figures for inspection
            plot_green_function_laplace(g_laplace, g_laplace_err, solver, label)
            plot_green_function_charge(g_charge, g_charge_err, label)

            results[label] = {
                "G_L": g_laplace,
                "G_L_err": g_laplace_err,
                "G_C": g_charge,
                "G_C_err": g_charge_err,
                "time": wall_time,
            }

    # Simple consistency check:
    # with boundary phi = 1 everywhere and zero charge, the result should be 1
    if rank == 0:
        print("\n--- Check (uniform 1 V boundary, f = 0) ---")
        n_grid = solver.grid_size
        uniform_bc = np.ones((n_grid, n_grid), dtype=np.float64)
        zero_charge = np.zeros((n_grid, n_grid), dtype=np.float64)

        for label, res in results.items():
            phi, phi_err = solver.potential_from_green(
                res["G_L"],
                res["G_C"],
                uniform_bc,
                zero_charge,
                res["G_L_err"],
                res["G_C_err"],
            )
            print(
                f"  {label}: phi = {phi:.6f} V  "
                f"(err = {phi_err:.2e})  [expect 1.0]"
            )


if __name__ == "__main__":
    main()
