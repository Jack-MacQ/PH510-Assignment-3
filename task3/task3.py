"""
Standalone Task 3 driver for evaluating and visualising Monte Carlo Green's
functions at the three required sample points.

This version computes the Green's functions directly, so Task 3 does not
depend on Task 2 having been run first, however it does use the green_function.py
file from the Task 2 directory. It saves its own cached arrays inside the local
task3/data directory, produces graphical output for both boundary and charge Green's
functions, and prints one unified summary table together with a simple consistency check.

It is written slightly differently from task 2 and includes some other plots/data
that is not necessarily requested. These were done to check results and present
the data in a different/new way.

Copyright (c) 2026 Jack MacQuarrie

This code is released under the MIT License. See the LICENSE file in the
repository for details.

Python Version: 3.9.21
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpi4py import MPI

# Import the Monte Carlo solver from the Task 2 directory.
sys.path.append(os.path.abspath("../task2"))
from green_function import GreenFunctionMC


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

N = 101
LENGTH = 1.0
N_WALKERS = 1_000_000
SEED = 12345

POINTS = {
    "centre": {"name": "Centre", "xy": (0.50, 0.50)},
    "corner": {"name": "Near corner", "xy": (0.02, 0.02)},
    "midface": {"name": "Mid-face", "xy": (0.02, 0.50)},
}


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def full_point_label(key: str) -> str:
    """
    Return the point name together with its physical coordinates.
    """
    x, y = POINTS[key]["xy"]
    return f'{POINTS[key]["name"]} ({x:.2f} m, {y:.2f} m)'


def point_name(key: str) -> str:
    """
    Return the short name used in tables.
    """
    return POINTS[key]["name"]


def unfold_boundary(g_laplace: np.ndarray, n: int, length: float) -> tuple:
    """
    Convert the 1D boundary Green's function into a perimeter coordinate.

    This is used for the line plots, where the boundary data are shown as a
    function of distance around the square boundary. The ordering is assumed
    to match the ordering used internally by the solver.
    """
    h = length / (n - 1)
    total = 4 * (n - 1)

    return (
        np.arange(total, dtype=np.float64) * h,
        np.array(g_laplace[:total], copy=True),
        [0.0, (n - 1) * h, 2.0 * (n - 1) * h, 3.0 * (n - 1) * h, 4.0 * (n - 1) * h],
        ["(0,0) m", "(1,0) m", "(1,1) m", "(0,1) m", "(0,0) m"],
    )


def boundary_to_grid(
    values: np.ndarray,
    errors: np.ndarray,
    solver: GreenFunctionMC,
) -> tuple:
    """
    Place the boundary Green's function and its error back onto an N x N grid.

    Only boundary points are populated. Interior points are left as NaN so that
    they are ignored in the 2D boundary plots.
    """
    n = solver.grid_size
    val_grid = np.full((n, n), np.nan, dtype=np.float64)
    err_grid = np.full((n, n), np.nan, dtype=np.float64)

    for b_idx in range(solver.n_boundary_points):
        i, j = solver.linear_to_boundary(b_idx)
        val_grid[i, j] = values[b_idx]
        err_grid[i, j] = errors[b_idx]

    return val_grid, err_grid


def sci_tick_label_math(x: float, pos: int) -> str:
    """
    Format colour-bar tick labels in a compact scientific notation.

    The labels are written in maths text so they appear as powers of ten rather
    than standard e-notation.
    """
    if np.isclose(x, 0.0):
        return "0"

    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10.0 ** exponent)

    if np.isclose(mantissa, 1.0, rtol=1.0e-10, atol=1.0e-12):
        return rf"$10^{{{exponent:d}}}$"

    return rf"${mantissa:.1f} \times 10^{{{exponent:d}}}$"


def add_sci_colorbar(fig, mappable, ax, label: str):
    """
    Add a colour bar with scientific-notation tick labels.
    """
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        format=FuncFormatter(sci_tick_label_math),
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_label(label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.yaxis.get_offset_text().set_visible(False)
    cbar.update_ticks()
    return cbar


def plot_two_maps(
    left_data: np.ndarray,
    right_data: np.ndarray,
    key: str,
    left_title: str,
    right_title: str,
    left_cbar: str,
    right_cbar: str,
    cmap: str,
    filename: str,
) -> None:
    """
    Draw a pair of 2D maps with a consistent layout and colour-bar style.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    extent = (0.0, LENGTH, 0.0, LENGTH)
    point_text = full_point_label(key)

    for ax, data, title, cbar_label in (
        (axes[0], left_data, left_title, left_cbar),
        (axes[1], right_data, right_title, right_cbar),
    ):
        im = ax.imshow(
            data.T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            aspect="equal",
        )
        add_sci_colorbar(fig, im, ax, cbar_label)
        ax.set_xlabel("x (m)", fontsize=10)
        ax.set_ylabel("y (m)", fontsize=10)
        ax.set_title(f"{title}\n{point_text}", fontsize=10.5, pad=8)
        ax.tick_params(labelsize=9)

    plt.savefig(os.path.join("plots", filename), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_boundary_exit_maps(
    g_laplace: np.ndarray,
    g_laplace_err: np.ndarray,
    solver: GreenFunctionMC,
    key: str,
) -> None:
    """
    Plot the boundary exit probabilities and their estimated error as 2D maps.
    """
    val_grid, err_grid = boundary_to_grid(g_laplace, g_laplace_err, solver)

    plot_two_maps(
        val_grid,
        err_grid,
        key,
        "Boundary exit probability",
        "Boundary exit probability error",
        "Exit probability",
        "Standard error",
        "viridis",
        f"boundary_exit_maps_{key}.png",
    )


def plot_charge_maps(
    g_charge: np.ndarray,
    g_charge_err: np.ndarray,
    key: str,
) -> None:
    """
    Plot the charge Green's function and its estimated error as 2D maps.
    """
    plot_two_maps(
        g_charge,
        g_charge_err,
        key,
        "Charge Green's function",
        "Charge Green's function error",
        r"$G_{\mathrm{charge}}$ (m$^2$)",
        "Standard error (m$^2$)",
        "plasma",
        f"charge_maps_{key}.png",
    )


def plot_boundary_exit_distribution(
    g_laplace: np.ndarray,
    g_laplace_err: np.ndarray,
    solver: GreenFunctionMC,
    key: str,
) -> None:
    """
    Plot the boundary exit probability around the square perimeter.

    The shaded band shows the estimated one-sigma uncertainty from the Monte
    Carlo sampling.
    """
    arc, vals, ticks, tick_labels = unfold_boundary(g_laplace, solver.grid_size, LENGTH)
    _, errs, _, _ = unfold_boundary(g_laplace_err, solver.grid_size, LENGTH)

    fig, ax = plt.subplots(1, 1, figsize=(10.0, 4.8), constrained_layout=True)

    ax.fill_between(arc, vals - errs, vals + errs, alpha=0.30, label=r"$\pm 1\sigma$")
    ax.plot(arc, vals, lw=1.3, label="Exit probability")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_xlim(arc[0], arc[-1])
    ax.set_xlabel("Arc length along boundary (m)", fontsize=10)
    ax.set_ylabel("Exit probability", fontsize=10)
    ax.set_title(
        "Boundary exit probabilities\n" + full_point_label(key),
        fontsize=10.5,
        pad=8,
    )

    # Draw guide lines at the corners so the four edges are easier to identify.
    for tick in ticks[1:-1]:
        ax.axvline(tick, color="gray", lw=0.7, ls="--")

    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, loc="best")

    plt.savefig(
        os.path.join("plots", f"boundary_exit_distribution_{key}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def plot_comparison_summary(results: dict, solver: GreenFunctionMC) -> None:
    """
    Produce one summary figure comparing all three starting points.

    The top row shows the charge Green's function, and the bottom row shows
    the boundary exit probability plotted on the square boundary.
    """
    extent = (0.0, LENGTH, 0.0, LENGTH)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    fig.suptitle("Comparison of Green's functions", fontsize=12)

    for col, key in enumerate(POINTS):
        point_text = full_point_label(key)

        ax_top = axes[0, col]
        im1 = ax_top.imshow(
            results[key]["G_C"].T,
            origin="lower",
            extent=extent,
            cmap="plasma",
            aspect="equal",
        )
        add_sci_colorbar(fig, im1, ax_top, r"$G_{\mathrm{charge}}$ (m$^2$)")
        ax_top.set_title(f"Charge Green's function\n{point_text}", fontsize=9.5, pad=6)
        ax_top.set_xlabel("x (m)", fontsize=9)
        ax_top.set_ylabel("y (m)", fontsize=9)
        ax_top.tick_params(labelsize=8)

        val_grid, _ = boundary_to_grid(
            results[key]["G_L"],
            results[key]["G_L_err"],
            solver,
        )
        ax_bot = axes[1, col]
        im2 = ax_bot.imshow(
            val_grid.T,
            origin="lower",
            extent=extent,
            cmap="viridis",
            aspect="equal",
        )
        add_sci_colorbar(fig, im2, ax_bot, "Exit probability")
        ax_bot.set_title(f"Boundary exit probability\n{point_text}", fontsize=9.5, pad=6)
        ax_bot.set_xlabel("x (m)", fontsize=9)
        ax_bot.set_ylabel("y (m)", fontsize=9)
        ax_bot.tick_params(labelsize=8)

    plt.savefig(os.path.join("plots", "comparison_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------

def data_paths(key: str) -> dict:
    """
    Return the cache file paths associated with one starting point.
    """
    return {
        "G_L": os.path.join("data", f"G_laplace_{key}.npy"),
        "G_L_err": os.path.join("data", f"G_laplace_err_{key}.npy"),
        "G_C": os.path.join("data", f"G_charge_{key}.npy"),
        "G_C_err": os.path.join("data", f"G_charge_err_{key}.npy"),
    }


def load_or_compute(
    key: str,
    solver: GreenFunctionMC,
    start_i: int,
    start_j: int,
    rank: int,
    use_cache: bool = True,
) -> dict:
    """
    Load cached Green's functions if they already exist, otherwise compute them.

    Cached data are stored locally in task3/data so that the script can be
    rerun without repeating the full Monte Carlo calculation each time.
    """
    paths = data_paths(key)
    have_cache = rank == 0 and use_cache and all(os.path.exists(path) for path in paths.values())
    have_cache = MPI.COMM_WORLD.bcast(have_cache, root=0)

    if have_cache:
        if rank == 0:
            t0 = time.perf_counter()
            return {
                "G_L": np.load(paths["G_L"]),
                "G_L_err": np.load(paths["G_L_err"]),
                "G_C": np.load(paths["G_C"]),
                "G_C_err": np.load(paths["G_C_err"]),
                "time": time.perf_counter() - t0,
            }
        return {"G_L": None, "G_L_err": None, "G_C": None, "G_C_err": None, "time": 0.0}

    t0 = time.perf_counter()
    g_laplace, g_laplace_err, g_charge, g_charge_err = solver.compute_green_function(
        start_i, start_j
    )
    elapsed = time.perf_counter() - t0

    if rank == 0:
        np.save(paths["G_L"], g_laplace)
        np.save(paths["G_L_err"], g_laplace_err)
        np.save(paths["G_C"], g_charge)
        np.save(paths["G_C_err"], g_charge_err)
        return {
            "G_L": g_laplace,
            "G_L_err": g_laplace_err,
            "G_C": g_charge,
            "G_C_err": g_charge_err,
            "time": elapsed,
        }

    return {"G_L": None, "G_L_err": None, "G_C": None, "G_C_err": None, "time": 0.0}


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def print_summary_table(results: dict, solver: GreenFunctionMC) -> None:
    """
    Print a compact summary of the main Green's function outputs.
    """
    sep = "=" * 88
    print()
    print(sep)
    print("Task 3 - Unified Green's Function Summary")
    print(sep)
    print(
        "Grid: {}x{}, h = {:.1f} cm, walkers = {:,}".format(
            solver.grid_size,
            solver.grid_size,
            solver.grid_spacing * 100.0,
            solver.n_walkers,
        )
    )
    print(sep)
    print(
        "{:<22} {:>10} {:>15} {:>15} {:>15} {:>10}".format(
            "Point",
            "G_L sum",
            "G_L max sigma",
            "G_C max val",
            "G_C max sigma",
            "Time (s)",
        )
    )
    print("-" * 88)

    for key in POINTS:
        res = results[key]
        print(
            "{:<22} {:>10.6f} {:>15.3e} {:>15.4e} {:>15.3e} {:>10.1f}".format(
                point_name(key),
                res["G_L"].sum(),
                res["G_L_err"].max(),
                res["G_C"].max(),
                res["G_C_err"].max(),
                res["time"],
            )
        )

    print(sep)


def run_uniform_boundary_check(results: dict, solver: GreenFunctionMC) -> None:
    """
    Run a simple consistency check using a uniform 1 V boundary condition.

    With zero interior source, the potential should come out close to 1 V at
    each of the three sample points.
    """
    print()
    print("=" * 96)
    print("Consistency check: uniform 1 V boundary, zero charge")
    print("=" * 96)

    n = solver.grid_size
    uniform_bc = np.ones((n, n), dtype=np.float64)
    zero_charge = np.zeros((n, n), dtype=np.float64)

    print("{:<22} {:>15} {:>15}".format("Point", "phi (V)", "estimated sigma"))
    print("-" * 96)

    for key in POINTS:
        res = results[key]
        phi, phi_err = solver.potential_from_green(
            res["G_L"],
            res["G_C"],
            uniform_bc,
            zero_charge,
            res["G_L_err"],
            res["G_C_err"],
        )
        print("{:<22} {:>15.6f} {:>15.3e}".format(point_name(key), phi, phi_err))

    print("-" * 96)
    print("Expected result: phi = 1.0 V at all three points.")
    print("=" * 96)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run the full Task 3 workflow: load or compute data, make plots, and print
    a short numerical summary.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        os.makedirs("plots", exist_ok=True)
        os.makedirs("data", exist_ok=True)
    comm.Barrier()

    solver = GreenFunctionMC(
        grid_size=N,
        length=LENGTH,
        n_walkers=N_WALKERS,
        seed=SEED,
    )

    if rank == 0:
        print()
        print("Task 3 - Standalone Green's function evaluation")
        print("Grid: {}x{}, h = {:.1f} cm".format(N, N, solver.grid_spacing * 100.0))
        print("Walkers: {:,} across {} MPI ranks".format(N_WALKERS, size))
        print("=" * 72)

    results = {}

    for key, point in POINTS.items():
        x, y = point["xy"]
        results[key] = load_or_compute(
            key=key,
            solver=solver,
            start_i=solver.coord_to_index(x),
            start_j=solver.coord_to_index(y),
            rank=rank,
            use_cache=True,
        )

    if rank == 0:
        for key in POINTS:
            res = results[key]
            plot_boundary_exit_maps(res["G_L"], res["G_L_err"], solver, key)
            plot_charge_maps(res["G_C"], res["G_C_err"], key)
            plot_boundary_exit_distribution(res["G_L"], res["G_L_err"], solver, key)

        plot_comparison_summary(results, solver)
        print_summary_table(results, solver)
        run_uniform_boundary_check(results, solver)


if __name__ == "__main__":
    main()
