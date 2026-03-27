"""
Computes electrostatic potentials at three sample points using Green's
functions obtained in Task 2/3.

The potential is evaluated via superposition of boundary and charge
contributions for all required boundary condition and charge distribution
combinations. Monte Carlo uncertainties are propagated from the Green's
functions to the final results.

Copyright (c) 2026 Jack MacQuarrie

This code is released under the MIT License. See the LICENSE file in the
repository for details.

Python Version: 3.9.21
"""

import os
import sys
import numpy as np
from dataclasses import dataclass

sys.path.append(os.path.abspath("../task2"))
from green_function import GreenFunctionMC


# ---------------
#  Configuration
# ---------------

N = 101
LENGTH = 1.0
SEED = 12345
N_WALKERS = 1_000_000

POINTS = {
    "centre": {"name": "Centre", "xy": (0.50, 0.50)},
    "corner": {"name": "Near corner", "xy": (0.02, 0.02)},
    "midface": {"name": "Mid-face", "xy": (0.02, 0.50)},
}


# -----------------------------
#  Boundary condition builders
# -----------------------------

def bc_all_plus100(n: int) -> np.ndarray:
    """All four edges at +100 V.

    Args:
        n: Grid size.

    Returns:
        N x N array with edge values set to +100 V.
    """
    phi = np.zeros((n, n), dtype=np.float64)
    phi[0, :] = 100.0    # left edge
    phi[-1, :] = 100.0   # right edge
    phi[:, 0] = 100.0    # bottom edge
    phi[:, -1] = 100.0   # top edge
    return phi


def bc_tb_plus_lr_minus(n: int) -> np.ndarray:
    """Top and bottom at +100 V, left and right at -100 V.

    Args:
        n: Grid size.

    Returns:
        N x N boundary array.
    """
    phi = np.zeros((n, n), dtype=np.float64)
    phi[0, :] = -100.0   # left edge
    phi[-1, :] = -100.0  # right edge
    phi[:, 0] = 100.0    # bottom edge
    phi[:, -1] = 100.0   # top edge
    return phi


def bc_mixed(n: int) -> np.ndarray:
    """Top +200 V, left +200 V, bottom 0 V, right -400 V.

    Args:
        n: Grid size.

    Returns:
        N x N boundary array.
    """
    phi = np.zeros((n, n), dtype=np.float64)
    phi[0, :] = 200.0    # left edge
    phi[-1, :] = -400.0  # right edge
    phi[:, 0] = 0.0      # bottom edge
    phi[:, -1] = 200.0   # top edge
    return phi


# ------------------------------
#  Charge distribution builders
# ------------------------------

def charge_uniform(n: int, length: float) -> np.ndarray:
    """10 C total charge spread uniformly over the whole grid.

    The total charge is 10 C so the density is 10 / (length^2) C/m^2.

    Args:
        n: Grid size.
        length: Physical side length in metres.

    Returns:
        N x N charge density array in C/m^2.
    """
    density = 10.0 / (length ** 2)
    return np.full((n, n), density, dtype=np.float64)


def charge_gradient(n: int, length: float) -> np.ndarray:
    """Linear charge gradient: 1 C/m^2 at the top edge, 0 at the bottom.

    The density varies linearly from f = 0 at j = 0 (y = 0, bottom) to
    f = 1 C/m^2 at j = N - 1 (y = length, top).

    Args:
        n: Grid size.
        length: Physical side length in metres.

    Returns:
        N x N charge density array in C/m^2.
    """
    y = np.linspace(0.0, 1.0, n, dtype=np.float64)
    f = np.zeros((n, n), dtype=np.float64)
    f[:, :] = y[np.newaxis, :]
    return f


def charge_exponential(n: int, length: float) -> np.ndarray:
    """Exponentially decaying charge centred on the grid: exp(-10 |r|) C/m^2.

    |r| is the Euclidean distance in metres from the centre of the grid.

    Args:
        n: Grid size.
        length: Physical side length in metres.

    Returns:
        N x N charge density array in C/m^2.
    """
    x = np.linspace(0.0, length, n, dtype=np.float64)
    y = np.linspace(0.0, length, n, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    cx = cy = length / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.exp(-10.0 * r)


# -----------------------
#  Combination catalogue
# -----------------------

def build_combinations(n: int, length: float) -> list:
    """Return the ordered list of boundary/charge combinations.

    The ordering follows the assignment sheet: first the three BC sets with
    zero charge, then the three BC sets repeated for each of the three charge
    distributions.

    Args:
        n: Grid size.
        length: Physical side length in metres.

    Returns:
        List of dicts, each with keys 'bc_label', 'charge_label',
        'bc' (N x N array), 'charge' (N x N array).
    """
    bcs = [
        ("All edges +100 V", bc_all_plus100(n)),
        ("Top/bot +100 V, left/right -100 V", bc_tb_plus_lr_minus(n)),
        ("Top/left +200 V, bot 0 V, right -400 V", bc_mixed(n)),
    ]

    zero = np.zeros((n, n), dtype=np.float64)
    charges = [
        ("f = 0", zero),
        ("Uniform 10 C", charge_uniform(n, length)),
        ("Linear gradient", charge_gradient(n, length)),
        ("Exponential exp(-10|r|)", charge_exponential(n, length)),
    ]

    combos = []

    for bc_label, bc in bcs:
        combos.append({
            "bc_label": bc_label,
            "charge_label": "f = 0",
            "bc": bc,
            "charge": zero,
        })

    for charge_label, charge in charges[1:]:
        for bc_label, bc in bcs:
            combos.append({
                "bc_label": bc_label,
                "charge_label": charge_label,
                "bc": bc,
                "charge": charge,
            })

    return combos


# ----------------------
#  Potential evaluation
# ----------------------

@dataclass
class PotentialResult:
    """Stores the potential and its estimated error at a single point."""

    point_key: str
    bc_label: str
    charge_label: str
    phi: float
    phi_err: float


def evaluate_all(
    solver: GreenFunctionMC,
    results: dict,
    combos: list,
) -> list:
    """Evaluate potentials for every point / combination pair.

    Args:
        solver: GreenFunctionMC instance.
        results: Dict keyed by point key with G_L, G_L_err, G_C, G_C_err.
        combos: List of combination dicts from build_combinations.

    Returns:
        List of PotentialResult objects.
    """
    all_results = []

    for combo in combos:
        for key in POINTS:
            res = results[key]
            phi, phi_err = solver.potential_from_green(
                res["G_L"],
                res["G_C"],
                combo["bc"],
                combo["charge"],
                res["G_L_err"],
                res["G_C_err"],
            )
            all_results.append(
                PotentialResult(
                    point_key=key,
                    bc_label=combo["bc_label"],
                    charge_label=combo["charge_label"],
                    phi=phi,
                    phi_err=phi_err,
                )
            )

    return all_results


# ----------------
#  Output: tables
# ----------------

def print_results_table(all_results: list) -> None:
    """Print a formatted table of all Task 4 potentials to stdout."""

    combos_seen = []
    combo_map = {}
    for result in all_results:
        key = (result.bc_label, result.charge_label)
        if key not in combo_map:
            combo_map[key] = []
            combos_seen.append(key)
        combo_map[key].append(result)

    bc_width = 38
    charge_width = 24
    value_width = 16
    error_width = 16

    gap = "     "         # wider gap for numbers
    header_gap = "   "    # smaller gap for headers

    point_headers = {
        "centre": "Centre (0.50,0.50)",
        "corner": "Corner (0.02,0.02)",
        "midface": "Mid-face (0.02,0.50)",
    }

    header = (
        f"{'BC':<{bc_width}} "
        f"{'Charge':<{charge_width}} "
        f"{point_headers['centre']:>{value_width}} "
        f"{point_headers['corner']:>{value_width}} "
        f"{point_headers['midface']:>{value_width}}"
        f"{header_gap}"
        f"{'sigma Centre':>{error_width}} "
        f"{'sigma Corner':>{error_width}} "
        f"{'sigma Mid-face':>{error_width}}"
    )

    sep = "=" * len(header)

    print()
    print(sep)
    print("Task 4 - Potentials from Green's function")
    print(sep)
    print(header)
    print("-" * len(header))

    prev_bc = None
    for bc_label, charge_label in combos_seen:
        if prev_bc is not None and bc_label != prev_bc:
            print()
        prev_bc = bc_label

        rows = combo_map[(bc_label, charge_label)]
        row_vals = {row.point_key: row for row in rows}

        line = (
            f"{bc_label:<{bc_width}} "
            f"{charge_label:<{charge_width}} "
            f"{row_vals['centre'].phi:>{value_width}.4f} "
            f"{row_vals['corner'].phi:>{value_width}.4f} "
            f"{row_vals['midface'].phi:>{value_width}.4f}"
            f"{gap}"
            f"{row_vals['centre'].phi_err:>{error_width}.2e} "
            f"{row_vals['corner'].phi_err:>{error_width}.2e} "
            f"{row_vals['midface'].phi_err:>{error_width}.2e}"
        )
        print(line)

    print(sep)
    print("Potentials in Volts. sigma denotes the one-sigma Monte Carlo uncertainty.")
    print(sep)

def save_results_csv(all_results: list, filename: str = "task4_results.csv") -> None:
    """Save all Task 4 results to a CSV file for Task 5 comparison.

    Args:
        all_results: List of PotentialResult objects.
        filename: Output filename.
    """
    path = os.path.join("data", filename)
    with open(path, "w", encoding="utf-8") as file_handle:
        file_handle.write("point_key,bc_label,charge_label,phi_V,phi_err_V\n")
        for result in all_results:
            file_handle.write(
                f"{result.point_key},"
                f'"{result.bc_label}",'
                f'"{result.charge_label}",'
                f"{result.phi:.8f},"
                f"{result.phi_err:.8e}\n"
            )
    print(f"Saved {path}")


# --------------
#  Data loading
# --------------

def load_green_functions(key: str) -> dict:
    """Load cached Green's function arrays for one starting point.

    Args:
        key: Point key string matching the filenames saved in Task 3.

    Returns:
        Dict with G_L, G_L_err, G_C, G_C_err arrays.

    Raises:
        FileNotFoundError: If the Task 3 cache files are missing.
    """
    data_dir = os.path.join("..", "task3", "data")
    files = {
        "G_L": f"G_laplace_{key}.npy",
        "G_L_err": f"G_laplace_err_{key}.npy",
        "G_C": f"G_charge_{key}.npy",
        "G_C_err": f"G_charge_err_{key}.npy",
    }

    loaded = {}
    for field_name, filename in files.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Task 3 cache file not found: {path}\n"
                "Run task3.py first to generate the Green's functions."
            )
        loaded[field_name] = np.load(path)

    return loaded


# ------
#  Main
# ------

def main() -> None:
    """Run the full Task 4 workflow."""
    os.makedirs("data", exist_ok=True)

    solver = GreenFunctionMC(
        grid_size=N,
        length=LENGTH,
        n_walkers=N_WALKERS,
        seed=SEED,
    )

    print("\nTask 4 - Potential evaluation from Green's functions")
    print(f"Grid: {N}x{N},  h = {solver.grid_spacing * 100:.1f} cm")

    print("\nLoading Green's functions from Task 3 cache...")
    green_data = {}
    for key in POINTS:
        green_data[key] = load_green_functions(key)
        print(f"  Loaded {POINTS[key]['name']}")

    combos = build_combinations(N, LENGTH)
    print(f"\nBuilt {len(combos)} boundary condition / charge combinations.")

    print("Evaluating potentials...")
    all_results = evaluate_all(solver, green_data, combos)

    print_results_table(all_results)
    save_results_csv(all_results)

    print("\nCheck (all edges +100 V, f = 0):")
    for result in all_results:
        if result.bc_label == "All edges +100 V" and result.charge_label == "f = 0":
            print(
                f"  {POINTS[result.point_key]['name']:15s}:  "
                f"phi = {result.phi:.6f} V"
            )


if __name__ == "__main__":
    main()
