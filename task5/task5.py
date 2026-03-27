"""
Task 5 driver for comparing Monte Carlo Green's function solutions
(Task 4) with the deterministic Successive Over-Relaxation (SOR)
solver (Task 1).

The script:
1. Loads the computed potentials and uncertainties from task4_results.csv.
2. Recomputes the corresponding potentials using the SOR method for all
   boundary condition and charge distribution combinations.
3. Evaluates both methods at the three specified sample points.
4. Computes the difference between Monte Carlo and SOR results, and
   expresses this discrepancy in units of the Monte Carlo standard error
   (n_sigma).
5. Reports whether agreement is achieved within 2 sigma for each case.
6. Outputs a formatted comparison table and saves the results to CSV
   (and Excel if available).

Copyright (c) 2026 Jack MacQuarrie
Released under the MIT License. See LICENSE for details.

Python Version: 3.9.21
"""

import os
import sys
import csv
import numpy as np
from dataclasses import dataclass

sys.path.append(os.path.abspath("../task1"))
from task1 import PoissonSOR


# ---------------
#  Configuration
# ---------------

N = 101
LENGTH = 1.0

POINTS = {
    "centre": {"name": "Centre", "xy": (0.50, 0.50)},
    "corner": {"name": "Near corner", "xy": (0.02, 0.02)},
    "midface": {"name": "Mid-face", "xy": (0.02, 0.50)},
}

# Use a fairly tight tolerance so the deterministic values are converged
# well beyond the Monte Carlo error bars.
SOR_TOLERANCE = 1e-8


# ----------------------------------------
#  Boundary condition and charge builders
# ----------------------------------------
# These are reproduced from task4.py so that task5.py remains self-contained
# and uses exactly the same cases and labels.

def bc_all_plus100(n: int) -> np.ndarray:
    """Construct the uniform +100 V boundary condition."""
    phi = np.zeros((n, n), dtype=np.float64)
    phi[0, :] = 100.0
    phi[-1, :] = 100.0
    phi[:, 0] = 100.0
    phi[:, -1] = 100.0
    return phi


def bc_tb_plus_lr_minus(n: int) -> np.ndarray:
    """Construct the case with top/bottom at +100 V and left/right at -100 V."""
    phi = np.zeros((n, n), dtype=np.float64)
    phi[0, :] = -100.0
    phi[-1, :] = -100.0
    phi[:, 0] = 100.0
    phi[:, -1] = 100.0
    return phi


def bc_mixed(n: int) -> np.ndarray:
    """Construct the mixed boundary condition from the assignment."""
    phi = np.zeros((n, n), dtype=np.float64)
    phi[0, :] = 200.0
    phi[-1, :] = -400.0
    phi[:, 0] = 0.0
    phi[:, -1] = 200.0
    return phi


def charge_uniform(n: int, length: float) -> np.ndarray:
    """Construct the uniform 10 C charge density."""
    return np.full((n, n), 10.0 / length ** 2, dtype=np.float64)


def charge_gradient(n: int, length: float) -> np.ndarray:
    """Construct the linear charge gradient from bottom to top."""
    y = np.linspace(0.0, 1.0, n, dtype=np.float64)
    f = np.zeros((n, n), dtype=np.float64)
    f[:, :] = y[np.newaxis, :]
    return f


def charge_exponential(n: int, length: float) -> np.ndarray:
    """Construct the centred exponential charge distribution exp(-10 |r|)."""
    x = np.linspace(0.0, length, n, dtype=np.float64)
    y = np.linspace(0.0, length, n, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    cx = cy = length / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.exp(-10.0 * r)


def build_combinations(n: int, length: float) -> list:
    """Return the same ordered list of combinations used in task4.py."""
    zero = np.zeros((n, n), dtype=np.float64)

    bcs = [
        ("All edges +100 V", bc_all_plus100(n)),
        ("Top/bot +100 V, left/right -100 V", bc_tb_plus_lr_minus(n)),
        ("Top/left +200 V, bot 0 V, right -400 V", bc_mixed(n)),
    ]

    charges = [
        ("f = 0", zero),
        ("Uniform 10 C", charge_uniform(n, length)),
        ("Linear gradient", charge_gradient(n, length)),
        ("Exponential exp(-10|r|)", charge_exponential(n, length)),
    ]

    combos = []

    # Add the three Laplace-only cases.
    for bc_label, bc in bcs:
        combos.append({
            "bc_label": bc_label,
            "charge_label": "f = 0",
            "bc": bc,
            "charge": zero,
        })

    # Add the nine Poisson cases.
    for charge_label, charge in charges[1:]:
        for bc_label, bc in bcs:
            combos.append({
                "bc_label": bc_label,
                "charge_label": charge_label,
                "bc": bc,
                "charge": charge,
            })

    return combos


# -------------
#  SOR runner
# -------------

def run_sor(
    bc_array: np.ndarray,
    charge_array: np.ndarray,
    n: int,
    length: float,
) -> np.ndarray:
    """Run the SOR solver for one boundary/charge combination.

    Args:
        bc_array: N x N array containing the fixed boundary potentials.
        charge_array: N x N charge density array in C/m^2.
        n: Grid size.
        length: Physical side length in metres.

    Returns:
        Converged N x N potential array in volts.
    """
    solver = PoissonSOR(
        grid_size=n,
        length=length,
        tolerance=SOR_TOLERANCE,
    )
    solver.set_boundary_array(bc_array)
    solver.set_charge(charge_array)
    return solver.solve()


def extract_sor_values(phi: np.ndarray, h: float) -> dict:
    """Read off the SOR potential at the three Task 3 sample points.

    Args:
        phi: Converged N x N SOR potential array.
        h: Grid spacing in metres.

    Returns:
        Dict keyed by point key with float potential values.
    """
    values = {}
    for key, info in POINTS.items():
        x_val, y_val = info["xy"]
        i_idx = round(x_val / h)
        j_idx = round(y_val / h)
        values[key] = float(phi[i_idx, j_idx])
    return values


# ----------------
#  Data structure
# ----------------

@dataclass
class ComparisonRow:
    """Store one row of the Task 5 comparison table.

    Attributes:
        bc_label: Boundary condition description.
        charge_label: Charge distribution description.
        point_key: Short key for the evaluation point.
        phi_mc: Monte Carlo Green's function potential in volts.
        phi_mc_err: One-sigma Monte Carlo uncertainty in volts.
        phi_sor: Deterministic SOR potential in volts.
        diff: Difference phi_mc - phi_sor in volts.
        n_sigma: Absolute discrepancy measured in units of sigma.
        agrees: True if abs(diff) <= 2 * phi_mc_err.
    """

    bc_label: str
    charge_label: str
    point_key: str
    phi_mc: float
    phi_mc_err: float
    phi_sor: float
    diff: float
    n_sigma: float
    agrees: bool


# ----------------------
#  Loading Task 4 CSV
# ----------------------

def load_task4_csv(path: str) -> dict:
    """Load the Task 4 results CSV into a lookup dictionary.

    Args:
        path: Path to task4_results.csv.

    Returns:
        Dict keyed by (bc_label, charge_label, point_key), with values
        (phi_mc, phi_mc_err).

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Task 4 results CSV not found: {path}\n"
            "Run task4.py first."
        )

    data = {}
    with open(path, newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            key = (row["bc_label"], row["charge_label"], row["point_key"])
            data[key] = (float(row["phi_V"]), float(row["phi_err_V"]))
    return data


# -------------
#  Reporting
# -------------

def print_comparison_table(rows: list) -> None:
    """Print the full comparison table to stdout.

    Args:
        rows: List of ComparisonRow objects.
    """
    # Group rows by boundary condition and charge case so the table reads in
    # the same order as the Task 4 output.
    groups = {}
    for row in rows:
        group_key = (row.bc_label, row.charge_label)
        groups.setdefault(group_key, []).append(row)

    bc_width = 38
    charge_width = 24
    point_width = 13
    value_width = 11
    sigma_width = 12
    diff_width = 11
    nsigma_width = 9
    ok_width = 5

    header = (
        f"{'BC':<{bc_width}} "
        f"{'Charge':<{charge_width}} "
        f"{'Point':<{point_width}} "
        f"{'MC (V)':>{value_width}} "
        f"{'sigma (V)':>{sigma_width}} "
        f"{'SOR (V)':>{value_width}} "
        f"{'Diff (V)':>{diff_width}} "
        f"{'n_sigma':>{nsigma_width}} "
        f"{'OK':>{ok_width}}"
    )

    sep = "=" * len(header)

    print()
    print(sep)
    print("Task 5 - Comparison: Monte Carlo Green's function vs SOR relaxation")
    print(sep)
    print(header)
    print("-" * len(header))

    n_total = 0
    n_agree = 0
    prev_bc = None

    for (bc_label, charge_label), group_rows in groups.items():
        if prev_bc is not None and bc_label != prev_bc:
            print()
        prev_bc = bc_label

        for row in group_rows:
            ok_flag = "yes" if row.agrees else "no"
            print(
                f"{row.bc_label:<{bc_width}} "
                f"{row.charge_label:<{charge_width}} "
                f"{POINTS[row.point_key]['name']:<{point_width}} "
                f"{row.phi_mc:>{value_width}.4f} "
                f"{row.phi_mc_err:>{sigma_width}.2e} "
                f"{row.phi_sor:>{value_width}.4f} "
                f"{row.diff:>{diff_width}.4f} "
                f"{row.n_sigma:>{nsigma_width}.2f} "
                f"{ok_flag:>{ok_width}}"
            )
            n_total += 1
            if row.agrees:
                n_agree += 1

    print("-" * len(header))
    print(
        f"Agreement within 2 sigma: {n_agree}/{n_total} rows "
        f"({100.0 * n_agree / n_total:.1f} %)"
    )
    print(sep)
    print(
        "Diff = MC - SOR. n_sigma gives the discrepancy in units of the "
        "Monte Carlo standard error."
    )
    print("Agreement criterion: abs(diff) <= 2 * sigma.")
    print(sep)


def save_comparison_csv(
    rows: list,
    filename: str = "task5_comparison.csv",
) -> None:
    """Save the comparison table to CSV for reference.

    Args:
        rows: List of ComparisonRow objects.
        filename: Output filename inside the data directory.
    """
    path = os.path.join("data", filename)
    with open(path, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow([
            "bc_label",
            "charge_label",
            "point_key",
            "phi_mc_V",
            "phi_mc_err_V",
            "phi_sor_V",
            "diff_V",
            "n_sigma",
            "agrees",
        ])
        for row in rows:
            writer.writerow([
                row.bc_label,
                row.charge_label,
                row.point_key,
                f"{row.phi_mc:.8f}",
                f"{row.phi_mc_err:.4e}",
                f"{row.phi_sor:.8f}",
                f"{row.diff:.8f}",
                f"{row.n_sigma:.4f}",
                row.agrees,
            ])
    print(f"Saved {path}")


# ------
#  Main
# ------

def main() -> None:
    """Run the full Task 5 comparison workflow."""
    os.makedirs("data", exist_ok=True)

    h = LENGTH / (N - 1)

    print("\nTask 5 - MC Green's function vs SOR relaxation comparison")
    print(f"Grid: {N}x{N},  h = {h * 100:.1f} cm,  SOR tolerance = {SOR_TOLERANCE:.0e}")

    # Load the Monte Carlo results saved by task4.py.
    task4_csv = os.path.join("..", "task4", "data", "task4_results.csv")
    print(f"\nLoading Task 4 results from {task4_csv} ...")
    mc_data = load_task4_csv(task4_csv)
    print(f"  Loaded {len(mc_data)} entries.")

    # Rebuild the same list of cases used in Task 4.
    combos = build_combinations(N, LENGTH)
    print(f"\nRunning SOR solver for {len(combos)} combinations...")

    comparison_rows = []

    for combo_idx, combo in enumerate(combos):
        label = f"{combo['bc_label']} | {combo['charge_label']}"
        print(f"  [{combo_idx + 1:2d}/{len(combos)}] {label}")

        phi_sor = run_sor(combo["bc"], combo["charge"], N, LENGTH)
        sor_vals = extract_sor_values(phi_sor, h)

        for key in POINTS:
            csv_key = (combo["bc_label"], combo["charge_label"], key)
            mc_entry = mc_data.get(csv_key)

            if mc_entry is None:
                raise KeyError(
                    f"Could not find MC result for {csv_key} in Task 4 CSV.\n"
                    "Check that task4.py and task5.py use identical labels."
                )

            phi_mc, phi_mc_err = mc_entry
            diff = phi_mc - sor_vals[key]
            n_sigma = abs(diff) / phi_mc_err if phi_mc_err > 0 else 0.0
            agrees = abs(diff) <= 2.0 * phi_mc_err

            comparison_rows.append(
                ComparisonRow(
                    bc_label=combo["bc_label"],
                    charge_label=combo["charge_label"],
                    point_key=key,
                    phi_mc=phi_mc,
                    phi_mc_err=phi_mc_err,
                    phi_sor=sor_vals[key],
                    diff=diff,
                    n_sigma=n_sigma,
                    agrees=agrees,
                )
            )

    # Print and save the comparison results.
    print_comparison_table(comparison_rows)
    save_comparison_csv(comparison_rows)


if __name__ == "__main__":
    main()
