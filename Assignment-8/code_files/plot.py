#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

CSV_FILE = "timing_data.csv"
PLOT_PREFIX = "performance"

def generate_graphs():
    """Read CSV and create speedup & execution time plots."""
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Run run_benchmarks.py first.")
        return

    df = pd.read_csv(CSV_FILE, header=None,
                     names=["mpi", "omp", "nx", "ny", "npoints",
                            "total", "interp", "norm", "reverse", "mover"])
    df["cores"] = df["mpi"] * df["omp"]

    # Find serial time (cores=1)
    serial_df = df[df["cores"] == 1]
    if len(serial_df) == 0:
        print("No serial run found (cores=1). Cannot compute speedup.")
        return
    serial_time = serial_df["total"].values[0]
    df["speedup"] = serial_time / df["total"]

    # --- Speedup plot ---
    plt.figure(figsize=(10, 6))
    for (nx, ny, npoints), group in df.groupby(["nx", "ny", "npoints"]):
        group = group.sort_values("cores")
        plt.plot(group["cores"], group["speedup"], marker='o', linewidth=2,
                 markersize=8, label=f"Grid {nx}x{ny}, Np={npoints}")
    max_cores = df["cores"].max()
    plt.plot([1, max_cores], [1, max_cores], 'k--', linewidth=2, label="Ideal")
    plt.xlabel("Number of Cores (MPI × OpenMP)", fontsize=12)
    plt.ylabel("Speedup", fontsize=12)
    plt.title("Parallel Speedup", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{PLOT_PREFIX}_speedup.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Execution time plot ---
    plt.figure(figsize=(10, 6))
    for (nx, ny, npoints), group in df.groupby(["nx", "ny", "npoints"]):
        group = group.sort_values("cores")
        plt.plot(group["cores"], group["total"], marker='s', linewidth=2,
                 markersize=8, label=f"Grid {nx}x{ny}, Np={npoints}")
    plt.xlabel("Number of Cores", fontsize=12)
    plt.ylabel("Total Execution Time (seconds)", fontsize=12)
    plt.title("Execution Time Scaling", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{PLOT_PREFIX}_time.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Phase breakdown plot (stacked bar) ---
    plt.figure(figsize=(12, 6))
    phases = ['interp', 'norm', 'reverse', 'mover']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    # Sort by cores for x-axis order
    df_sorted = df.sort_values("cores")
    cores = df_sorted["cores"]
    bottom = np.zeros(len(df_sorted))
    for phase, color in zip(phases, colors):
        plt.bar(cores, df_sorted[phase], bottom=bottom,
                label=phase.capitalize(), color=color)
        bottom += df_sorted[phase]
    plt.xlabel("Number of Cores", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.title("Phase-wise Execution Time Breakdown", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(f"{PLOT_PREFIX}_phases.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nGraphs saved:")
    print(f"  - {PLOT_PREFIX}_speedup.png")
    print(f"  - {PLOT_PREFIX}_time.png")
    print(f"  - {PLOT_PREFIX}_phases.png")
    print("\nTiming summary:")
    print(df_sorted[["cores", "total", "speedup"]].to_string(index=False))

if __name__ == "__main__":
    generate_graphs()
