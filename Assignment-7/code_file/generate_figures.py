import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
})

# ------------------- LOAD CSV -------------------
df = pd.read_csv("results.csv")

configs = sorted(df['Config'].unique())
threads = sorted(df['Cores'].unique())

CONFIGS = [str(c) for c in configs]
THREADS = threads

exec_times, int_times, move_times = [], [], []

for cfg in configs:
    sub = df[df['Config'] == cfg].sort_values('Cores')
    exec_times.append(sub['Total_Alg_Time'].values)
    int_times.append(sub['Int_Time'].values)
    move_times.append(sub['Mover_Time'].values)

exec_times = np.array(exec_times)
int_times  = np.array(int_times)
move_times = np.array(move_times)

speedup = exec_times[:, 0:1] / exec_times
efficiency = speedup / np.array(THREADS)

COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
MARKERS = ["o", "s", "^", "D", "v"]

# Create output directory
output_dir = "Assignment_Plots"
os.makedirs(output_dir, exist_ok=True)

# ------------------- EXECUTION -------------------
def make_exec():
    for i in range(len(CONFIGS)):
        fig, ax = plt.subplots()
        ax.plot(THREADS, exec_times[i], marker=MARKERS[i], color=COLORS[i])
        ax.set_title(f"Execution Time - {CONFIGS[i]}")
        ax.set_xlabel("Threads")
        ax.set_ylabel("Time (s)")
        ax.grid(True)
        fig.savefig(f"{output_dir}/exec_{CONFIGS[i]}.pdf")
        plt.close(fig)

# ------------------- SPEEDUP -------------------
def make_speedup():
    for i in range(len(CONFIGS)):
        fig, ax = plt.subplots()
        ax.plot(THREADS, THREADS, "k--", label="Ideal")
        ax.plot(THREADS, speedup[i], marker=MARKERS[i], color=COLORS[i])
        ax.set_title(f"Speedup - {CONFIGS[i]}")
        ax.set_xlabel("Threads")
        ax.set_ylabel("Speedup")
        ax.legend()
        ax.grid(True)
        fig.savefig(f"{output_dir}/speedup_{CONFIGS[i]}.pdf")
        plt.close(fig)

# ------------------- EFFICIENCY -------------------
def make_efficiency():
    for i in range(len(CONFIGS)):
        fig, ax = plt.subplots()
        ax.axhline(1.0, linestyle="--", color="black")
        ax.plot(THREADS, efficiency[i], marker=MARKERS[i], color=COLORS[i])
        ax.set_title(f"Efficiency - {CONFIGS[i]}")
        ax.set_xlabel("Threads")
        ax.set_ylabel("Efficiency")
        ax.grid(True)
        fig.savefig(f"{output_dir}/efficiency_{CONFIGS[i]}.pdf")
        plt.close(fig)

# ------------------- INTERPOLATION -------------------
def make_interpolation():
    for i in range(len(CONFIGS)):
        fig, ax = plt.subplots()
        ax.plot(THREADS, int_times[i], marker=MARKERS[i], color=COLORS[i])
        ax.set_title(f"Interpolation Time - {CONFIGS[i]}")
        ax.set_xlabel("Threads")
        ax.set_ylabel("Time (s)")
        ax.grid(True)
        fig.savefig(f"{output_dir}/interpolation_{CONFIGS[i]}.pdf")
        plt.close(fig)

# ------------------- MOVER -------------------
def make_mover():
    for i in range(len(CONFIGS)):
        fig, ax = plt.subplots()
        ax.plot(THREADS, move_times[i], marker=MARKERS[i], color=COLORS[i])
        ax.set_title(f"Mover Time - {CONFIGS[i]}")
        ax.set_xlabel("Threads")
        ax.set_ylabel("Time (s)")
        ax.grid(True)
        fig.savefig(f"{output_dir}/mover_{CONFIGS[i]}.pdf")
        plt.close(fig)

# ------------------- COMBINED PLOTS -------------------
def make_combined_exec():
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(CONFIGS)):
        ax.plot(THREADS, exec_times[i], marker=MARKERS[i], color=COLORS[i], 
                linewidth=2, markersize=7, label=f'Config {CONFIGS[i]}')
    ax.set_title('Combined Execution Time vs Cores', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Cores', fontsize=12)
    ax.set_ylabel('Execution Time (Seconds)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/Combined_Execution_Time.png', dpi=300)
    plt.close(fig)

def make_combined_speedup():
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(CONFIGS)):
        ax.plot(THREADS, speedup[i], marker=MARKERS[i], color=COLORS[i], 
                linewidth=2, markersize=7, label=f'Config {CONFIGS[i]}')
    ax.plot(THREADS, THREADS, linestyle='--', color='black', linewidth=2, label='Ideal Speedup')
    ax.set_title('Combined Speedup vs Cores', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Cores', fontsize=12)
    ax.set_ylabel('Speedup (S = T1 / Tn)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/Combined_Speedup.png', dpi=300)
    plt.close(fig)

def make_combined_phase():
    fig, ax = plt.subplots(figsize=(12, 7))
    for i in range(len(CONFIGS)):
        ax.plot(THREADS, int_times[i], marker='o', linestyle='-', color=COLORS[i], 
                linewidth=2, markersize=6, label=f'Config {CONFIGS[i]} (Int)')
        ax.plot(THREADS, move_times[i], marker='^', linestyle='--', color=COLORS[i], 
                linewidth=2, markersize=6, label=f'Config {CONFIGS[i]} (Mover)')
    ax.set_title('Combined Phase Analysis (Interpolation vs Mover)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Cores', fontsize=12)
    ax.set_ylabel('Execution Time (Seconds)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/Combined_Phase_Analysis.png', dpi=300)
    plt.close(fig)

# ------------------- MAIN -------------------
if __name__ == "__main__":
    make_exec()
    make_speedup()
    make_efficiency()
    make_interpolation()
    make_mover()
    make_combined_exec()
    make_combined_speedup()
    make_combined_phase()

    print(f"All plots generated successfully in {output_dir}/ directory!")
    print("Individual plots: exec, speedup, efficiency, interpolation, mover (PDF format)")
    print("Combined plots: Combined_Execution_Time, Combined_Speedup, Combined_Phase_Analysis (PNG format)")