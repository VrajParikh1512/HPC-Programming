import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# ------------------- EXECUTION -------------------
def make_exec():
    for i in range(len(CONFIGS)):
        fig, ax = plt.subplots()
        ax.plot(THREADS, exec_times[i], marker=MARKERS[i], color=COLORS[i])
        ax.set_title(f"Execution Time - {CONFIGS[i]}")
        ax.set_xlabel("Threads")
        ax.set_ylabel("Time (s)")
        ax.grid(True)
        fig.savefig(f"exec_{CONFIGS[i]}.pdf")
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
        fig.savefig(f"speedup_{CONFIGS[i]}.pdf")
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
        fig.savefig(f"efficiency_{CONFIGS[i]}.pdf")
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
        fig.savefig(f"interpolation_{CONFIGS[i]}.pdf")
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
        fig.savefig(f"mover_{CONFIGS[i]}.pdf")
        plt.close(fig)

# ------------------- MAIN -------------------
if __name__ == "__main__":
    make_exec()
    make_speedup()
    make_efficiency()
    make_interpolation()
    make_mover()

    print("All individual plots generated (exec, speedup, efficiency, interpolation, mover).")