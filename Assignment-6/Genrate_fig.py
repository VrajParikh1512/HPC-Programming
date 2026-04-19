"""
generate_figures.py
===================
Generates all figures required by hpc_lab6_report.tex.

Run:
    python3 generate_figures.py

Output files (place in the same directory as the .tex file):
    fig1_parallel_diagram.pdf   — Parallel implementation diagram (Section 1)
    fig2_exec_time.pdf          — Execution time vs. cores  (Section 3)
    fig3_speedup.pdf            — Speedup vs. cores          (Section 3)
    fig4_efficiency.pdf         — Parallel efficiency vs. cores (Section 4)

NOTE ─ REPLACE PLACEHOLDER DATA
    The timing arrays below (exec_times) contain synthetic/placeholder values.
    Replace them with your actual measured runtimes before submitting.
    Format: exec_times[config_index][thread_count_index]
    Thread counts: [1, 2, 4, 8, 16]  (index 0 = serial / 1 thread)
"""

import matplotlib
matplotlib.use("Agg")          # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

# ─── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 7,
})

THREADS = [1, 2, 4, 8, 16]          # thread counts (index 0 = serial)
THREAD_LABELS = [str(t) for t in THREADS]
CONFIGS = ["C1 (250, 100, 900K)",
           "C2 (250, 100, 5M)",
           "C3 (500, 200, 3.6M)",
           "C4 (500, 200, 20M)",
           "C5 (1000, 400, 14M)"]
COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
MARKERS = ["o", "s", "^", "D", "v"]

# ─────────────────────────────────────────────────────────────────────────────
# !! REPLACE these rows with your measured execution times (seconds) !!
# Rows = configurations C1..C5
# Cols = thread counts  [1, 2, 4, 8, 16]
# ─────────────────────────────────────────────────────────────────────────────
exec_times = np.array([
    # 1-thread  2-thr   4-thr   8-thr  16-thr
    [0.011808,      0.006256,   0.004018,   0.002603,   0.003552],   # C1
    [0.063993,      0.033464,   0.018739,   0.014988,   0.010854],   # C2
    [0.052775,      0.053515,   0.033297,   0.021589,   0.015633],   # C3
    [0.289826,      0.226787,   0.131001,   0.084367,   0.050171],   # C4
    [0.231002,      0.221152,   0.123602,   0.089742,   0.108284],   # C5
])
# ─────────────────────────────────────────────────────────────────────────────

# Derived quantities
speedup    = exec_times[:, 0:1] / exec_times          # S(p) = T1 / Tp
efficiency = speedup / np.array(THREADS)              # E(p) = S(p) / p


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Parallel implementation diagram (matplotlib drawing)
# ═════════════════════════════════════════════════════════════════════════════
def make_fig1():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Figure 1: Parallel Bilinear Interpolation — OpenMP Thread Layout",
                 fontweight="bold", pad=12)

    # ── Particles pool ───────────────────────────────────────────────────────
    rect_particles = mpatches.FancyBboxPatch(
        (0.2, 3.5), 2.2, 1.2, boxstyle="round,pad=0.1",
        facecolor="#AED6F1", edgecolor="#2980B9", linewidth=1.5)
    ax.add_patch(rect_particles)
    ax.text(1.3, 4.12, "Particle Array\n(N particles)", ha="center", va="center",
            fontsize=10, fontweight="bold")

    # ── OpenMP fork ──────────────────────────────────────────────────────────
    ax.annotate("", xy=(3.3, 4.15), xytext=(2.4, 4.15),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(2.85, 4.38, "omp parallel\nfor (static)", ha="center", fontsize=8,
            color="#555555")

    # ── Thread boxes ─────────────────────────────────────────────────────────
    thread_colors = ["#ABEBC6", "#FAD7A0", "#F1948A", "#C39BD3"]
    for k in range(4):
        y0 = 5.8 - k * 1.4
        rect = mpatches.FancyBboxPatch(
            (3.3, y0), 2.4, 1.1, boxstyle="round,pad=0.08",
            facecolor=thread_colors[k], edgecolor="#555", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(4.5, y0 + 0.55,
                f"Thread {k}\nlocal_mesh[{k}][...]",
                ha="center", va="center", fontsize=9)
        # arrows from particle pool
        ax.annotate("", xy=(3.3, y0 + 0.55), xytext=(2.42, 4.15),
                    arrowprops=dict(arrowstyle="->", color="#777", lw=1.0,
                                   connectionstyle="arc3,rad=0.0"))

    ax.text(3.3, 0.55, '...', ha="center", fontsize=16)

    # ── Reduction arrow ──────────────────────────────────────────────────────
    ax.annotate("", xy=(7.3, 4.15), xytext=(5.72, 4.15),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(6.5, 4.45, "Parallel\nreduction\n(sum + / N)", ha="center",
            fontsize=8, color="#555555")

    # ── Final mesh ───────────────────────────────────────────────────────────
    rect_mesh = mpatches.FancyBboxPatch(
        (7.3, 3.5), 2.4, 1.2, boxstyle="round,pad=0.1",
        facecolor="#F9E79F", edgecolor="#B7950B", linewidth=1.5)
    ax.add_patch(rect_mesh)
    ax.text(8.5, 4.12, "mesh_value\n(GRID_X × GRID_Y)",
            ha="center", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig("fig1_parallel_diagram_cluster.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_parallel_diagram.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Execution time vs. cores
# ═════════════════════════════════════════════════════════════════════════════
def make_fig2():
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (cfg, col, mk) in enumerate(zip(CONFIGS, COLORS, MARKERS)):
        ax.plot(THREADS, exec_times[idx], color=col, marker=mk, label=cfg)

    ax.set_xlabel("Number of OpenMP Threads")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("Figure 2: Execution Time vs. Number of Cores")
    ax.set_xticks(THREADS)
    ax.set_xticklabels(THREAD_LABELS)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig("fig2_exec_time_cluster.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_exec_time.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Speedup vs. cores
# ═════════════════════════════════════════════════════════════════════════════
def make_fig3():
    fig, ax = plt.subplots(figsize=(8, 5))
    # Ideal speedup line
    ax.plot(THREADS, THREADS, "k--", linewidth=1.5, label="Ideal speedup")
    for idx, (cfg, col, mk) in enumerate(zip(CONFIGS, COLORS, MARKERS)):
        ax.plot(THREADS, speedup[idx], color=col, marker=mk, label=cfg)

    ax.set_xlabel("Number of OpenMP Threads")
    ax.set_ylabel("Speedup  $S(p) = T_1 / T_p$")
    ax.set_title("Figure 3: Speedup vs. Number of Cores")
    ax.set_xticks(THREADS)
    ax.set_xticklabels(THREAD_LABELS)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig("fig3_speedup_cluster.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_speedup.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Parallel efficiency vs. cores
# ═════════════════════════════════════════════════════════════════════════════
def make_fig4():
    fig, ax = plt.subplots(figsize=(8, 5))
    # Perfect efficiency reference
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5,
               label="Perfect efficiency (E = 1)")
    for idx, (cfg, col, mk) in enumerate(zip(CONFIGS, COLORS, MARKERS)):
        ax.plot(THREADS, efficiency[idx], color=col, marker=mk, label=cfg)

    ax.set_xlabel("Number of OpenMP Threads")
    ax.set_ylabel("Parallel Efficiency  $E(p) = S(p) / p$")
    ax.set_title("Figure 4: Parallel Efficiency vs. Number of Cores")
    ax.set_xticks(THREADS)
    ax.set_xticklabels(THREAD_LABELS)
    ax.set_ylim(0, 1.25)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig("fig4_efficiency_cluster.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig4_efficiency.pdf")


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    print("\nAll figures generated successfully.")
    print("Place all .pdf files in the same directory as hpc_lab6_report.tex")
    print("and compile with:  pdflatex hpc_lab6_report.tex")