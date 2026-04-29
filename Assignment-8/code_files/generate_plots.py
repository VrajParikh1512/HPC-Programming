#!/usr/bin/env python3
"""
Performance Visualization Suite
Generates comprehensive performance analysis plots with custom styling
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import glob

# Custom color schemes - COMPLETELY DIFFERENT from matplotlib default
COLOR_SCHEMES = {
    'primary': ['#8B4789', '#E85D75', '#FFA07A', '#20B2AA', '#9370DB', '#FF6347'],
    'phases': ['#FF69B4', '#FFD700', '#00CED1', '#FF4500']
}

PLOT_DIR = "Output"
FIGURE_DIR = "figures"

def setup_plot_style():
    """Configure matplotlib with custom styling"""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 150

def create_speedup_visualization(df, output_path, dataset_name):
    """Generate speedup analysis with efficiency curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate efficiency
    df['efficiency'] = (df['speedup'] / df['cores']) * 100
    
    # Left plot: Speedup
    for idx, ((nx, ny, npoints), group) in enumerate(df.groupby(["nx", "ny", "npoints"])):
        group = group.sort_values("cores")
        color = COLOR_SCHEMES['primary'][idx % len(COLOR_SCHEMES['primary'])]
        ax1.plot(group["cores"], group["speedup"], 
                marker='D', linewidth=2.5, markersize=9, 
                color=color, label=f"{nx}×{ny} grid, {npoints:,} particles",
                markeredgecolor='white', markeredgewidth=1.5)
    
    # Ideal speedup line
    max_cores = df["cores"].max()
    ax1.plot([1, max_cores], [1, max_cores], 
            'k--', linewidth=2.5, alpha=0.7, label="Linear Speedup")
    
    ax1.set_xlabel("Number of Processing Cores (MPI × OpenMP)", fontweight='bold')
    ax1.set_ylabel("Speedup Factor", fontweight='bold')
    ax1.set_title(f"Parallel Speedup Analysis - {dataset_name}", fontweight='bold', pad=15)
    ax1.legend(loc='upper left', framealpha=0.95)
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    
    # Right plot: Parallel Efficiency
    for idx, ((nx, ny, npoints), group) in enumerate(df.groupby(["nx", "ny", "npoints"])):
        group = group.sort_values("cores")
        color = COLOR_SCHEMES['primary'][idx % len(COLOR_SCHEMES['primary'])]
        ax2.plot(group["cores"], group["efficiency"], 
                marker='o', linewidth=2.5, markersize=9,
                color=color, label=f"{nx}×{ny} grid",
                markeredgecolor='white', markeredgewidth=1.5)
    
    ax2.axhline(y=100, color='k', linestyle='--', linewidth=2.5, alpha=0.7, label='Ideal (100%)')
    ax2.set_xlabel("Number of Processing Cores", fontweight='bold')
    ax2.set_ylabel("Parallel Efficiency (%)", fontweight='bold')
    ax2.set_title(f"Parallel Efficiency - {dataset_name}", fontweight='bold', pad=15)
    ax2.legend(loc='upper right', framealpha=0.95)
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0, top=110)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {output_path}")

def create_execution_time_plot(df, output_path, dataset_name):
    """Generate execution time scaling visualization"""
    fig, ax = plt.subplots(figsize=(11, 7))
    
    for idx, ((nx, ny, npoints), group) in enumerate(df.groupby(["nx", "ny", "npoints"])):
        group = group.sort_values("cores")
        color = COLOR_SCHEMES['primary'][idx % len(COLOR_SCHEMES['primary'])]
        ax.plot(group["cores"], group["total_time"], 
               marker='s', linewidth=2.5, markersize=10,
               color=color, label=f"Grid {nx}×{ny}, {npoints:,} particles",
               markeredgecolor='white', markeredgewidth=1.5)
    
    ax.set_xlabel("Number of Processing Cores (MPI × OpenMP)", fontweight='bold')
    ax.set_ylabel("Total Execution Time (seconds)", fontweight='bold')
    ax.set_title(f"Strong Scaling Performance - {dataset_name}", fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    # Add annotation for best performance
    min_time_row = df.loc[df['total_time'].idxmin()]
    ax.annotate(f'Best: {min_time_row["total_time"]:.4f}s\n@ {int(min_time_row["cores"])} cores',
               xy=(min_time_row["cores"], min_time_row["total_time"]),
               xytext=(20, 30), textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=2))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {output_path}")

def create_phase_breakdown(df, output_path, dataset_name):
    """Generate stacked area chart for phase analysis"""
    fig, ax = plt.subplots(figsize=(13, 7))
    
    phases = ['interp', 'norm', 'reverse', 'mover']
    phase_labels = ['Interpolation', 'Normalization', 'Reverse Interp.', 'Particle Mover']
    colors = COLOR_SCHEMES['phases']
    
    df_sorted = df.sort_values("cores")
    x = df_sorted["cores"].values
    
    # Create stacked area chart
    y_stack = np.zeros(len(df_sorted))
    for phase, label, color in zip(phases, phase_labels, colors):
        y_values = df_sorted[phase].values
        ax.fill_between(x, y_stack, y_stack + y_values, 
                        label=label, alpha=0.85, color=color, edgecolor='white', linewidth=1.5)
        y_stack += y_values
    
    ax.set_xlabel("Number of Processing Cores", fontweight='bold')
    ax.set_ylabel("Execution Time (seconds)", fontweight='bold')
    ax.set_title(f"Computational Phase Distribution - {dataset_name}", fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {output_path}")

def create_phase_percentage_plot(df, output_path, dataset_name):
    """Generate percentage breakdown of phases"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    phases = ['interp', 'norm', 'reverse', 'mover']
    phase_labels = ['Interpolation', 'Normalization', 'Reverse Interp.', 'Particle Mover']
    colors = COLOR_SCHEMES['phases']
    
    df_sorted = df.sort_values("cores")
    
    # Calculate percentages
    for phase in phases:
        df_sorted[f'{phase}_pct'] = (df_sorted[phase] / df_sorted['total_time']) * 100
    
    # Stacked bar chart
    x_pos = np.arange(len(df_sorted))
    bottom = np.zeros(len(df_sorted))
    
    for phase, label, color in zip(phases, phase_labels, colors):
        values = df_sorted[f'{phase}_pct'].values
        ax.bar(x_pos, values, bottom=bottom, label=label, 
              color=color, alpha=0.9, edgecolor='white', linewidth=1.2)
        bottom += values
    
    ax.set_xlabel("Number of Processing Cores", fontweight='bold')
    ax.set_ylabel("Percentage of Total Time (%)", fontweight='bold')
    ax.set_title(f"Phase-wise Time Distribution (%) - {dataset_name}", fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_sorted['cores'].astype(int))
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {output_path}")

def generate_summary_table(df, dataset_name):
    """Generate formatted summary statistics"""
    print(f"\n{'='*70}")
    print(f"  Performance Summary: {dataset_name}")
    print(f"{'='*70}")
    print(f"{'Cores':<8} {'Time(s)':<12} {'Speedup':<12} {'Efficiency(%)':<15}")
    print(f"{'-'*70}")
    
    df_sorted = df.sort_values("cores")
    for _, row in df_sorted.iterrows():
        print(f"{int(row['cores']):<8} {row['total_time']:<12.6f} "
              f"{row['speedup']:<12.4f} {row['efficiency']:<15.2f}")
    print(f"{'='*70}\n")

def process_dataset(csv_file):
    """Process a single timing dataset"""
    if not os.path.exists(csv_file):
        print(f"⚠ Warning: {csv_file} not found")
        return
    
    # Read data
    try:
        df = pd.read_csv(csv_file)
        if 'mpi' not in df.columns:
            raise ValueError("No header")
    except:
        df = pd.read_csv(csv_file, header=None,
                        names=["mpi", "omp", "nx", "ny", "npoints",
                               "total_time", "interp", "norm", "reverse", "mover"])
    
    # Calculate metrics
    df["cores"] = df["mpi"] * df["omp"]
    serial_df = df[df["cores"] == 1]
    
    if len(serial_df) == 0:
        print(f"⚠ No serial baseline in {csv_file}, skipping...")
        return
    
    serial_time = serial_df["total_time"].values[0]
    df["speedup"] = serial_time / df["total_time"]
    df["efficiency"] = (df["speedup"] / df["cores"]) * 100
    
    # Generate output paths
    basename = os.path.basename(csv_file).replace('.csv', '')
    dataset_name = basename.replace('timing_data', 'Dataset ')
    
    os.makedirs(FIGURE_DIR, exist_ok=True)
    
    print(f"\n📊 Processing {basename}...")
    
    # Generate all plots
    create_speedup_visualization(df, 
        f"{FIGURE_DIR}/{basename}_speedup_efficiency.png", dataset_name)
    create_execution_time_plot(df, 
        f"{FIGURE_DIR}/{basename}_execution_time.png", dataset_name)
    create_phase_breakdown(df, 
        f"{FIGURE_DIR}/{basename}_phase_breakdown.png", dataset_name)
    create_phase_percentage_plot(df, 
        f"{FIGURE_DIR}/{basename}_phase_percentage.png", dataset_name)
    
    # Print summary
    generate_summary_table(df, dataset_name)

def main():
    """Main execution function"""
    setup_plot_style()
    
    print("\n" + "="*70)
    print("  PERFORMANCE VISUALIZATION SUITE")
    print("="*70)
    
    # Find all CSV files
    csv_files = sorted(glob.glob(os.path.join(PLOT_DIR, "timing_data*.csv")))
    
    if not csv_files:
        print("❌ No timing data files found in Output directory")
        return
    
    print(f"\n✓ Found {len(csv_files)} dataset(s)")
    
    for csv_file in csv_files:
        process_dataset(csv_file)
    
    print("\n" + "="*70)
    print(f"  ✅ All visualizations saved to '{FIGURE_DIR}/' directory")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
