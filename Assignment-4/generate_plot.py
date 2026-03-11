import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import io

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

def parse_cluster_exp1(filepath):
    """Parses result01.csv with sparse Grid column."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data = []
    current_grid = None
    
    for line in lines[1:]: # Skip header
        parts = line.strip().split(',')
        if len(parts) < 3: continue
        
        grid_str = parts[0].strip()
        if grid_str:
            current_grid = grid_str.replace(' ', '')
        
        if not current_grid: continue
        
        try:
            particles = float(parts[1])
            time = float(parts[2])
            data.append({'Grid': current_grid, 'Particles': particles, 'Time': time})
        except ValueError:
            continue
            
    df = pd.DataFrame(data)
    return df

def parse_lab_exp1(filepath):
    """Parses results_exp01.csv text format."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    configs = {}
    # Split by "Configuration"
    sections = re.split(r'Configuration \d+', content)
    
    # We expect 3 configs. The first split might be empty.
    # Improve regex to capture config ID if needed, but sequential is fine for now if order matches.
    
    # Let's verify grid sizes to be safe
    data = []
    
    # Regex to find NX=... NY=...
    # And then list of Particles: ... Time = ...
    
    # Cleaner approach: line by line state machine
    lines = content.split('\n')
    current_grid = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('NX='):
            # e.g. NX=250 NY=100
            nx_match = re.search(r'NX=(\d+)', line)
            ny_match = re.search(r'NY=(\d+)', line)
            if nx_match and ny_match:
                current_grid = f"{nx_match.group(1)}X{ny_match.group(1)}"
        elif line.startswith('Particles:'):
            if current_grid:
                p_current = float(line.split(':')[1])
        elif line.startswith('Total interpolation time'):
            # Total interpolation time = 0.000015 seconds
            t_match = re.search(r'=\s*([\d\.]+)\s*seconds', line)
            if t_match and current_grid:
                t_val = float(t_match.group(1))
                data.append({'Grid': current_grid, 'Particles': p_current, 'Time': t_val})

    return pd.DataFrame(data)

def parse_cluster_exp2(filepath):
    """Parses result02.csv."""
    # This file has 3 columns corresponding to 3 grids.
    # It lists iteration times and a total at the bottom.
    # We just need the totals for Experiment 2 plot? 
    # "x-axis represents the problem index (1 to 3) and the y-axis represents the total interpolation time"
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    totals = []
    
    # Find the line starting with "Total Interpolation Time"
    for line in lines:
        if "Total Interpolation Time" in line:
            # Split by comma
            parts = line.split(',')
            # Format: Total Interpolation Time (Over 10 Iters) = 22.321155 seconds
            for part in parts:
                match = re.search(r'=\s*([\d\.]+)\s*seconds', part)
                if match:
                    totals.append(float(match.group(1)))
                    
    # Assuming totals order matches Grid 1, 2, 3 (250, 500, 1000)
    # The header confirms this: Grid: 250..., Grid: 500..., Grid: 1000...
    return totals

def parse_lab_exp2(filepath):
    """Parses results_exp02.csv."""
    with open(filepath, 'r') as f:
        content = f.read()
        
    data = []
    lines = content.split('\n')
    
    # Look for "Total interpolation time = ..."
    # It appears once per configuration
    
    for line in lines:
        if line.strip().startswith('Total interpolation time'):
             match = re.search(r'=\s*([\d\.]+)\s*seconds', line)
             if match:
                 data.append(float(match.group(1)))
    return data

def parse_cluster_exp3(filepath):
    """Parses result03.csv."""
    # Contains Serial_mover and Parallel_mover blocks
    # Header: Iter,Interp(s),Mover(s),Total(s),,
    # Row: 1,0.3651,,0.3111,,0.6762
    
    with open(filepath, 'r') as f:
        content = f.read()
        
    # Split into Serial and Parallel blocks
    blocks = content.split('Parallel_mover')
    
    serial_block = blocks[0]
    parallel_block = blocks[1] if len(blocks) > 1 else ""
    
    def parse_block(block_text):
        data = []
        lines = block_text.split('\n')
        start_parsing = False
        for line in lines:
            if 'Iter,Interp(s)' in line:
                start_parsing = True
                continue
            if not start_parsing: continue
            if 'Total Execution Time' in line: break
            if '---' in line: continue
            if not line.strip(): continue
            
            parts = line.split(',')
            # Based on inspection:
            # Col 0: Iter
            # Col 1: Interp
            # Col 3: Mover
            # Col 5: Total (sometimes 5 or 6 depending on trailing commas)
            
            try:
                iter_num = int(parts[0])
                interp = float(parts[1])
                mover = float(parts[3]) 
                # Total might be index 5
                total = float(parts[5])
                data.append({'Iter': iter_num, 'Interp': interp, 'Mover': mover, 'Total': total})
            except (ValueError, IndexError):
                continue
        return pd.DataFrame(data)

    df_serial = parse_block(serial_block)
    df_parallel = parse_block(parallel_block)
    
    return df_serial, df_parallel

def parse_lab_exp3(filepath):
    """Parses results_exp03.csv."""
    with open(filepath, 'r') as f:
        content = f.read()
        
    # Split into Serial and Parallel
    blocks = content.split('Parallel')
    serial_block = blocks[0]
    parallel_block = blocks[1] if len(blocks) > 1 else ""
    
    def parse_block(block_text):
        data = []
        lines = block_text.split('\n')
        start_parsing = False
        for line in lines:
            if 'Iter' in line and 'Interpolation' in line:
                start_parsing = True
                continue
            if not start_parsing: continue
            if not line.strip(): continue
            
            # Text based table
            parts = line.split()
            if len(parts) < 4: continue
            
            try:
                iter_num = int(parts[0])
                interp = float(parts[1])
                mover = float(parts[2])
                total = float(parts[3])
                data.append({'Iter': iter_num, 'Interp': interp, 'Mover': mover, 'Total': total})
            except ValueError:
                continue
        return pd.DataFrame(data)

    df_serial = parse_block(serial_block)
    df_parallel = parse_block(parallel_block)
    return df_serial, df_parallel

def main():
    # --- Experiment 1 ---
    print("Processing Experiment 1...")
    df_clust_1 = parse_cluster_exp1('Assignment-4/data_cluster/result01.csv')
    df_lab_1 = parse_lab_exp1('Assignment-4/data_lab/results_exp01.csv')
    
    grids = ['250X100', '500X200', '1000X400']
    
    # 3 Comparison Plots
    for grid in grids:
        plt.figure(figsize=(10, 6))
        
        # Cluster Data
        sub_c = df_clust_1[df_clust_1['Grid'] == grid]
        plt.loglog(sub_c['Particles'], sub_c['Time'], label='HPC Cluster', marker='o')
        
        # Lab Data
        sub_l = df_lab_1[df_lab_1['Grid'] == grid]
        plt.loglog(sub_l['Particles'], sub_l['Time'], label='Lab PC', marker='x')
        
        plt.title(f'Exp 1: Particles vs Time (Grid {grid})')
        plt.xlabel('Number of Particles (Log Scale)')
        plt.ylabel('Execution Time (s) (Log Scale)')
        plt.legend()
        plt.grid(True, which="both", ls="-")
        plt.savefig(f'plots/Exp1_{grid}.png')
        plt.close()

    # --- Experiment 2 ---
    print("Processing Experiment 2...")
    totals_clust = parse_cluster_exp2('Assignment-4/data_cluster/result02.csv')
    totals_lab = parse_lab_exp2('Assignment-4/data_lab/results_exp02.csv')
    
    indices = [1, 2, 3] # Problem indices
    
    # Lab Plot
    plt.figure(figsize=(8, 6))
    plt.bar(indices, totals_lab, color='skyblue')
    plt.title('Exp 2: Total Interpolation Time vs Problem Index (Lab PC)')
    plt.xlabel('Problem Index (1=250x100, 2=500x200, 3=1000x400)')
    plt.ylabel('Total Interpolation Time (s)')
    plt.xticks(indices)
    plt.grid(axis='y')
    plt.savefig('plots/Exp2_Lab.png')
    plt.close()
    
    # Cluster Plot
    plt.figure(figsize=(8, 6))
    plt.bar(indices, totals_clust, color='salmon')
    plt.title('Exp 2: Total Interpolation Time vs Problem Index (HPC Cluster)')
    plt.xlabel('Problem Index (1=250x100, 2=500x200, 3=1000x400)')
    plt.ylabel('Total Interpolation Time (s)')
    plt.xticks(indices)
    plt.grid(axis='y')
    plt.savefig('plots/Exp2_Cluster.png')
    plt.close()

    # --- Experiment 3 ---
    print("Processing Experiment 3...")
    c_serial, c_parallel = parse_cluster_exp3('Assignment-4/data_cluster/result03.csv')
    l_serial, l_parallel = parse_lab_exp3('Assignment-4/data_lab/results_exp03.csv')
    
    environments = [
        ('HPC Cluster', c_serial, c_parallel),
        ('Lab PC', l_serial, l_parallel)
    ]
    
    for name, ser, par in environments:
        safe_name = name.replace(' ', '_')
        
        # 1. Iteration vs (Interp, Mover, Total) -> Using PARALLEL run usually, or Serial? 
        # "Plot a single graph showing three curves" -> Usually implies the optimized version or just one.
        # I'll plot the Parallel version as it's the "result" of the assignment.
        
        plt.figure(figsize=(10, 6))
        plt.plot(par['Iter'], par['Interp'], label='Interpolation', marker='.')
        plt.plot(par['Iter'], par['Mover'], label='Mover', marker='.')
        plt.plot(par['Iter'], par['Total'], label='Total', marker='.')
        plt.title(f'Exp 3: Component Times vs Iteration ({name}) - Parallel Run')
        plt.xlabel('Iteration')
        plt.ylabel('Time (s)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'plots/Exp3_Components_{safe_name}.png')
        plt.close()
        
        # 2. Mover Serial vs Parallel
        plt.figure(figsize=(10, 6))
        plt.plot(ser['Iter'], ser['Mover'], label='Serial Mover', marker='o')
        plt.plot(par['Iter'], par['Mover'], label='Parallel Mover', marker='x')
        plt.title(f'Exp 3: Mover Serial vs Parallel ({name})')
        plt.xlabel('Iteration')
        plt.ylabel('Mover Time (s)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'plots/Exp3_Mover_Comparison_{safe_name}.png')
        plt.close()
        
        # 3. Speedup
        # Speedup S = T_serial / T_parallel (Mover only?)
        # "Speedup achieved by the parallel Mover implementation" -> Just Mover Time.
        
        # Ensure lengths match
        min_len = min(len(ser), len(par))
        ser_mover = ser['Mover'][:min_len]
        par_mover = par['Mover'][:min_len]
        iterations = ser['Iter'][:min_len]
        
        speedup = ser_mover / par_mover
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, speedup, label='Speedup', marker='d', color='green')
        plt.title(f'Exp 3: Mover Speedup vs Iteration ({name})')
        plt.xlabel('Iteration')
        plt.ylabel('Speedup (T_serial / T_parallel)')
        plt.grid(True)
        plt.savefig(f'plots/Exp3_Speedup_{safe_name}.png')
        plt.close()

if __name__ == "__main__":
    main()