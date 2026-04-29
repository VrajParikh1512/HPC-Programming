#!/usr/bin/env python
from __future__ import print_function
import subprocess
import os
import sys
import struct

MPI_PATH = "/usr/mpi/gcc/openmpi-1.8.8/bin"
MPICXX = os.path.join(MPI_PATH, "mpicxx")
MPIRUN = os.path.join(MPI_PATH, "mpirun")
EXECUTABLE = "./hpc"
INPUT_FILE = "input.bin"
CSV_FILE = "timing_data.csv"
HOSTFILE = "sources.txt"          # list of nodes (gics1, gics2, gics3, gics4)

total_cores_list = [2, 4, 8, 16, 32, 64]
decompositions = {
    2:  [(1,2), (2,1)],
    4:  [(1,4), (2,2), (4,1)],
    8:  [(1,8), (2,4), (4,2), (8,1)],
    16: [(2,8), (4,4), (8,2), (16,1)],
    32: [(4,8), (8,4), (16,2), (32,1)],
    64: [(8,8), (16,4), (32,2), (64,1)],
}

def run_command(cmd, env=None):
    if env is None:
        env = os.environ.copy()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    stdout, stderr = proc.communicate()
    return proc.returncode, stdout, stderr

def compile_program():
    print("Compiling program...")
    cmd = [MPICXX, "-fopenmp", "-O3", "-march=native", "main.cpp", "utils.cpp", "-o", "hpc"]
    print("Running:", " ".join(cmd))
    ret, out, err = run_command(cmd)
    if ret != 0:
        print("Compilation failed:", err)
        sys.exit(1)
    print("Compilation successful.")

def parse_output(stdout):
    """Parse the output line: Total: X, Interp: Y, Norm: Z, Reverse: W, Mover: V"""
    for line in stdout.splitlines():
        if line.startswith("Total:"):
            # Example: "Total: 0.281278, Interp: 0.129007, Norm: 0.001426, Reverse: 0.120289, Mover: 0.030556"
            parts = line.split(',')
            total = float(parts[0].split()[1])
            interp = float(parts[1].split()[1])
            norm = float(parts[2].split()[1])
            reverse = float(parts[3].split()[1])
            mover = float(parts[4].split()[1])
            return total, interp, norm, reverse, mover
    return None, None, None, None, None

def run_single_config(mpi_ranks, omp_threads):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_threads)
    cmd = [MPIRUN, "-np", str(mpi_ranks)]
    if mpi_ranks > 1 and os.path.exists(HOSTFILE):
        cmd += ["--hostfile", HOSTFILE]
    cmd += [EXECUTABLE, INPUT_FILE]
    print("  Running: {} with OMP_NUM_THREADS={}".format(" ".join(cmd), omp_threads))
    ret, stdout, stderr = run_command(cmd, env)
    if ret != 0:
        print("  ERROR (code {}): {}".format(ret, stderr))
        return None
    return parse_output(stdout)

def get_grid_info():
    """Read NX, NY, NUM_Points from input.bin (binary, little-endian ints)."""
    with open(INPUT_FILE, 'rb') as f:
        nx = struct.unpack('<i', f.read(4))[0]
        ny = struct.unpack('<i', f.read(4))[0]
        npoints = struct.unpack('<i', f.read(4))[0]
        maxiter = struct.unpack('<i', f.read(4))[0]
    return nx, ny, npoints

def collect_data():
    # Remove old CSV
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)
    
    # Get grid info once
    nx, ny, npoints = get_grid_info()
    
    # Write header
    with open(CSV_FILE, 'w') as f:
        f.write("mpi,omp,nx,ny,npoints,total_time,interp,norm,reverse,mover\n")
    
    # Serial baseline
    print("\n=== Serial (1 core) ===")
    res = run_single_config(1, 1)
    if res is None:
        sys.exit(1)
    total, interp, norm, reverse, mover = res
    with open(CSV_FILE, 'a') as f:
        f.write("1,1,{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
            nx, ny, npoints, total, interp, norm, reverse, mover))
    
    # For each core count, try decompositions and keep best (lowest total time)
    for cores in total_cores_list:
        print("\n=== Total cores = {} ===".format(cores))
        best_time = None
        best_data = None
        for mpi, omp in decompositions.get(cores, []):
            print("  Trying MPI={}, OMP={}".format(mpi, omp))
            res = run_single_config(mpi, omp)
            if res is not None:
                t, interp, norm, reverse, mover = res
                if best_time is None or t < best_time:
                    best_time = t
                    best_data = (mpi, omp, t, interp, norm, reverse, mover)
        if best_data:
            mpi, omp, t, interp, norm, reverse, mover = best_data
            print("  Best: MPI={}, OMP={}, time={:.4f}s".format(mpi, omp, t))
            with open(CSV_FILE, 'a') as f:
                f.write("{},{},{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
                    mpi, omp, nx, ny, npoints, t, interp, norm, reverse, mover))
        else:
            print("  No successful run for {} cores".format(cores))

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print("Input file {} not found.".format(INPUT_FILE))
        sys.exit(1)
    if not os.path.exists(MPICXX):
        print("MPI compiler not found at {}".format(MPICXX))
        sys.exit(1)
    compile_program()
    collect_data()
    print("\nBenchmark complete. Results saved to {}".format(CSV_FILE))
"""#!/usr/bin/env python
from __future__ import print_function
import subprocess
import os
import sys

# Configuration
MPI_PATH = "/usr/mpi/gcc/openmpi-1.8.8/bin"
MPICXX = os.path.join(MPI_PATH, "mpicxx")
MPIRUN = os.path.join(MPI_PATH, "mpirun")
EXECUTABLE = "./hpc"
INPUT_FILE = "input.bin"
CSV_FILE = "timing_data.csv"

# Thread counts to test (OpenMP)
threads_list = [1, 2, 4, 6, 8, 16]

def run_command(cmd, env=None):
    Run a command and return (returncode, stdout, stderr).
    if env is None:
        env = os.environ.copy()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    stdout, stderr = proc.communicate()
    return proc.returncode, stdout, stderr

def compile_program():
    Compile the program using mpicxx.
    print("Compiling program...")
    cmd = [MPICXX, "-fopenmp", "-O3", "-march=native",
           "main.cpp", "utils.cpp", "-o", "hpc"]
    print("Running: {}".format(" ".join(cmd)))
    returncode, stdout, stderr = run_command(cmd)
    if returncode != 0:
        print("Compilation failed:")
        print(stderr)
        sys.exit(1)
    print("Compilation successful.")

def run_single_config(mpi_ranks, omp_threads):
    Run the program with given MPI ranks and OMP_NUM_THREADS.
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_threads)
    cmd = [MPIRUN, "-np", str(mpi_ranks), EXECUTABLE, INPUT_FILE]
    print("Running: {} with OMP_NUM_THREADS={}".format(" ".join(cmd), omp_threads))
    returncode, stdout, stderr = run_command(cmd, env)
    if returncode != 0:
        print("Error:", stderr)
        return False
    # Print first line of output (contains total time)
    if stdout:
        lines = stdout.splitlines()
        if lines:
            print(lines[0])
    return True

def collect_data():
    Run all configurations and collect timing data into CSV.
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)

    # Serial first
    run_single_config(1, 1)

    # Other OpenMP thread counts
    for omp in threads_list[1:]:
        run_single_config(1, omp)

    # Optional: distributed runs (uncomment if needed)
    # if os.path.exists("sources.txt"):
    #     print("\nRunning distributed benchmarks...")
    #     for omp in [8, 16]:
    #         for mpi in [32, 64]:
    #             env = os.environ.copy()
    #             env["OMP_NUM_THREADS"] = str(omp)
    #             cmd = [MPIRUN, "-np", str(mpi), "--hostfile", "sources.txt",
    #                    EXECUTABLE, INPUT_FILE]
    #             print("Running: {}".format(" ".join(cmd)))
    #             returncode, stdout, stderr = run_command(cmd, env)
    #             if returncode == 0 and stdout:
    #                 lines = stdout.splitlines()
    #                 if lines:
    #                     print(lines[0])

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print("Input file {} not found. Generate it using input_file_maker first.".format(INPUT_FILE))
        sys.exit(1)

    if not os.path.exists(MPICXX):
        print("MPI compiler not found at {}".format(MPICXX))
        sys.exit(1)

    compile_program()
    print("\nStarting benchmark suite...")
    collect_data()
    print("\nBenchmark complete. Timing data saved to {}".format(CSV_FILE))"""
