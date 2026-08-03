# HPC Programming Coursework Repository

This repository contains a collection of High Performance Computing assignments and experiments organized by assignment number. The codebase is primarily written in C/C++ and focuses on performance benchmarking, numerical kernels, interpolation workflows, particle motion, and parallel execution.

## Project Purpose

This repository is a coursework archive for learning and practicing HPC concepts through practical C/C++ implementations. The main goal is to explore how scientific computing problems are structured, how performance is measured, and how serial execution can be improved through parallel approaches such as OpenMP and MPI.

## Assignment-by-Assignment Structure Summary

The workspace is divided into the following assignment folders:

- Assignment-1
  - Serial benchmark code for vector-triad style performance experiments
  - Data and result artifacts under the data folders

- Assignment-2
  - Matrix multiplication benchmark implementation
  - Data folders for experiment inputs and outputs

- Assignment-3
  - Serial interpolation on scattered data
  - Input generator and output mesh generation workflow

- Assignment-4
  - Particle interpolation and mover experiments
  - OpenMP-based experiment folders and result outputs

- Assignment-5
  - Multiple experiment approaches for interpolation and mover studies
  - Separate code and results folders for comparison

- Assignment-6
  - Serial and parallel implementations
  - Performance plots, result datasets, and benchmarking artifacts

- Assignment-7
  - Interpolation pipeline with particle mover logic
  - Input generation, output files, and figure generation scripts

- Assignment-8
  - Hybrid MPI/OpenMP implementation style
  - Code, dataset folders, and result outputs

## Common Project Structure

Across the assignments, the source organization is largely consistent:

- `main.cpp` contains the main driver and experiment loop
- `init.cpp` / `init.h` handle initialization and data setup
- `utils.cpp` / `utils.h` contain helper routines and numerical kernels
- `input_file_maker.cpp` appears in some assignments for generating binary input files
- `data_cluster`, `data_lab`, `data_serial`, and similar folders contain the experiment data
- `results/` or `Results/` folders store generated outputs, figures, and summaries

## Typical Build and Run Pattern

Most assignments are compiled independently from their local folder using standard C/C++ tools.

### Serial build

```bash
g++ main.cpp init.cpp utils.cpp -lm -o main
./main
```

### OpenMP build

```bash
g++ main.cpp init.cpp utils.cpp -fopenmp -o main
./main
```

### MPI-enabled build

```bash
mpic++ main.cpp init.cpp utils.cpp -o main
```

## Usage Notes

- This repository is organized as a coursework archive rather than a single monolithic application.
- Each assignment folder is generally self-contained and can be compiled and run independently.
- The code repeatedly demonstrates the same HPC workflow:
  1. initialize data
  2. run the numerical kernel
  3. measure time
  4. store or compare results
- Use the assignment-specific README files as the source of truth for exact input files, expected outputs, and experiment configurations.

## Summary

This repository documents a progression from basic serial performance experiments to more advanced interpolations, particle-mover workflows, and parallel/distributed HPC implementations. It is best understood as a collection of assignment-based scientific computing experiments with supporting data and result artifacts.
