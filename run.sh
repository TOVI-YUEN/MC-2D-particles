#!/usr/bin/env bash
# Simple pipeline:
#   1) Compile MC.c with OpenMP and optimizations
#   2) Run MC (CPU) to generate mc_final.bin
#   3) Run MC_GPU.py (GPU HMC + visualization)

set -euo pipefail

SRC_C="MC.c"
EXE_MC="MC"
PY_GPU="MC_GPU.py"
BIN_OUT="mc_final.bin"

# Optional: set number of OpenMP threads
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

echo "=== Step 1: Compile ${SRC_C} -> ${EXE_MC} ==="
if [[ ! -f "${SRC_C}" ]]; then
    echo "Error: ${SRC_C} not found in current directory."
    exit 1
fi

# NOTE: add -lm for exp(), sqrt(), etc.
gcc "${SRC_C}" -O3 -march=native -ffast-math -fopenmp -o "${EXE_MC}" -lm

echo "=== Step 2: Run CPU MC (${EXE_MC}) ==="
./"${EXE_MC}"

if [[ ! -f "${BIN_OUT}" ]]; then
    echo "Error: ${BIN_OUT} was not created. Check MC.c / MC program."
    exit 1
fi

echo "=== Step 3: Run GPU HMC (${PY_GPU}) ==="
if [[ ! -f "${PY_GPU}" ]]; then
    echo "Error: ${PY_GPU} not found in current directory."
    exit 1
fi

python3 "${PY_GPU}"

echo "=== Done. ==="

