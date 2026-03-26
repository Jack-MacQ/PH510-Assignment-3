#!/bin/bash

#======================================================
# Job script for Task 2: parallel Green's function MC
#======================================================
#SBATCH --export=ALL
#SBATCH --partition=teaching
#SBATCH --account=teaching
#SBATCH --ntasks=16
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --job-name=task2
#SBATCH --output=green_function-%j.out

module purge
module load openmpi/latest

/opt/software/scripts/job_prologue.sh

# Run the Green's function computation
mpirun -np $SLURM_NPROCS python3 task2script.py

/opt/software/scripts/job_epilogue.sh
