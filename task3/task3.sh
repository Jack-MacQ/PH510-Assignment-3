#!/bin/bash

#=======================================================================
# Job script for Task 3: parallel Green's function MC at selected points
#=======================================================================
#SBATCH --export=ALL
#SBATCH --partition=teaching
#SBATCH --account=teaching
#SBATCH --ntasks=16
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --job-name=task3
#SBATCH --output=task3-%j.out

module purge
module load openmpi/gcc-14.2.1/4.1.8

/opt/software/scripts/job_prologue.sh

# Run the Green's function computation
mpirun -np $SLURM_NPROCS python3 task3.py

/opt/software/scripts/job_epilogue.sh
