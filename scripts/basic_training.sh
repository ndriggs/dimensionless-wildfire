#!/bin/bash

#SBATCH --time=6:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --output=job_output/nondim_test.out
#SBATCH --error=job_output/nondim_test.err
#SBATCH --mem-per-cpu=16384M   # memory per CPU core
#SBATCH --qos=dw87


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
python -m dimensionless_wildfire.training.train --data "nondim" --nondim_setup "keep"