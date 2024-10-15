#!/bin/bash

#SBATCH --time=6:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --job-name=big_batch
#SBATCH --output=job_output/sinexp_test.out
#SBATCH --error=job_output/sinexp_test.err
# comment SBATCH --output=job_output/train_%A_%a.out
# comment SBATCH --error=job_output/train_%A_%a.err
# comment SBATCH --array=0-9
#SBATCH --mem-per-cpu=16384M   # memory per CPU core
#SBATCH --qos=dw87

data=("scaled" "normalized")
model_types=("aspp_cnn" "cnn_ae")
lr_schedules=("poly" "sinexp")
batch_sizes=(32 64)


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
python -m dimensionless_wildfire.training.train --data "normalized" --model "cnn_ae" --accelerator "cpu" --lr_schedule "sinexp"