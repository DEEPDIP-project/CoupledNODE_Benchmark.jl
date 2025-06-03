#!/bin/bash
#SBATCH --job-name=Plots
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --array=1-1

module load 2023
module load juliaup/1.14.5-GCCcore-12.3.0

echo "Slurm job ID: $SLURM_JOB_ID"
echo "Slurm array task ID: $SLURM_ARRAY_TASK_ID"

julia --project -t auto benchmark.jl
