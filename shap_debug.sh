#!/bin/bash
#SBATCH --job-name=shapley_parallel
#SBATCH --output=shap_log/shapley_%A_%a.out
#SBATCH --error=shap_log/shapley_%A_%a.err
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
srun python3 shapley_debug.py cuda cls ${SLURM_ARRAY_TASK_ID} round1_saturday