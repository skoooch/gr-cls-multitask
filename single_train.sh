#!/bin/bash
#SBATCH --job-name=55
#SBATCH --output=shap_log/shapley_%A_%a.out
#SBATCH --error=shap_log/shapley_%A_%a.err
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
srun python3 single_train.py cuda 