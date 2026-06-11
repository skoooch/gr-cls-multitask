#!/bin/bash
#SBATCH --job-name=shapley_anal
#SBATCH --output=shap_log/shapley_anal_%A_%a.out
#SBATCH --error=shap_log/shapley_anal_%A_%a.err
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
srun python3 shapley_analysis.py 
