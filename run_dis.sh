#!/bin/bash
#SBATCH --job-name=dis_job_array
#SBATCH --nodes=1
#SBATCH --time=0:20:00
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a4500:1
# Loop through values 0..4
for i in {0..4}
do
    echo "Starting run with i=$i"
    srun --exclusive -N1 -n1 python3 dissociation.py $i &
done

wait
echo "All runs finished."