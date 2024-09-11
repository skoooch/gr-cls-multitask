# gr-cls-multitask
Multi-tasking model for grasping and classification (in progress)

you can see the model architecture in [multi_task_models/grcn_multi_alex.py](multi_task_models/grcn_multi_alex.py) 
# shapley instructions
Log into psych cluster 

navigate to `Desktop/ewan/gr-cls-multitask/`

All python packages should already be installed on cluster (if not, install -r requirements.txt)

Edit the glabal values `LAYER` and `TASK` based on what you want to run shapley on. Currently these values are set to `features.0` and `cls`.
 
run `tmux` to start a tmux session (it will let you close the window and keep the program running \[does the same thing as `screen`\])

run the command `srun -p gpu --gpus=1 --mem=\[X\] python3 shapley_cb_run.py cuda \[Y\]`

X = amount of memory you want to allocate. As of right now, the cluster's gpu partition is fully in use, but you can decide this number based on how much ram is available

Y = name of the parallel instance for this layer/task combo. **Important**: this value **must be unique**. If there already exists an h5py file with the same instance name (this can be checked by looking value after the last underscore on the h5py files in a give layer/task's folder with the `shap` folder) it will be overwritten.

To exit the tmux session, press `Ctrl+b` then type d.

To reattach, simply type `tmux attach`.

To run the shapley analysis, (ie make the plot) 

## multiAlexMap_top5_v1.5
Size of divergent heads: 4 layers

Weighted Loss Ratio (Grasp : Classification): 1.5 : 0.5 

Epochs: 150

Batch Size: 5

Grasp Accuracies - Training: 83.65 - Test: 81.5

Classification Accuracies - Training: 99.02 - Test: 85.0

## multiAlexMap_top5_v1.4
Size of divergent heads: 4 layers

Weighted Loss Ratio (Grasp : Classification): 0.5 : 1.5 

Epochs: 150

Batch Size: 5

Grasp Accuracies - Training: 77.9 - Test: 75.5

Classification Accuracies - Training: 97.98 - Test: 84.5

## multiAlexMap_top5_v1.3
Size of divergent heads: 4 layers

Epochs: 130

Batch Size: 5

Grasp Accuracies - Training: 79.95 - Test: 79.5

Classification Accuracies - Training: 98.17 - Test: 82.75


## multiAlexMap_top5_v1.2
Size of divergent heads: 1 layer

Epochs: 150

Batch Size: 2

Grasp Accuracies - Training: 72.4 - Test: 67.0

Classification Accuracies - Training: 98.53 - Test: 89.25

## multiAlexMap_top5_v1.1
Size of divergent heads: 1 layer

Epochs: 150

Batch Size: 5

Grasp Accuracies - Training: 72.22 - Test: 75.75

Classification Accuracies - Training: 98.5 - Test: 82.75




