# gr-cls-multitask
Multi-tasking model for grasping and classification (in progress)

you can see the model architecture in [multi_task_models/grcn_multi_alex.py](multi_task_models/grcn_multi_alex.py) 


## multiAlexMap_top5_v1.5
Task| Recogniton | Grasping
--- | --- | --- 
Train Accuracy (%) | 99.02 | 83.65 
Test Accuracy (%) | 85.0| 81.5 
Learning Rate | 301 | 283 
Epoch | 150 | 150 

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




