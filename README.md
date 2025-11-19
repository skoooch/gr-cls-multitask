# gr-cls-multitask

This repository contains code for the publication: 

[Insert publication title, authors, and link].

The relvant human and neural network data can be accessed: [put link here]

This code implements [insert blurb about project purpose, like an abstract]. Multi-tasking model for grasping and classification.

Model architecture: [multi_task_models/grcn_multi_alex.py](multi_task_models/grcn_multi_alex.py) 

## Table of Contents
* [Prerequisites](#prerequisites)
* [Dataset](#dataset-generationpreprocessing)
* [Model Training](#model-training)
* [Neuron Shapely](#neuron-shapley)
* [Graph](#graph-to-make-a-more-explanatory-name)
* [MATLAB](#matlab-code)
* [Figures](#figuresplots)
* [Results](#results)
* [Citations](#citations)



## Prerequisites
Python version
All neural networks are implemented in pytorch.

create venv ```python -m venv .venv```

install requirements running ```pip install -r requirements.txt```

## Dataset Generation/Preprocessing

To generate/preprocess images used in this work run `datasets/FOLDER/FILE`

[Explain briefly here what the preprocessing is doing, what it takes in, what it converts to]

```python data_processing/data_preprocess.py```

*What about our own images, and `new_data` folders?*

## Model Training

**NOTE IF ANY COMMAND LINE ARGUMENTS ARE NEEDED**
There is both single-task and multi-task model in this repository. To train the single-task model, run `python single_task\train.py`. 

For multi-task model, run `multi_train.py`. 

* make any notes about pre-existing configs and how user can change/customize them.
* note about hyperparameters
* note about where output/weights are stored

`trained-models/`

## Neuron Shapely
Descibr describe describe.

`shapley_analysis.py`: Calculates and plots correlation and dist

`get_top_shapley.py`:

`shap/`:

`shap_arrays/`:

## Graph (to make a more explanatory name)

`graph_analysis_shapley.py`: exxplain briefly

`graph_analysis_weights.py`: explain briefly


## (MATLAB CODE)
The PsychToolbox code to run the grasping and classification experiment can be found in
the 'FOLDER_NAME' folder.

## Figures/Plots
Code to prepare the figure panels can be found in the 'plot' folder. (need to move around the script to cretae this folder, or just provide diff instructions in readme for plotting?)

`visualize_filters.py`

## Results

| Model Variant              | Div. Heads | Epochs | Batch Size | Loss Ratio (Grasp:Class) | Grasp Acc (Train/Test) | Class Acc (Train/Test) |
|---------------------------|------------|--------|------------|---------------------------|------------------------|------------------------|
| multiAlexMap_top5_v1.5    | 4 layers   | 150    | 5          | 1.5 : 0.5                 | 83.65 / 81.5           | 99.02 / 85.0           |
| multiAlexMap_top5_v1.4    | 4 layers   | 150    | 5          | 0.5 : 1.5                 | 77.9 / 75.5            | 97.98 / 84.5           |
| multiAlexMap_top5_v1.3    | 4 layers   | 130    | 5          | —                         | 79.95 / 79.5           | 98.17 / 82.75          |
| multiAlexMap_top5_v1.2    | 1 layer    | 150    | 2          | —                         | 72.4 / 67.0            | 98.53 / 89.25          |
| multiAlexMap_top5_v1.1    | 1 layer    | 150    | 5          | —                         | 72.22 / 75.75          | 98.5 / 82.75           |


## Citations
Enter citations and/or acknowledgements here
