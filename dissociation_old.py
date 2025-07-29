import torch
import os
import time
import sys
import matplotlib.pyplot as plt
from utils.parameters import Params
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3
from training.single_task.evaluation import get_cls_acc, get_grasp_acc, denormalize_grasp, map2singlegrasp
import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from data_processing.data_loader_v2 import DataLoader
from utils.utils import get_correct_cls_preds_from_map
from utils.grasp_utils import get_correct_grasp_preds

params = Params()
params =data_loader = DataLoader(params.TEST_PATH, 1, params.TRAIN_VAL_SPLIT)

def get_cls_dist(model, include_depth=True, seed=None, dataset=params.TEST_PATH, truncation=None, device=params.DEVICE):
    loss = 0
    correct = 0
    total = 0
    dist = np.zeros(len(data_loader.load_grasp()))
    if(dataset == params.TEST_PATH):
        for i, (img, cls_map, label) in enumerate(data_loader.load_cls()):
            if truncation is not None and (i * params.BATCH_SIZE / data_loader.n_data) > truncation:
                break
            output = model(img, is_grasp=False,dissociate=dissociate)
            batch_correct, batch_total = get_correct_cls_preds_from_map(output, label)
            correct += batch_correct
            total += batch_total
            dist[i] = batch_correct*100.0
    return dist        
def get_grasp_dist(model, include_depth=True, seed=None, dataset=params.TEST_PATH, truncation=None, device=params.DEVICE):
    """Returns the test accuracy and loss of a Grasp model."""
    loss = 0
    correct = 0
    total = 0
    dist = np.zeros(data_loader.load_grasp())
    if(dataset == params.TEST_PATH):
        for i, (img, map, candidates) in enumerate(data_loader.load_grasp()):
            if truncation is not None and (i * params.BATCH_SIZE / data_loader.n_data) > truncation:
                break
            output = model(img, is_grasp=True,dissociate=dissociate)
            # Move grasp channel to the end
            output = torch.moveaxis(output, 1, -1)
            # Denoramlize grasps
            denormalize_grasp(output)

            # Convert grasp map into single grasp prediction
            output_grasp = map2singlegrasp(output)
            output_grasp = torch.unsqueeze(output_grasp, dim=1).repeat(1, candidates.shape[0], 1)
            batch_correct, batch_total =  get_correct_grasp_preds(output_grasp, candidates[None, :, :]) #get_correct_grasp_preds_from_map(output, map)
            correct += batch_correct
            total += batch_total
            dist[i] = batch_correct*100.0
    return dist
SEED=42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
model_name = params.MODEL_NAME
weights_dir = params.MODEL_PATH
weights_path = os.path.join(weights_dir, model_name, model_name + '_final.pth')
# AlexNet with 1st, 2nd layer pretrained on Imagenet
model =  Multi_AlexnetMap_v3().to('cuda')
model.load_state_dict(torch.load(weights_path))
model.eval()
path = params.TEST_PATH
if len(sys.argv) > 1:
    path = params.TEST_PATH_SHUFFLE
    
    
###-------- Change these values for different layer and anaysis -----------
diff = True
layer = 4
load_data = True
### ------------------------------

if diff: top = torch.tensor(np.load("shap_arrays/sort_shap_indices_diff_normalized.npy"), dtype=int)
else: top = torch.tensor(np.load("shap_arrays/sort_shap_indices.npy"), dtype=int)

print(top.shape)
print(top[0, :, 0])
data_length = 16

if not load_data:
    data = np.zeros((data_length, 2, 2,len(data_loader.load_grasp())), np.float16)
    for lesion_i in range(data_length):
        for i in range(2):
            dissociate = top[:,:lesion_i,i]
            dissociate[:layer, :] = 0
            dissociate[layer + 1:, :] = 0
            # Get test acc for CLS model
            c_accuracy, c_loss = get_cls_acc(model, include_depth=True, seed=None, dataset=path, truncation=None, dissociate=dissociate)
            # Get test acc for Grasp model
            accuracy, loss = get_grasp_acc(model, include_depth=True, seed=None, dataset=path, truncation=None, dissociate=dissociate)
            data[lesion_i, i, 0] = c_accuracy
            data[lesion_i, i, 1] = accuracy
    np.save(f'layer_{layer}_data.npy', data)
else:
    data = np.load(f'layer_{layer}_data.npy')
# Plot the data array as two separate plots
x = range(data.shape[0])

#### T CURVES ###########3
from scipy.stats import ttest_ind

# ...existing code for data generation...

# Calculate t-values for each curve compared to the baseline (lesion_i=0)
t_values = np.zeros_like(data)
baseline = data[0]  # shape: (2, 2)

for lesion_i in range(data_length):
    for i in range(2):  # 0: class kernels, 1: grasp kernels
        for j in range(2):  # 0: classification accuracy, 1: grasp accuracy
            # t-test: compare current accuracy to baseline
            print([data[lesion_i, i, j]])
            print(baseline[i, j])
            t_stat, _ = ttest_ind([data[lesion_i, i, j]], [baseline[i, j]], equal_var=False)
            t_values[lesion_i, i, j] = t_stat

# Plot t-value curves
plt.figure(figsize=(10, 8))
plt.plot(x, t_values[:, 0, 0], label='T-value: Classification (CLS Kernels)', linestyle='--', color='red')
plt.plot(x, t_values[:, 0, 1], label='T-value: Grasp (CLS Kernels)', linestyle='-', color='blue')
plt.plot(x, t_values[:, 1, 0], label='T-value: Classification (Grasp Kernels)', linestyle='-', color='orange')
plt.plot(x, t_values[:, 1, 1], label='T-value: Grasp (Grasp Kernels)', linestyle='--', color='green')
plt.title(f'T-value Curves for Lesioning (Layer {layer + 1})')
plt.xlabel('Lesion Index')
plt.ylabel('T-value')
plt.legend()
plt.tight_layout()
if diff:
    plt.savefig(f'vis/lesion/tvalue_curves_diff_normalized_layer_{layer}.png', dpi=300)
else:
    plt.savefig(f'vis/lesion/tvalue_curves_layer_{layer}.png', dpi=300)
plt.close()

anova_f_values = []
for lesion_i in range(data_length):
    # Prepare data for ANOVA: 4 measurements (task × lesion type)
    df = pd.DataFrame({
        'accuracy': data[lesion_i].flatten(),
        'task': ['CLS', 'CLS', 'Grasp', 'Grasp'],
        'lesion_type': ['CLS', 'Grasp', 'CLS', 'Grasp'],
        'subject': [0, 0, 0, 0]  # If you have multiple seeds, use actual subject IDs
    })
    # Run repeated-measures ANOVA
    aov = AnovaRM(df, 'accuracy', 'subject', within=['task', 'lesion_type'])
    res = aov.fit()
    # Get F-value for interaction
    f_val = res.anova_table.loc['task:lesion_type', 'F Value']
    anova_f_values.append(f_val)

# Plot F-value curve
plt.figure(figsize=(10, 8))
plt.plot(x, anova_f_values, label='F-value (Task × LesionType interaction)', color='purple')
plt.axhline(y=4.0, color='gray', linestyle='dashed', label='Significance threshold (F)')
plt.title(f'F-value Curve for Double Dissociation (Layer {layer + 1})')
plt.xlabel('Lesion Index')
plt.ylabel('F-value')
plt.legend()
plt.tight_layout()
if diff:
    plt.savefig(f'vis/lesion/fvalue_curve_diff_normalized_layer_{layer}.png', dpi=300)
else:
    plt.savefig(f'vis/lesion/fvalue_curve_layer_{layer}.png', dpi=300)
plt.close()
### ACCURACY CURVES ########

# # Third plot: Combined plot with all data
# plt.figure(figsize=(10, 8))
# plt.plot(x, data[:, 0, 0], label='Classification Accuracy (CLS Kernels)', linestyle='--', color = "red")
# plt.plot(x, data[:, 0, 1], label='Grasp Accuracy (CLS Kernels)', linestyle='-', color = "blue")
# plt.plot(x, data[:, 1, 0], label='Classification Accuracy (Grasp Kernels)', linestyle='-', color = "red")
# plt.plot(x, data[:, 1, 1], label='Grasp Accuracy (Grasp Kernels)', linestyle='--', color = "blue")
# plt.title(f'Convolutional Layer {layer + 1}')
# if diff: plt.xlabel('Lesion Index (Difference in Shapley Score)') 
# else: plt.xlabel('Lesion Index') 
# plt.ylabel('Accuracy')
# plt.legend()
# plt.tight_layout()
# if diff: plt.savefig(f'vis/lesion/combined_lesion_accuracy_diff_normalized_layer_{layer}.png',dpi=300)
# else: plt.savefig(f'vis/lesion/combined_lesion_accuracy_layer_{layer}.png')
# plt.close()
