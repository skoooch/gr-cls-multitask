"""
This file tests the model's performance on the testing dataset.

For CLS, this script returns the testing accuracy.
For Grasp, this script returns the testing accuracy and visualizes the
grasp prediction.


Comment or uncomment certain lines of code for swapping between
training CLS model and Grasping model.

E.g. Uncomment the lines with NO SPACE between '#' and the codes: 
# Get test acc for CLS model
#accuracy, loss = get_test_acc(model)
# Get test acc for Grasp model
accuracy, loss = get_grasp_acc(model)

----->

# Get test acc for CLS model
accuracy, loss = get_test_acc(model)
# Get test acc for Grasp model
#accuracy, loss = get_grasp_acc(model)
"""

import torch
import os
import time
import sys
import matplotlib.pyplot as plt
from utils.parameters import Params
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3
from training.single_task.evaluation import get_cls_acc, get_grasp_acc, visualize_grasp, visualize_cls
import numpy as np
params = Params()
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
layer = 1
### ------------------------------

if diff: top = torch.tensor(np.load("sort_shap_indices_diff_normalized.npy"), dtype=int)
else: top = torch.tensor(np.load("sort_shap_indices.npy"), dtype=int)

print(top.shape)
print(top[0, :, 0])
data_length = 64


data = np.zeros((data_length, 2, 2), np.float16)
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
# ...existing code...

# Plot the data array as two separate plots
x = range(data.shape[0])
# Third plot: Combined plot with all data
plt.figure(figsize=(10, 8))
plt.plot(x, data[:, 0, 0], label='Classification Accuracy (CLS Kernels)', linestyle='--', color = "coral")
plt.plot(x, data[:, 0, 1], label='Grasp Accuracy (CLS Kernels)', linestyle='-', color = "turquoise")
plt.plot(x, data[:, 1, 0], label='Classification Accuracy (Grasp Kernels)', linestyle='-', color = "coral")
plt.plot(x, data[:, 1, 1], label='Grasp Accuracy (Grasp Kernels)', linestyle='--', color = "turquoise")
plt.title(f'Combined Accuracy for Lesioning CLS and Grasp Kernels in Conv Layer {layer + 1}')
if diff: plt.xlabel('Lesion Index (Difference in Shapley Score [Normalized])') 
else: plt.xlabel('Lesion Index') 
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
if diff: plt.savefig(f'vis/lesion/combined_lesion_accuracy_diff_normalized_layer_{layer}.png')
else: plt.savefig(f'vis/lesion/combined_lesion_accuracy_layer_{layer}.png')
plt.close()
# ...existing code...    
# Visualize CLS predictions one by one
#visualize_cls(model)
# Visualize grasp predictions one by one
#visualize_grasp(model)
