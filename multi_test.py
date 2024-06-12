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

from utils.parameters import Params
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3
from training.single_task.evaluation import get_cls_acc, get_grasp_acc, visualize_grasp, visualize_cls

params = Params()

model_name = params.MODEL_NAME
weights_dir = params.MODEL_PATH
weights_path = os.path.join(weights_dir, model_name, model_name + '_epoch131.pth')

# AlexNet with 1st, 2nd layer pretrained on Imagenet
model =  Multi_AlexnetMap_v3().to('cuda')
model.load_state_dict(torch.load(weights_path))
model.eval()
# Get test acc for CLS model
c_accuracy, c_loss = get_cls_acc(model, include_depth=True, seed=None, dataset=params.TRAIN_PATH, truncation=None)
# Get test acc for Grasp model
accuracy, loss = get_grasp_acc(model, include_depth=True, seed=None, dataset=params.TRAIN_PATH, truncation=None)

print('Grasp: %s' % accuracy, loss)
print('CLS: %s' % c_accuracy, c_loss)

# Visualize CLS predictions one by one
#visualize_cls(model)
# Visualize grasp predictions one by one
#visualize_grasp(model)
