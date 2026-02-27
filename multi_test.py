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
from utils.parameters import Params
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3
from training_utils.evaluation import get_cls_acc, get_grasp_acc, visualize_grasp, visualize_cls

params = Params() 
SEED = params.SEED

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

MODEL_NAME = params.MODEL_NAME_SEED
MODEL_PATH = params.MODEL_WEIGHT_PATH_SEED
model = Multi_AlexnetMap_v3().to("cuda")
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
path = params.TEST_PATH
if len(sys.argv) > 1:
    path = params.TEST_PATH_SHUFFLE

# Get test acc for CLS model
c_accuracy, c_loss = get_cls_acc(model, include_depth=True, seed=None, dataset=path, truncation=None)
# Get test acc for Grasp model
accuracy, loss = get_grasp_acc(model, include_depth=True, seed=None, dataset=path, truncation=None)

print('Grasp: %s' % accuracy, loss)
print('CLS: %s' % c_accuracy, c_loss)

# Visualize CLS predictions one by one
#visualize_cls(model)
# Visualize grasp predictions one by one
#visualize_grasp(model)
