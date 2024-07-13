import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import sys

from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3
from data_processing.data_loader_v2 import DataLoader
from utils.parameters import Params

params = Params()
activation = {}

def get_model(model_path, device=params.DEVICE):
    model = Multi_AlexnetMap_v3().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


LAYER = 'rgb_features.0'
DEVICE = sys.argv[1]
MODEL_NAME = params.MODEL_NAME
MODEL_PATH = params.MODEL_WEIGHT_PATH
model = get_model(MODEL_PATH, DEVICE)

model.rgb_features[0].register_forward_hook(get_activation(LAYER))

data_loader = DataLoader(params.TEST_PATH, params.BATCH_SIZE, params.TRAIN_VAL_SPLIT)
for i, (img, cls_map, label) in enumerate(data_loader.load_batch()):
    model(img, is_grasp=False)
    print(activation[LAYER])
    break