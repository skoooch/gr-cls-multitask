
import torch
from utils.parameters import Params

LAYERS = ['first','features.0','features.4', 'features.7', 'features.10']
SIZES = [128,32,64,64,64]
params = Params()

MODEL_NAME = params.MODEL_NAME_SEED
MODEL_PATH = params.MODEL_WEIGHT_PATH_SEED
model_seeds = [43, 203, 301]
for i in range(5):
    LAYER = LAYERS[i]
    for temp_seed in model_seeds:
        MODEL_NAME_TEMP = MODEL_NAME.split("_")
        MODEL_NAME_TEMP[-1] = str(temp_seed)
        MODEL_NAME_TEMP = "_".join(MODEL_NAME_TEMP)
        activations_0 = torch.load(f'shap/activations/{MODEL_NAME_TEMP}_{LAYER}.pt').float()
        non_zero_count = activations_0 > 0
        
        print(f"Average activation - layer {LAYER}, seed {temp_seed}: mean = {abs(activations_0).mean()}, sparsity = {non_zero_count.sum()}")