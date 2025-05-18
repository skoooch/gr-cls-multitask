
def get_shapley_graph():
    
    return


def get_activation_graph():
    
    return
import pickle
from utils.parameters import Params
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3  
from data_processing.data_loader_v2 import DataLoader  
import torch
import torch.nn.functional as F
from torch import nn
import networkx as nx
from scipy.stats import pearsonr, spearmanr
from torchvision.models import alexnet
from tqdm import tqdm
import sys
import copy
import os
params = Params()
SEED=42

model_name = params.MODEL_NAME
weights_dir = params.MODEL_PATH
weights_path = os.path.join(weights_dir, model_name, model_name + '_final.pth')
model =  Multi_AlexnetMap_v3().to('cuda')
model.load_state_dict(torch.load(weights_path))
model.eval()

# Move to CPU or GPU as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Dummy input
dummy_input = torch.randn(1, 4, 64, 64).to(device)  # 4-channel (RGB + D)

# Extract layers of interest
layers = [
    ('rgb_features.0', model.rgb_features[0]),
    ('features.0', model.features[0]),
    ('features.4', model.features[4]),
    ('features.7', model.features[7]),
    ('features.10', model.features[10]),
]

# Helper to forward pass and store outputs
def get_activations(input_tensor, layers):
    activations = []
    x = input_tensor[:, :3]
    d = input_tensor[:, 3:]
    d = torch.cat((d, d, d), dim=1)
    
    x_rgb = model.rgb_features[0](x)
    x_rgb = model.rgb_features[1](x_rgb)
    x_rgb = model.rgb_features[2](x_rgb)

    x_d = model.d_features(d)
    x = torch.cat((x_rgb, x_d), dim=1)

    out = x
    idx = 0
    for i, layer in enumerate(model.features):
        out = layer(out)
        if f'features.{i}' in dict(layers):
            if i == 0: activations.append(F.relu(F.max_pool2d(out, 3, 2), inplace=True).clone().detach())
            else: activations.append(F.relu(out, inplace=True).clone().detach())
            idx += 1  
    return [F.relu(F.max_pool2d(x_rgb, 3, 2), inplace=True)] + activations

# Compute cosine similarity between two feature maps
def compute_kernel_similarity(kernel_Y, fmap_X):
    out = F.conv2d(fmap_X, kernel_Y.weight.data, bias=None, stride=1, padding=kernel_Y.padding)
    return out

# Main function to compute kernel connectivity graph
def build_connectivity_graph(graph, model):
    for layer_idx in range(len(layers) - 1):
        name_X, layer_X = layers[layer_idx]
        name_Y, layer_Y = layers[layer_idx + 1]
        normalized_weights = torch.mean(layer_Y.weight.data.clone(),dim=(2,3))
        normalized_weights -= normalized_weights.min(1, keepdim=True)[0]
        normalized_weights /= normalized_weights.max(1, keepdim=True)[0]
        
        for i in range(layer_X.weight.data.shape[0]):
            for j in range(layer_Y.weight.data.shape[0]):
                weight = normalized_weights[j:j+1, i:i+1].clone()
                src = f"{name_X}_k{i}"
                tgt = f"{name_Y}_k{j}"
                if weight:
                    graph.add_edge(src, tgt, weight=weight.detach().to('cpu')[0][0].item())

    return graph
graphs = []
data_loader = DataLoader(params.TEST_PATH, 1, params.TRAIN_VAL_SPLIT)
graph = nx.DiGraph()
build_connectivity_graph(graph, model)

edges = sorted(graph.edges(data=True), key=lambda x: abs(x[2]['weight']))
for src, tgt, data in edges[:10]:
    print(f"{src} → {tgt}, weight = {data['weight']:.4f}")
pickle.dump(graph, open('just_weights_mean.pickle', 'wb'))
