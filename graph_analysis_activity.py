import pickle
from utils.parameters import Params
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3  
from data_processing.data_loader_v2 import DataLoader  
import torch
import torch.nn.functional as F
from torch import nn
import networkx as nx
from tqdm import tqdm
import sys
import os
from graph_analysis_shapley import normalize_edge_weights
from own_images import load_images_to_arrays
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
def build_connectivity_graph(graph, model, input_tensor, weights=False):
    activations = get_activations(input_tensor, layers)
    for layer_idx in range(len(layers) - 1):
        name_X, layer_X = layers[layer_idx]
        name_Y, layer_Y = layers[layer_idx + 1]
        fmap_X = activations[layer_idx]  # Shape: [1, Cx, H, W]
        fmap_Y = activations[layer_idx + 1]  # Shape: [1, Cy, H, W]
        Cx = fmap_X.shape[1]
        Cy = fmap_Y.shape[1]
        for i in range(Cx):
            fmap_Xi = fmap_X[:, i:i+1]  # shape: [1, 1, H, W]
            for j in range(Cy):
                # Convolve fmap_Xi with kernel j from layer_Y
                kernel_Y_j = nn.Conv2d(1, 1, kernel_size=layer_Y.kernel_size, padding=layer_Y.padding).to(device)
                kernel_Y_j.weight.data = layer_Y.weight.data[j:j+1, i:i+1].clone()
                kernel_Y_j.bias.data.zero_()
                response = F.relu(compute_kernel_similarity(kernel_Y_j, fmap_Xi), inplace=True)  # shape: [1, 1, H, W]
                weight=torch.norm(response)
                src = f"{name_X}_k{i}"
                tgt = f"{name_Y}_k{j}"
                graph.add_edge(src, tgt, weight=weight)
    return graph
def edit_connectivity_graph(graph, model, input_tensor, n, weights = False):
    activations = get_activations(input_tensor, layers)
    for layer_idx in range(len(layers) - 1):
        name_X, layer_X = layers[layer_idx]
        name_Y, layer_Y = layers[layer_idx + 1]
        
        fmap_X = activations[layer_idx]  # Shape: [1, Cx, H, W]
        fmap_Y = activations[layer_idx + 1]  # Shape: [1, Cy, H, W]
        
        Cx = fmap_X.shape[1]
        Cy = fmap_Y.shape[1]

        for i in range(Cx):
            fmap_Xi = fmap_X[:, i:i+1]  # shape: [1, 1, H, W]
            for j in range(Cy):
                # Convolve fmap_Xi with kernel j from layer_Y
                kernel_Y_j = nn.Conv2d(1, 1, kernel_size=layer_Y.kernel_size, padding=layer_Y.padding).to(device)
                kernel_Y_j.weight.data = layer_Y.weight.data[j:j+1, i:i+1].clone()
                kernel_Y_j.bias.data.zero_()
                response = F.relu(compute_kernel_similarity(kernel_Y_j, fmap_Xi), inplace=True)  # shape: [1, 1, H, W]
                weight=torch.norm(response)
                src = f"{name_X}_k{i}"
                tgt = f"{name_Y}_k{j}"
                graph[src][tgt]['weight'] = graph[src][tgt]['weight']*(n-1)/n + weight/n

# Build the graph
graphs = []
our_images = True

data_loader = DataLoader(params.TEST_PATH, 1, params.TRAIN_VAL_SPLIT)
graph = nx.DiGraph()
with tqdm(total=400, dynamic_ncols=True, file=sys.stdout) as pbar:
    if our_images:
        images = load_images_to_arrays(depth=True)
        for i, img in enumerate(images):
            if i == 0: build_connectivity_graph(graph, model, img)
            else: edit_connectivity_graph(graph, model, img, i)
            pbar.update(1)
            # if i == 10:
            #     break
    else:
        for i, (img, cls_map, label) in enumerate(data_loader.load_cls()):
            if i == 0: build_connectivity_graph(graph, model, img)
            else: edit_connectivity_graph(graph, model, img, i)
            pbar.update(1)
            # if i == 10:
            #     break
edges = sorted(graph.edges(data=True), key=lambda x: -abs(x[2]['weight']))
#normalize_edge_weights(graph)
# for src, tgt, data in edges[:20]:
#     print(f"{src} → {tgt}, weight = {data['weight']:.4f}")
pickle.dump(graph, open('graphs/activity.pickle', 'wb'))
