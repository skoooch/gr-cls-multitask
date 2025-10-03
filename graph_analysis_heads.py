
def get_shapley_graph():
    
    return


def get_activation_graph():
    
    return
import pickle
#from graph_analysis_shapley import normalize_edge_weights
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
import numpy as np
import os
from scipy.stats import linregress

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
    ('features.10', model.features[10]),
    ('cls.0', model.cls[0]),
    ('grasp.0', model.grasp[0]),
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
def build_connectivity_graph(graph, model, norm=False):
    for layer_idx in range(len(layers) - 1):
        name_X, layer_X = layers[0]
        name_Y, layer_Y = layers[layer_idx + 1]
        normalized_weights = torch.mean(layer_Y.weight.data.clone(),dim=(2,3))

        # normalized_weights -= normalized_weights.min(1, keepdim=True)[0]
        # normalized_weights /= normalized_weights.max(1, keepdim=True)[0]
        print(normalized_weights.shape)
        for i in range(64):
            for j in range(32):
                # print(layer_X.weight.data.shape)
                # print(layer_Y.weight.data.shape)
                weight = normalized_weights[i:i+1, j:j+1 ].clone()
                src = f"{name_X}_k{i}"
                tgt = f"{name_Y}_k{j}"
                if weight:
                   
                    graph.add_edge(src, tgt, weight=weight.detach().to('cpu')[0][0].item())
                
    return graph


data_loader = DataLoader(params.TEST_PATH, 1, params.TRAIN_VAL_SPLIT)
graph = nx.DiGraph()
shap_values = np.load("shap_arrays/shap_values.npy")
shap_min = shap_values.min(axis=1, keepdims=True)
shap_max = shap_values.max(axis=1, keepdims=True)
shap_norm = (shap_values - shap_min) / (shap_max - shap_min + 1e-8)
shap_norm = shap_norm[4]
print(shap_norm)

build_connectivity_graph(graph, model)

cls_edges = [(src, tgt, data) for src, tgt, data in graph.edges(data=True) if 'cls' in tgt]
cls_edges_sorted = sorted(cls_edges, key=lambda x: int(x[0].split('_k')[-1]))
src_nodes = []
avg_out_weights = []
for src, tgt, data in cls_edges_sorted:
    print(f"{src} → {tgt}, weight = {data['weight']:.4f}")
    # Compute average outgoing weight for each src node
    for src, _, _ in cls_edges_sorted:
        if src not in src_nodes:
            src_nodes.append(src)
for src in src_nodes:
    out_edges = graph.out_edges(src, data=True)
    if out_edges:
        avg_weight = np.mean([abs(data['weight']) for _, tgt, data in out_edges if 'cls' in tgt])
        avg_out_weights.append((src, avg_weight))
    else:
        avg_out_weights.append((src, 0.0))
# Create array of just the average weights (order matches src_nodes iteration)
cls_avg_out_weights_array = np.array([w for _, w in avg_out_weights])

#############
grasp_edges = [(src, tgt, data) for src, tgt, data in graph.edges(data=True) if 'grasp' in tgt]
grasp_edges_sorted = sorted(grasp_edges, key=lambda x: int(x[0].split('_k')[-1]))
src_nodes = []
for src, tgt, data in grasp_edges_sorted:
    print(f"{src} → {tgt}, weight = {data['weight']:.4f}")
    # Compute average outgoing weight for each src node
    
    for src, _, _ in grasp_edges_sorted:
        if src not in src_nodes:
            src_nodes.append(src)
avg_out_weights = []
for src in src_nodes:
    out_edges = graph.out_edges(src, data=True)
    if out_edges:
        avg_weight = np.mean([abs(data['weight']) for _, tgt, data in out_edges if 'grasp' in tgt])
        avg_out_weights.append((src, avg_weight))
    else:
        avg_out_weights.append((src, 0.0))
# Create array of just the average weights (order matches src_nodes iteration)
grasp_avg_out_weights_array = np.array([w for _, w in avg_out_weights])

print("Average outgoing weights array:", cls_avg_out_weights_array)
print("Average outgoing weights array:", grasp_avg_out_weights_array)
print(pearsonr(cls_avg_out_weights_array, shap_norm[:, 0]))
print(pearsonr(cls_avg_out_weights_array, shap_norm[:, 1]))
print(pearsonr(grasp_avg_out_weights_array, shap_norm[:, 0]))
print(pearsonr(grasp_avg_out_weights_array, shap_norm[:, 1]))
import matplotlib.pyplot as plt


def scatter_with_fit(ax, x, y, title, xlabel, ylabel):
    ax.scatter(x, y)
    # Line of best fit
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    x_fit = np.linspace(np.min(x), np.max(x), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color='red', linestyle='--')
    ax.set_title(f"{title}\n(r={r_value:.3g}, p={p_value:.3g})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

scatter_with_fit(axs[0, 0], cls_avg_out_weights_array, shap_norm[:, 0],
                 'Average Weight Magnitude to CLS Head vs Classification SHAP', 'CLS avg out weights', 'Normalized Classification Shapley Values')
scatter_with_fit(axs[0, 1], cls_avg_out_weights_array, shap_norm[:, 1],
                 'Average Weight Magnitude to CLS Head vs Grasping SHAP', 'CLS avg out weights', 'Normalized Grasping Shapley Values')
scatter_with_fit(axs[1, 0], grasp_avg_out_weights_array, shap_norm[:, 0],
                 'Average Weight Magnitude to Grasp Head vs Classificiation SHAP', 'Grasp avg out weights', 'Normalized Classification Shapley Values')
scatter_with_fit(axs[1, 1], grasp_avg_out_weights_array, shap_norm[:, 1],
                 'Average Weight Magnitude to Grasp Head vs Grasping SHAP', 'Grasp avg out weights', 'Normalized Grasping Shapley Values')

plt.tight_layout()
plt.savefig("test.png")

# edges = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'])
# for src, tgt, data in edges[:-10]:
#     print(f"{src} → {tgt}, weight = {data['weight']:.4f}")
pickle.dump(graph, open('graphs/head_weights.pickle', 'wb'))
