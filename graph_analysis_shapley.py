
from utils.parameters import Params
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3  
from data_processing.data_loader_v2 import DataLoader  
import torch
import torch.nn.functional as F
import networkx as nx
import os
import numpy as np
params = Params()
SEED=42

model_name = params.MODEL_NAME
weights_dir = params.MODEL_PATH
weights_path = os.path.join(weights_dir, model_name, model_name + '_final.pth')
model =  Multi_AlexnetMap_v3().to('cpu')
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

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

def normalize_edge_weights(graph, attr_name='weight'):
    """Normalizes edge weights in a NetworkX graph to the range [0, 1].

    Args:
        graph: A NetworkX graph.
        attr_name: The name of the edge attribute to normalize (default 'weight').
    """
    # Extract all weights from the graph
    weights = [data[attr_name] for _, _, data in graph.edges(data=True) if attr_name in data]
    
    if not weights:
        return  # No weights to normalize

    max_weight = max(weights)
    min_weight = min(weights)

    # Avoid division by zero
    if max_weight == min_weight:
        for u, v, data in graph.edges(data=True):
            if attr_name in data:
                data[attr_name] = 0.0  # or 1.0 depending on your preference
    else:
        for u, v, data in graph.edges(data=True):
            if attr_name in data:
                data[attr_name] = (data[attr_name] - min_weight) / (max_weight - min_weight)
            
# Main function to compute kernel connectivity graph
def build_connectivity_graph(graph, model):
    shap_values = np.load("shap_arrays/shap_values.npy")

    # Normalize shapley values to [0, 1] for each layer and channel separately
    shap_min = shap_values.min(axis=1, keepdims=True)
    shap_max = shap_values.max(axis=1, keepdims=True)
    shap_norm = (shap_values - shap_min) / (shap_max - shap_min + 1e-8)
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
                if (i == 13 and layer_idx == 2):
                    print((shap_norm[layer_idx, i,0] - shap_norm[layer_idx, i,1]) * (shap_norm[layer_idx + 1, j,0] - shap_norm[layer_idx + 1, j,1]))
                if weight:
                    graph.add_edge(src, tgt, weight=(shap_norm[layer_idx, i,0] - shap_norm[layer_idx, i,1]) * (shap_norm[layer_idx + 1, j,0] - shap_norm[layer_idx + 1, j, 1]))

    return graph
graphs = []
data_loader = DataLoader(params.TEST_PATH, 1, params.TRAIN_VAL_SPLIT)
graph = nx.DiGraph()
build_connectivity_graph(graph, model)
# normalize_edge_weights(graph)
edges = sorted(graph.edges(data=True), key=lambda x: -abs(x[2]['weight']))
# for src, tgt, data in edges[:20]:
#     print(f"{src} → {tgt}, weight = {data['weight']:.4f}")
# pickle.dump(graph, open('graphs/shap_graph_unnormalized.pickle', 'wb'))
