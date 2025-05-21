import networkx as nx
import pickle

import networkx
from sklearn import dummy
from utils.parameters import Params
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3  
from data_processing.data_loader_v2 import DataLoader  
import torch
import matplotlib
import matplotlib.pyplot
import os
from scipy.stats import pearsonr
from scipy.linalg import eigvalsh
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
layers = ['rgb_features.0', 'features.0','features.4','features.7','features.10']
def get_refined_graphs(graph_list, add_start=False):
    shap_indices = np.load("sort_shap_indices.npy")
    refined_graph_list = []
    for graph in graph_list:
        refined_graph_list.append(nx.DiGraph())
    for layer_idx in range(len(layers) - 1):
        name_X = layers[layer_idx]
        name_Y = layers[layer_idx + 1]
        kernels_X = set()
        kernels_Y = set()
        for j, kernels in enumerate([kernels_X, kernels_Y]):
            i_c = 0
            i_g = 0
            grasp_flag = False
            while len(kernels) < 6:
                i = i_c if not grasp_flag else i_g
                if shap_indices[layer_idx + j, i, int(grasp_flag)] not in kernels:
                    kernels.add(shap_indices[layer_idx + j, i, int(grasp_flag)])
                    i_c += int(not grasp_flag)
                    i_g += int(grasp_flag)
                    grasp_flag = not grasp_flag 
                else:
                    i_c += int(not grasp_flag)
                    i_g += int(grasp_flag)
        for x in kernels_X:
            for y in kernels_Y:
                src = f"{name_X}_k{x}"
                tgt = f"{name_Y}_k{y}"
                for i, refined_graph in enumerate(refined_graph_list):
                    try:
                        refined_graph.add_edge(src, tgt, weight=graph_list[i][src][tgt]["weight"])
                        
                        if layer_idx == 0 and add_start:
                            refined_graph.add_edge("0", src, weight=0.5)
                    except: 
                        pass
    return refined_graph_list


def calculate_graph_measures(graph):
    """
    Given a directed, weighted graph of kernel connectivity, compute graph-theoretic measures.
    
    Returns:
        pandas.DataFrame with node-wise metrics.
    """
    # Ensure weights are used where appropriate
    inverted_graph = nx.DiGraph()
    edges = sorted(graph.edges(data=True), key=lambda x: -abs(x[2]['weight']))
    for src, tgt, data in edges:
        inverted_graph.add_edge(src, tgt, weight=1-graph[src][tgt]['weight'])
    betweenness = nx.betweenness_centrality(inverted_graph, weight='weight', normalized=True)
    eigenvector = nx.eigenvector_centrality(graph, max_iter=1000, weight='weight')
    pagerank = nx.pagerank(graph, weight='weight')
    in_degree = dict(graph.in_degree(weight='weight'))
    out_degree = dict(graph.out_degree(weight='weight'))

    # Combine all into a dataframe
    df = pd.DataFrame({
        "betweenness": betweenness,
        "eigenvector": eigenvector,
        "pagerank": pagerank,
        "in_degree": in_degree,
        "out_degree": out_degree,
    })

    return df.sort_values("eigenvector", ascending=False)  # Sort by importance

def weighted_graph_similarity_cosine(G1, G2, attr_name='weight'):
    """Computes cosine similarity between two weighted graphs."""
    all_edges = sorted(set(G1.edges()) | set(G2.edges()))
    
    vec1 = graph_edge_weight_vector(G1, edge_order=all_edges, weight_attr=attr_name)
    vec2 = graph_edge_weight_vector(G2, edge_order=all_edges, weight_attr=attr_name)
    return cosine_similarity([vec1], [vec2])[0][0]

def get_laplacian_spectrum(graph, k=None, normed=False):
    """Returns sorted eigenvalues of the Laplacian matrix."""
    L = nx.normalized_laplacian_matrix(graph).todense() if normed else nx.laplacian_matrix(graph).todense()
    eigenvalues = eigvalsh(L)  # Efficient for symmetric matrices
    eigenvalues = np.sort(eigenvalues)
    if k:  # Use only the first k eigenvalues
        eigenvalues = eigenvalues[:k]
    return eigenvalues

def spectral_similarity(G1, G2, k=None, normed=False, method='cosine'):
    """Computes spectral similarity between two graphs."""
    ev1 = get_laplacian_spectrum(G1, k=k, normed=normed)
    ev2 = get_laplacian_spectrum(G2, k=k, normed=normed)

    # Zero-pad the shorter vector to make them the same length
    max_len = max(len(ev1), len(ev2))
    ev1 = np.pad(ev1, (0, max_len - len(ev1)))
    ev2 = np.pad(ev2, (0, max_len - len(ev2)))

    if method == 'cosine':
        return cosine_similarity([ev1], [ev2])[0][0]
    elif method == 'euclidean':
        return -np.linalg.norm(ev1 - ev2)  # Lower is more similar; use negative for "higher = better"
    elif method == 'l1':
        return -np.sum(np.abs(ev1 - ev2))
    else:
        raise ValueError("Unsupported method: choose 'cosine', 'euclidean', or 'l1'")

def graph_edge_weight_vector(graph, edge_order=None, default=0.0, weight_attr='weight'):
    """Returns a vector of edge weights in a consistent order."""
    if edge_order is None:
        edge_order = sorted(graph.edges())

    vector = []
    for u, v in edge_order:
        if graph.has_edge(u, v):
            w = graph[u][v].get(weight_attr, default)
        else:
            w = default
        vector.append(w)
    return np.array(vector)

def pearson_graph_similarity(G1, G2, weight_attr='weight'):
    """Computes Pearson correlation between the edge weight vectors of two graphs."""
    # Union of all edges from both graphs
    all_edges = sorted(set(G1.edges()) | set(G2.edges()))
    
    vec1 = graph_edge_weight_vector(G1, edge_order=all_edges, weight_attr=weight_attr)
    vec2 = graph_edge_weight_vector(G2, edge_order=all_edges, weight_attr=weight_attr)

    # Handle edge case: if all values are constant, correlation is undefined
    if np.std(vec1) == 0 or np.std(vec2) == 0:
        return 0.0

    r, p = pearsonr(vec1, vec2)
    return r, p 
def print_edges(G):
    edges = sorted(G.edges(data=True), key=lambda x: -abs(x[2]['weight']))
    for src, tgt, data in edges[:20]:
        print(f"{src} → {tgt}, weight = {data['weight']:.4f}")
refine_graphs = True

activity_graph = pickle.load(open('graphs/just_weights_mean.pickle', 'rb'))
shapley_graph = pickle.load(open('graphs/shap_graph_fix.pickle', 'rb'))
dummy_graph = nx.DiGraph()
edges = sorted(activity_graph.edges(data=True), key=lambda x: -abs(x[2]['weight']))
for src, tgt, data in edges:
    dummy_graph.add_edge(src, tgt, weight=10)  
if refine_graphs:
    activity_graph, shapley_graph, dummy_graph = get_refined_graphs([activity_graph, shapley_graph, dummy_graph], add_start = False) 

print_edges(activity_graph)
print("---------------------------------")
print_edges(shapley_graph)
print(weighted_graph_similarity_cosine(activity_graph, shapley_graph))
print(weighted_graph_similarity_cosine(dummy_graph, activity_graph))
print(weighted_graph_similarity_cosine(dummy_graph, shapley_graph))

print(pearson_graph_similarity(activity_graph, shapley_graph))
print(pearson_graph_similarity(dummy_graph, activity_graph))
print(pearson_graph_similarity(dummy_graph, shapley_graph))

similarity = spectral_similarity(activity_graph, shapley_graph, method='cosine')
print("Spectral Similarity:", similarity)
similarity = spectral_similarity(dummy_graph, activity_graph, method='cosine')
print("Spectral Similarity:", similarity)
similarity = spectral_similarity(shapley_graph, dummy_graph, method='cosine')
print("Spectral Similarity:", similarity)
exit()
shap_values = np.load("shap_values.npy")

# Normalize shapley values to [0, 1] for each layer and channel separately
shap_min = shap_values.min(axis=1, keepdims=True)
shap_max = shap_values.max(axis=1, keepdims=True)
shap_norm = (shap_values - shap_min) / (shap_max - shap_min + 1e-8)

graph = pickle.load(open('graphs/sim_weight.pickle', 'rb'))
graph = get_refined_graphs([graph], add_start=True)[0]
fig = matplotlib.pyplot.figure()
ax = fig.add_subplot()

# Get positions for nodes, then scale them outward for more space
pos = nx.bfs_layout(graph, '0')

# Extract edge weights for width scaling
edges = graph.edges(data=True)
edge_weights = [abs(data['weight']) * 5 for _, _, data in edges]

# Create labels that only display the kernel number (after '_k')
def kernel_label(node):
    if '_k' in node:
        return node.split('_k')[-1]
    elif node == '0':
        return '0'
    else:
        return node
labels = {node: kernel_label(node) for node in graph.nodes()}

# Set a light node color for better label readability
node_color = '#f0f0f0'  # very light gray

# Compute node sizes proportional to the largest of the two radii (classification/grasp)
min_radius = 0.025 
max_radius = 0.07 * 1.5

def norm(val):
    return min_radius + (max_radius - min_radius) * val

node_sizes = {}
for node in graph.nodes():
    if node == '0':
        node_sizes[node] = min_radius
        continue
    if '_k' in node:
        layer_name, kernel_idx = node.rsplit('_k', 1)
        try:
            layer_idx = layers.index(layer_name)
            kernel_idx = int(kernel_idx)
            shap_class = shap_norm[layer_idx, kernel_idx, 0]
            shap_grasp = shap_norm[layer_idx, kernel_idx, 1]
        except Exception:
            shap_class = shap_grasp = 0.0
    else:
        shap_class = shap_grasp = 0.0
    r_class = norm(shap_class)
    r_grasp = norm(shap_grasp)
    node_sizes[node] = max(r_class, r_grasp)

# Convert node_sizes (radius) to area for node_size parameter (matplotlib expects area)
node_size_list = []
for node in graph.nodes():
    radius = node_sizes[node]
    area = np.pi * (radius ** 2) * 10000  + 400# scale up for visibility 
    node_size_list.append(area)

# Draw edges first
nx.draw_networkx_edges(
    graph,
    pos=pos,
    ax=ax,
    width=edge_weights,
    node_size=node_size_list  # node_size now proportional to circle size
)

# Draw node double-circles for shapley scores
for node in graph.nodes():
    if node == '0':
        # Draw start node as a single circle
        circ = matplotlib.patches.Circle(pos[node], radius=min_radius, color=node_color, ec='black', zorder=2)
        ax.add_patch(circ)
        continue
    # Parse layer and kernel index
    if '_k' in node:
        layer_name, kernel_idx = node.rsplit('_k', 1)
        try:
            layer_idx = layers.index(layer_name)
            kernel_idx = int(kernel_idx)
            shap_class = shap_norm[layer_idx, kernel_idx, 0]
            shap_grasp = shap_norm[layer_idx, kernel_idx, 1]
        except Exception:
            shap_class = shap_grasp = 0.0
    else:
        shap_class = shap_grasp = 0.0

    r_class = norm(shap_class)
    r_grasp = norm(shap_grasp)

    # Draw outer circle for classification (blue), inner for grasp (orange)
    circ_class = matplotlib.patches.Circle(pos[node], radius=r_class, color='#a6cee3', ec='black', zorder=2 if r_class >= r_grasp else 3)
    circ_grasp = matplotlib.patches.Circle(pos[node], radius=r_grasp, color='#fdbf6f', ec='black', zorder=3 if r_class >= r_grasp else 2)
    ax.add_patch(circ_class)
    ax.add_patch(circ_grasp)

# Draw node labels
for node, (x, y) in pos.items():
    ax.text(x, y, labels[node], fontsize=8, ha='center', va='center', zorder=4)

# Add legend

legend_elements = [
    Patch(facecolor='#a6cee3', edgecolor='black', label='Blue: size ∝ class Shapley score'),
    Patch(facecolor='#fdbf6f', edgecolor='black', label='Orange: size ∝ grasp Shapley score'),
    Line2D([0], [0], color='black', lw=3, label='Edge thickness ∝ |activity × weight|'),
]
# Place legend outside the plot area (right side)
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.5, -0.05), fontsize=8, frameon=True)
# Add a title
ax.set_title("Visualizing Connections between top shapley score kernels", fontsize=12, pad=20)

ax.set_aspect('equal')
ax.axis('off')

# Save plot to file
matplotlib.use("Agg")
fig.savefig("graph.png")
# edges = sorted(G.edges(data=True), key=lambda x: -abs(x[2]['weight']))
# for src, tgt, data in edges[:20]:
#     print(f"{src} → {tgt}, weight = {data['weight']:.4f}")

# Calculate measures
# df_measures = calculate_graph_measures(G)

# Print top influential kernels
#print(df_measures.head(10))
