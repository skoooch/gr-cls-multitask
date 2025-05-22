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
def get_refined_graphs(graph_list, add_start=False, refinedness = 6):
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
            while len(kernels) < refinedness:
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
                        try:
                            refined_graph.add_edge(src, tgt, weight=graph_list[i][src][tgt]["weight"].cpu())
                        except:
                            refined_graph.add_edge(src, tgt, weight=graph_list[i][src][tgt]["weight"])
                        if layer_idx == 0 and add_start:
                            refined_graph.add_edge("image", src, weight=0.5)
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
        #and (not "features.0" in u)
        if graph.has_edge(u, v) and (not "rgb" in u) and (not "features.0" in u) :
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
    print(f"Correlating {len(vec1)} edge weights between the two graphs.")

    # Plot the correlation
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.scatter(vec1, vec2, alpha=0.6)
    plt.xlabel("Graph 1 edge weights")
    plt.ylabel("Graph 2 edge weights")
    plt.title("Edge Weight Correlation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("edge_weight_correlation.png")
    plt.close()
    r, p = pearsonr(vec1, vec2)
    return r, p 
def print_edges(G):
    edges = sorted(G.edges(data=True), key=lambda x: -abs(x[2]['weight']))
    for src, tgt, data in edges[:20]:
        print(f"{src} → {tgt}, weight = {data['weight']:.4f}")
        
def visualize_graph_discrete(threshold=0.15, shap_thresh=0.5):
    refinedness = 7
    shap_values = np.load("shap_values.npy")

    # Normalize shapley values to [0, 1] for each layer and channel separately
    shap_min = shap_values.min(axis=1, keepdims=True)
    shap_max = shap_values.max(axis=1, keepdims=True)
    shap_norm = (shap_values - shap_min) / (shap_max - shap_min + 1e-8)

    graph = pickle.load(open('graphs/sim_weight_activity.pickle', 'rb'))
    graph = get_refined_graphs([graph], add_start=True, refinedness=refinedness)[0]
    # --- Normalize edge weights between each pair of layers ---
    # Build mapping from node to layer
    node_to_layer = {}
    for node in graph.nodes():
        if node == 'image':
            node_to_layer[node] = 'start'
        elif '_k' in node:
            node_to_layer[node] = node.rsplit('_k', 1)[0]
        else:
            node_to_layer[node] = node

    # For each (layer_X, layer_Y) pair, collect edge weights and normalize
    for layer_idx in range(len(layers) - 1):
        layer_X = layers[layer_idx]
        layer_Y = layers[layer_idx + 1]
        # Collect all edges between these layers
        edge_list = []
        for u, v, data in graph.edges(data=True):
            if node_to_layer.get(u) == layer_X and node_to_layer.get(v) == layer_Y:
                edge_list.append((u, v, data))
        if not edge_list:
            continue
        weights = np.array([abs(data['weight']) for _, _, data in edge_list])
        w_min = weights.min()
        w_max = weights.max()
        # Avoid division by zero
        if w_max - w_min < 1e-8:
            norm_weights = np.ones_like(weights)
        else:
            norm_weights = (weights - w_min) / (w_max - w_min)
        # Assign normalized weights back
        for (u, v, data), nw in zip(edge_list, norm_weights):
            graph[u][v]['weight'] = nw

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot()

    pos = nx.bfs_layout(graph, 'image')

    # --- Node typing based on ranking ---
    node_types = {}
    top_k = 10  # number of top kernels per layer/channel

    # Build sets of top kernels for each layer and channel
    top_class = set()
    top_grasp = set()
    for layer_idx, layer in enumerate(layers):
        # Get indices of top_k for class and grasp
        class_ranks = np.argsort(-shap_values[layer_idx, :, 0])[:top_k]
        grasp_ranks = np.argsort(-shap_values[layer_idx, :, 1])[:top_k]
        for idx in class_ranks:
            top_class.add(f"{layer}_k{idx}")
        for idx in grasp_ranks:
            top_grasp.add(f"{layer}_k{idx}")

    for node in graph.nodes():
        if node == 'image':
            node_types[node] = 'start'
            continue
        if node in top_class and node in top_grasp:
            node_types[node] = 'no_pref'
        elif node in top_class:
            node_types[node] = 'class'
        elif node in top_grasp:
            node_types[node] = 'grasp'
        else:
            node_types[node] = 'no_pref'

    # --- Node colors ---
    node_color_map = {
        'class': '#4F8DF7',      # blue
        'grasp': '#F7A24F',      # orange
        'no_pref': '#B0B0B0',    # gray
        'start': '#f0f0f0'
    }
    node_colors = [node_color_map[node_types[n]] for n in graph.nodes()]

    # --- Draw nodes ---
    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        ax=ax,
        node_color=node_colors,
        node_size=600 *(3/refinedness),
        edgecolors='black'
    )

    # --- Node labels ---
    def kernel_label(node):
        if '_k' in node:
            return node.split('_k')[-1]
        elif node == 'image':
            return 'image'
        else:
            return node
    labels = {node: kernel_label(node) for node in graph.nodes()}
    for node, (x, y) in pos.items():
        ax.text(x, y, labels[node], fontsize=8, ha='center', va='center', zorder=4)

    # --- Edge coloring and filtering ---
    edge_colors = []
    edge_styles = []
    edges_to_draw = []
    for u, v, data in graph.edges(data=True):
        w = abs(data['weight'])
        if w < threshold:
            continue
        t1 = node_types.get(u, 'no_pref')
        t2 = node_types.get(v, 'no_pref')
        if t1 == t2 and t1 in ['class', 'grasp']:
            color = '#4F8DF7' if t1 == 'class' else '#F7A24F'
            style = 'solid'
        elif 'no_pref' in (t1, t2) or t1 == "image":
            color = '#B0B0B0'
            style = 'dotted'
        else:
            color = '#7D3C98'  # purple for opposite
            style = 'dashed'
        edge_colors.append(color)
        edge_styles.append(style)
        edges_to_draw.append((u, v, w))

    # Draw edges by style, all with the same thickness
    fixed_width = 1.5
    for style in set(edge_styles):
        idxs = [i for i, s in enumerate(edge_styles) if s == style]
        edge_list = [edges_to_draw[i][:2] for i in idxs]
        color = [edge_colors[i] for i in idxs]
        nx.draw_networkx_edges(
            graph,
            pos=pos,
            ax=ax,
            edgelist=edge_list,
            width=fixed_width,
            edge_color=color,
            style=style
        )

    # --- Legend ---
    legend_elements = [
        Patch(facecolor='#4F8DF7', edgecolor='black', label='Classification node'),
        Patch(facecolor='#F7A24F', edgecolor='black', label='Grasp node'),
        Patch(facecolor='#B0B0B0', edgecolor='black', label='No preference node'),
        Patch(facecolor='#f0f0f0', edgecolor='black', label='Start node'),
        Line2D([0], [0], color='#4F8DF7', lw=2, label='Class→Class edge', linestyle='solid'),
        Line2D([0], [0], color='#F7A24F', lw=2, label='Grasp→Grasp edge', linestyle='solid'),
        Line2D([0], [0], color='#7D3C98', lw=2, label='Class↔Grasp edge', linestyle='dashed'),
        Line2D([0], [0], color='#B0B0B0', lw=2, label='Edge to/from No preference', linestyle='dotted'),
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, frameon=True)
    ax.set_title("Node/Edge Types by Shapley Ranking", fontsize=12, pad=20)
    ax.set_aspect('equal')
    ax.axis('off')

    matplotlib.use("Agg")
    fig.tight_layout()
    fig.savefig("graph_discrete.png")
    
    
def visualize_graph():
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
    pos = nx.bfs_layout(graph, 'image')

    # Extract edge weights for width scaling
    edges = graph.edges(data=True)
    edge_weights = [abs(data['weight']) * 5 for _, _, data in edges]

    # Create labels that only display the kernel number (after '_k')
    def kernel_label(node):
        if '_k' in node:
            return node.split('_k')[-1]
        elif node == 'image':
            return 'image'
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
        if node == 'image':
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
        if node == 'image':
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
    
    
refine_graphs = True

activity_graph = pickle.load(open('graphs/sim_weight.pickle', 'rb'))
shapley_graph = pickle.load(open('graphs/shap_graph_fix.pickle', 'rb'))
dummy_graph = nx.DiGraph()
edges = sorted(activity_graph.edges(data=True), key=lambda x: -abs(x[2]['weight']))
for src, tgt, data in edges:
    dummy_graph.add_edge(src, tgt, weight=10)  
if refine_graphs:
    activity_graph, shapley_graph, dummy_graph = get_refined_graphs([activity_graph, shapley_graph, dummy_graph], add_start = False) 
visualize_graph_discrete()
exit()
print_edges(activity_graph)
print("---------------------------------")
print_edges(shapley_graph)
# print(weighted_graph_similarity_cosine(activity_graph, shapley_graph))
# print(weighted_graph_similarity_cosine(dummy_graph, activity_graph))
# print(weighted_graph_similarity_cosine(dummy_graph, shapley_graph))

print(pearson_graph_similarity(activity_graph, shapley_graph))
# print(pearson_graph_similarity(dummy_graph, activity_graph))
# print(pearson_graph_similarity(dummy_graph, shapley_graph))

# similarity = spectral_similarity(activity_graph, shapley_graph, method='cosine')
# print("Spectral Similarity:", similarity)
# similarity = spectral_similarity(dummy_graph, activity_graph, method='cosine')
# print("Spectral Similarity:", similarity)
# similarity = spectral_similarity(shapley_graph, dummy_graph, method='cosine')
# print("Spectral Similarity:", similarity)
exit()

# edges = sorted(G.edges(data=True), key=lambda x: -abs(x[2]['weight']))
# for src, tgt, data in edges[:20]:
#     print(f"{src} → {tgt}, weight = {data['weight']:.4f}")

# Calculate measures
# df_measures = calculate_graph_measures(G)

# Print top influential kernels
#print(df_measures.head(10))
