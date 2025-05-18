import networkx as nx
import pickle
from utils.parameters import Params
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3  
from data_processing.data_loader_v2 import DataLoader  
import torch
import os
from scipy.linalg import eigvalsh
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def calculate_graph_measures(graph):
    """
    Given a directed, weighted graph of kernel connectivity, compute graph-theoretic measures.
    
    Returns:
        pandas.DataFrame with node-wise metrics.
    """
    # Ensure weights are used where appropriate
    inverted_graph = nx.DiGraph()
    edges = sorted(G.edges(data=True), key=lambda x: -abs(x[2]['weight']))
    for src, tgt, data in edges:
        inverted_graph.add_edge(src, tgt, weight=1-G[src][tgt]['weight'])
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

def graph_edge_vector(graph, all_edges, attr_name='weight'):
    """Returns a vector of edge weights for the graph aligned with all_edges."""
    return np.array([
        graph[u][v][attr_name] if graph.has_edge(u, v) and attr_name in graph[u][v] else 0.0
        for u, v in all_edges
    ])

def weighted_graph_similarity_cosine(G1, G2, attr_name='weight'):
    """Computes cosine similarity between two weighted graphs."""
    all_edges = set(G1.edges()) | set(G2.edges())
    v1 = graph_edge_vector(G1, all_edges, attr_name)
    v2 = graph_edge_vector(G2, all_edges, attr_name)
    return cosine_similarity([v1], [v2])[0][0]

def weighted_graph_similarity_cosine(G1, G2, attr_name='weight'):
    """Computes cosine similarity between two weighted graphs."""
    all_edges = set(G1.edges()) | set(G2.edges())
    v1 = graph_edge_vector(G1, all_edges, attr_name)
    v2 = graph_edge_vector(G2, all_edges, attr_name)
    return cosine_similarity([v1], [v2])[0][0]

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
    
    
activity_graph = pickle.load(open('sim_weight.pickle', 'rb'))

shapley_graph = pickle.load(open('shap_graph_unnormalized.pickle', 'rb'))

dummy_graph = nx.DiGraph()
edges = sorted(activity_graph.edges(data=True), key=lambda x: -abs(x[2]['weight']))
for src, tgt, data in edges:
    dummy_graph.add_edge(src, tgt, weight=10)   
print(weighted_graph_similarity_cosine(activity_graph, shapley_graph))
print(weighted_graph_similarity_cosine(dummy_graph, activity_graph))
print(weighted_graph_similarity_cosine(dummy_graph, shapley_graph))

similarity = spectral_similarity(activity_graph, shapley_graph, method='cosine')
print("Spectral Similarity:", similarity)
similarity = spectral_similarity(dummy_graph, shapley_graph, method='cosine')
print("Spectral Similarity:", similarity)
similarity = spectral_similarity(activity_graph, dummy_graph, method='cosine')
print("Spectral Similarity:", similarity)
# edges = sorted(G.edges(data=True), key=lambda x: -abs(x[2]['weight']))
# for src, tgt, data in edges[:20]:
#     print(f"{src} → {tgt}, weight = {data['weight']:.4f}")

# Calculate measures
# df_measures = calculate_graph_measures(G)

# Print top influential kernels
#print(df_measures.head(10))
