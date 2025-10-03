import networkx as nx
import pickle

from sklearn import dummy
from utils.parameters import Params
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3  
from data_processing.data_loader_v2 import DataLoader  

from scipy.stats import linregress
from scipy.stats import pearsonr
from scipy.linalg import eigvalsh
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

layers = ['rgb_features.0', 'features.0','features.4','features.7','features.10']

grasp_kernels=[[62,17,],[19,2,7],[42,13,22,5,50],[41,22,21,13,12],[1,45,42,20]]
cls_kernels=[[20],[17],[40,47,32,28],[58,59,54,45,],[13,10,12,6,4]]

def detach_graph(graph):
    edges = sorted(graph.edges(data=True), key=lambda x: -abs(x[2]['weight']))
    for src, tgt, data in edges:
        graph[src][tgt]["weight"] = graph[src][tgt]["weight"].cpu()
def get_refined_graphs(graph_list, add_start=False, refinedness = 6):
    shap_indices = np.load("shap_arrays/sort_shap_indices.npy")
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
            while len(kernels) < (refinedness / (int((layer_idx + j) == 1) + 1)):
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
        #(not "rgb" in u) and
        if graph.has_edge(u, v) and "features.7" in v:
            w = graph[u][v].get(weight_attr, default)
            vector.append(w)
        else:
            continue
        
            
    return np.array(vector), edge_order

def pearson_graph_similarity(G1, G2, weight_attr='weight'):
    """Computes Pearson correlation between the edge weight vectors of two graphs."""
    # Union of all edges from both graphs
    all_edges = sorted(set(G1.edges()) & set(G2.edges()))

    vec1, edge_1 = graph_edge_weight_vector(G1, edge_order=all_edges, weight_attr=weight_attr)
    vec2, edge_2 = graph_edge_weight_vector(G2, edge_order=all_edges, weight_attr=weight_attr)
    assert(edge_1 == edge_2)
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
        
def normalize_edges_per_layer(graph):
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

# ...existing code...

import scipy.stats as stats

def build_node_types(shap_values, method='hand_picked', top_k=15):
    """
    Returns dict: node_name -> {'class','grasp','no_pref','start'}.
    """
    shap_min = shap_values.min(axis=1, keepdims=True)
    shap_max = shap_values.max(axis=1, keepdims=True)
    shap_norm = (shap_values - shap_min) / (shap_max - shap_min + 1e-8)

    node_types = {}
    if method == 'hand_picked':
        top_class = set()
        top_grasp = set()
        for layer_idx, layer in enumerate(layers):
            for idx in cls_kernels[layer_idx]:
                top_class.add(f"{layer}_k{idx}")
            for idx in grasp_kernels[layer_idx]:
                top_grasp.add(f"{layer}_k{idx}")
        for layer in layers:
            pass
        # Assign
        for layer_idx, layer in enumerate(layers):
            # Mark only those kernels we explicitly touched; others default no_pref
            max_k = [64,32,64,64,64][layer_idx] if layer_idx < len(layers) else 0
            for k in range(max_k):
                name = f"{layer}_k{k}"
                if name in top_class and name in top_grasp:
                    node_types[name] = 'no_pref'
                elif name in top_class:
                    node_types[name] = 'class'
                elif name in top_grasp:
                    node_types[name] = 'grasp'
                else:
                    node_types[name] = 'no_pref'
    elif method == 'topk_each':
        # Pick top_k per task per layer by raw shap score
        for layer_idx, layer in enumerate(layers):
            vals_class = shap_values[layer_idx, :, 0]
            vals_grasp = shap_values[layer_idx, :, 1]
            top_c = set(np.argsort(-vals_class)[:top_k])
            top_g = set(np.argsort(-vals_grasp)[:top_k])
            max_k = vals_class.shape[0]
            for k in range(max_k):
                name = f"{layer}_k{k}"
                c = k in top_c
                g = k in top_g
                if c and g:
                    node_types[name] = 'no_pref'
                elif c:
                    node_types[name] = 'class'
                elif g:
                    node_types[name] = 'grasp'
                else:
                    node_types[name] = 'no_pref'
    else:
        raise ValueError("Unsupported method for node typing.")

    node_types['image'] = 'start'
    return node_types

def count_specialized_edges_per_transition(graph, node_types):
    """
    Returns:
        transitions: list of "layerA->layerB"
        spec_counts: list counts of specialized (class-class + grasp-grasp) edges
        other_counts: list counts of all other edges between those layers
    """
    # Helper: map node -> layer name base
    def node_layer(n):
        if n == 'image':
            return None
        if '_k' in n:
            return n.rsplit('_k',1)[0]
        return n
    transitions = []
    spec_counts = []
    other_counts = []
    for i in range(len(layers)-1):
        Ls = layers[i]
        Lt = layers[i+1]
        spec = 0
        other = 0
        for u,v,data in graph.edges(data=True):
            lu = node_layer(u)
            lv = node_layer(v)
            if lu == Ls and lv == Lt:
                t_u = node_types.get(u,'no_pref')
                t_v = node_types.get(v,'no_pref')
                # specialized edge: both specialized and same type (class->class OR grasp->grasp)
                if t_u == t_v and t_u in ('class','grasp'):
                    spec += 1
                else:
                    other += 1
        transitions.append(f"{Ls}->{Lt}")
        spec_counts.append(spec)
        other_counts.append(other)
    return transitions, spec_counts, other_counts

def run_chi_square_specialization(graph_path='graphs/just_weights.pickle',
                                  refinedness=10,
                                  node_type_method='hand_picked',
                                  print_details=True):
    """
    Loads graph, refines, classifies nodes, builds contingency table and runs chi-square.
    """
    import numpy as np
    shap_values = np.load("shap_arrays/shap_values.npy")
    base_graph = pickle.load(open(graph_path,'rb'))
    # Use same refinement logic as visualization to keep kernel selection consistent
    refined = get_refined_graphs([base_graph], add_start=False, refinedness=refinedness)[0]
    node_types = build_node_types(shap_values, method=node_type_method)
    transitions, spec_counts, other_counts = count_specialized_edges_per_transition(base_graph, node_types)

    contingency = np.array([spec_counts, other_counts]).T  # rows: transitions, cols: [spec, other]
    chi2, p, dof, expected = stats.chi2_contingency(contingency)

    if print_details:
        print("Layer transitions:", transitions)
        print("Specialized counts:", spec_counts)
        print("Other counts:", other_counts)
        print("Contingency table (rows=transitions, cols=[specialized, other]):")
        print(contingency)
        print(f"Chi-square={chi2:.3f}, dof={dof}, p-value={p:.4g}")
        # Simple trend (logistic regression) to test increasing specialization
        try:
            import numpy as np
            from math import isfinite
            # Build per-edge dataset
            X_layer = []
            y_spec = []
            # Re-iterate edges
            def node_layer(n):
                if n == 'image': return None
                if '_k' in n: return n.rsplit('_k',1)[0]
                return n
            layer_to_index = {layer:i for i,layer in enumerate(layers)}
            for u,v,data in refined.edges(data=True):
                lu = node_layer(u); lv = node_layer(v)
                if lu in layer_to_index and lv in layer_to_index and layer_to_index[lv] == layer_to_index[lu]+1:
                    t_u = node_types.get(u,'no_pref')
                    t_v = node_types.get(v,'no_pref')
                    specialized = int(t_u == t_v and t_u in ('class','grasp'))
                    y_spec.append(specialized)
                    X_layer.append(layer_to_index[lu])
            if len(set(y_spec)) > 1:
                # Fit simple logistic via statsmodels if available
                try:
                    import statsmodels.api as sm
                    X = sm.add_constant(np.array(X_layer))
                    model = sm.Logit(np.array(y_spec), X).fit(disp=False)
                    slope_p = model.pvalues[1]
                    slope = model.params[1]
                    print(f"Logistic trend slope={slope:.3f}, p={slope_p:.4g} (positive & significant -> increasing specialization)")
                except Exception as e:
                    print("Trend logistic regression skipped:", e)
            else:
                print("All edges same class (cannot fit trend).")
        except Exception as e:
            print("Trend analysis skipped:", e)
    return {
        "transitions": transitions,
        "specialized": spec_counts,
        "other": other_counts,
        "chi2": chi2,
        "p": p,
        "expected": expected
    }

# ---------------- Run the analysis ----------------

# Comment out earlier exit() above before running this block.
results = run_chi_square_specialization(
    graph_path='graphs/just_weights.pickle',
    refinedness=32,
    node_type_method='topk_each'
)

# ...existing code...