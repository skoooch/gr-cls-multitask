import math
from os import sep
import networkx as nx
import pickle

from sklearn import dummy
from utils.parameters import Params
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3  
from data_processing.data_loader_v2 import DataLoader  
from matplotlib.patches import Patch
from scipy.stats import linregress
from scipy.stats import pearsonr
from scipy.linalg import eigvalsh
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime

METHOD_DESCRIPTIONS = {
    'topk_each': "Per layer pick top-k kernels by SHAP for each task; overlaps (in both top-k) become no_pref. Simple rank-based selectivity.",
    'z_ratio':   "Requires high z-score for one task, low for the other, plus dominance ratio threshold; combines magnitude and relative emphasis.",
    'effect':    "Uses normalized difference (effect size style); assigns if |(c-g)/( |c|+|g| )| exceeds threshold; otherwise no_pref.",
    'mixture':   "Fits 3-component Gaussian Mixture to (class−grasp) differences; extreme components => specialized, middle => no_pref."
}
import matplotlib.pyplot as plt
from matplotlib import font_manager
import scipy.stats as stats

font_manager._load_fontmanager(try_read_cache=False)
font_path = 'ARIAL.TTF'  # Replace with the actual path
font_entry = font_manager.FontEntry(fname=font_path, name='MyCustomFontName')
font_manager.fontManager.ttflist.insert(0, font_entry) # Add to the beginning of the list
plt.rcParams['font.family'] = ['MyCustomFontName'] # Set as default
def compute_per_transition_chi_p(spec_counts, other_counts, cross_counts, cross=False, remove_cross=False):
    """
    For each transition i build 2x2 table:
        [[spec_i, other_i],
         [spec_rest, other_rest]]
    Returns (chi_list, p_list)
    """
    spec_counts = np.array(spec_counts, dtype=float)
    other_counts = np.array(other_counts, dtype=float)
    cross_counts = np.array(cross_counts, dtype = float)
    if remove_cross or cross:
        other_counts = other_counts - cross_counts
    if cross: spec_counts = cross_counts
    total_spec = spec_counts.sum()
    total_other = other_counts.sum()
    chi_list = []
    p_list = []
    for i in range(len(spec_counts) - 1):
        total_spec = spec_counts[i: i+1].sum()
        total_other = other_counts[i:i+1].sum()
        table = np.array([
            [spec_counts[i+1], other_counts[i+1]],
            [total_spec, total_other]
        ])
        directional = spec_counts[i+1] - (spec_counts[i+1] + other_counts[i+1]) * \
            (spec_counts[i+1] + total_spec) / (total_spec + total_other + spec_counts[i+1] + other_counts[i+1]) 
        direction = (int(directional) > 0) - (int(directional) < 0) 
        print(table)
        chi2, p, dof, exp = stats.chi2_contingency(table, correction=False)
        chi_list.append(chi2 * direction)
        p_list.append(p)
    return chi_list, p_list
def fix_labels(mylabels, tooclose=0.1, sepfactor=2):
    vecs = np.zeros((len(mylabels), len(mylabels), 2))
    dists = np.zeros((len(mylabels), len(mylabels)))
    for i in [1,3]:
        for j in [0]:
            a = np.array(mylabels[i].get_position())
            b = np.array(mylabels[j].get_position())
            dists[i,j] = np.linalg.norm(a-b)
            vecs[i,j,:] = a-b
            if dists[i,j] < tooclose:
                print(mylabels[i])
                if dists[i,j] < 0.02: temp_sepfactor = 5
                else: temp_sepfactor = sepfactor
                mylabels[i].set_x(a[0] + (1.4-dists[i,j]*9) * temp_sepfactor*vecs[i,j,0])
                mylabels[i].set_y(a[1] + (1.4-dists[i,j]*9) * temp_sepfactor*vecs[i,j,1])
                # mylabels[j].set_x(b[0] - (1.5-dists[i,j]*10) * temp_sepfactor*vecs[i,j,0])
                # mylabels[j].set_y(b[1] - (1.5-dists[i,j]*10) * temp_sepfactor*vecs[i,j,1])
def plot_transition_chi(all_results,
                        save_path='chi_per_transition_significance.png',
                        title='Per-transition Chi-square',
                        show=False, cross=False, remove_cross=False):
    """
    For each method plot 4 chi-square values (one per layer transition)
    with significance stars: * p<0.05, ** p<0.01, *** p<0.001
    """
    if not all_results:
        print("No results to plot.")
        return
    method_colors = {
        'topk_each': '#B5651D',
        'z_ratio':   '#800080',
        'effect':    '#006400',
        'mixture':   '#000000',
    }
    transition_names = ["Layer 2-3", "Layer 3-4", "Layer 4-5"]
    transitions = all_results[0]['transitions'][1:]
    x = np.arange(len(transitions))  # 0..3
    plt.figure(figsize=(7,4))
    plt.axhline(y=0, color='gray', linestyle='dashed')
    for res in all_results:
        chi_vals, p_vals = compute_per_transition_chi_p(res['specialized'], res['other'],res['cross'], cross, remove_cross=remove_cross)
        color = method_colors.get(res['method'], None)
        print(x)
        print(chi_vals)
        plt.plot(x, chi_vals, marker='o', linewidth=2, markersize=6, label=res['method'], color=color)
        # Add significance stars slightly above point
        for xi, (chi_v, p_v) in enumerate(zip(chi_vals, p_vals)):
            if p_v < 0.001:
                stars = '***'
            elif p_v < 0.01:
                stars = '**'
            elif p_v < 0.05:
                stars = '*'
            else:
                stars = ''
            if stars:
                plt.text(xi, chi_v + (0.03 * max(chi_vals + [1])), stars,
                         ha='center', va='bottom', fontsize=10, color=color)
    plt.xticks(x, transition_names, rotation=0)
    plt.xlabel("Layer transition")
    plt.ylabel("Chi-square (transition vs rest)")
    
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=200)
    
    if show:
        plt.show()
    plt.close()
    print(f"Saved plot with significance stars to {save_path}")
    #---------------------Plot pie charts-------------------------------------------
    import os
    transition_names = ["Layer 1-2", "Layer 2-3", "Layer 3-4", "Layer 4-5"]
    out_dir = os.path.dirname(save_path) or '.'
    os.makedirs(out_dir, exist_ok=True)
    for res in all_results:
        fig, sub_plots = plt.subplots(1, 4, figsize=(10, 3))
        counts_cls = np.array(res['cls'], dtype=float)
        counts_grasp = np.array(res['grasp'], dtype=float)
        counts_no_pref = np.array(res['other'], dtype=float) - np.array(res['cross'], dtype=float)
        counts_cross = np.array(res['cross'], dtype=float)
        for i, ax in enumerate(sub_plots):
            vals = [counts_cls[i],counts_grasp[i], counts_no_pref[i], counts_cross[i]]
            total = sum(vals)
            percents = np.array(vals)/total
            wedges, labels, autopct = ax.pie(vals,
                   colors=['red', 'blue', 'lightgray', 'purple'],autopct='%1.1f%%',
       pctdistance=1.25,textprops={'fontsize': 8})
            print(autopct)
            fix_labels(autopct, tooclose = 0.15, sepfactor=3.5)
            ax.set_title(transition_names[i])
        method_fname = f"{res['method']}_pie.png"
        save_p = os.path.join(out_dir, method_fname)
        fig.suptitle(res['method'])
        handles = [
            Patch(facecolor='lightgray', edgecolor='k', label='no preference'),
            Patch(facecolor='red', edgecolor='k', label='classification'),
            Patch(facecolor='blue', edgecolor='k', label='grasp'),
            Patch(facecolor='purple', edgecolor='k', label='cross'),
        ]
        fig.legend(handles=handles, loc='lower right', frameon=False, fontsize=10)
        fig.savefig(save_p, dpi=300)

        plt.close(fig)
        print(f"Saved pie chart to {save_p}")
    
    
layers = ['rgb_features.0', 'features.0','features.4','features.7','features.10']

grasp_kernels=[[62,17,],[19,2,7],[42,13,22,5,50],[41,22,21,13,12],[1,45,42,20]]
cls_kernels=[[20],[17],[40,47,32,28],[58,59,54,45,],[13,10,12,6,4]]

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

def build_node_types_advanced(shap_values,
                              method='z_ratio',
                              z_thresh=0.15,
                              cross_max=0.15,
                              ratio_high=0.3,
                              ratio_low=0.3,
                              effect_thresh=0.5,
                              top_k=20):
    """
    Advanced node typing.
    Methods:
      'topk_each'  : per-layer top_k per task (ties -> no_pref if in both)
      'z_ratio'    : z-score + ratio joint rule
      'effect'     : effect-size style normalized difference
      'mixture'    : 3-component GMM on differences (requires sklearn)
    """
    L, K_max, T = shap_values.shape
    assert T == 2, "Expect shape (..., 2) for (class, grasp)"

    # Per-layer z-score per task
    z = np.zeros_like(shap_values, dtype=float)
    for l in range(L):
        for t in range(T):
            v = shap_values[l,:,t]
            z[l,:,t] = (v - v.mean()) / (v.std() + 1e-8)

    # Min-max norm per layer for ratio computations
    mm = (shap_values - shap_values.min(axis=1, keepdims=True)) / \
         (shap_values.max(axis=1, keepdims=True) - shap_values.min(axis=1, keepdims=True) + 1e-8)

    node_types = {}
    layer_count = [64,32,64,64,64]
    for l, layer in enumerate(layers):
        # Determine actual channel count if variable; else assume K_max
        k_count = shap_values[l,:,0].shape[0]
        k_count = layer_count[l]
        vals_class = shap_values[l,:k_count,0]
        vals_grasp = shap_values[l,:k_count,1]

        if method == 'topk_each':
            top_c = set(np.argsort(-vals_class)[:top_k])
            top_g = set(np.argsort(-vals_grasp)[:top_k])

        if method == 'mixture':
            from sklearn.mixture import GaussianMixture
            diff = vals_class - vals_grasp
            diff = diff.reshape(-1,1)
            gmm = GaussianMixture(n_components=3, max_iter=50)
            gmm.fit(diff)
            
            # Order components by mean
            means = gmm.means_.flatten()
            
            order = np.argsort(means)
            if l == 4:
                print(means)
                exit()
            comp_to_label = {order[0]:'grasp', order[1]:'no_pref', order[2]:'class'}
            comps = gmm.predict(diff)

        for k in range(k_count):
            name = f"{layer}_k{k}"
            if method == 'topk_each':
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

            elif method == 'z_ratio':
                zc = z[l,k,0]; zg = z[l,k,1]
                rc = mm[l,k,0] / (mm[l,k,0] + mm[l,k,1] + 1e-8)
                # Dual criteria
                if (zc >= z_thresh) and (zg <= cross_max): #  
                    node_types[name] = 'class'
                elif (zg >= z_thresh) and (zc <= cross_max): #  and (rc <= ratio_low)
                    node_types[name] = 'grasp'
                else:
                    node_types[name] = 'no_pref'

            elif method == 'effect':
                rc = mm[l,k,0] / (mm[l,k,0] + mm[l,k,1] + 1e-8)
                c = vals_class[k]; g = vals_grasp[k]
                d = (c - g) / (abs(c) + abs(g) + 1e-8)
                if d >= effect_thresh and (rc >= ratio_high):
                    node_types[name] = 'class'
                elif d <= -effect_thresh and (rc <= ratio_low):
                    node_types[name] = 'grasp'
                else:
                    node_types[name] = 'no_pref'

            elif method == 'mixture':
                node_types[name] = comp_to_label[comps[k]]

            else:
                raise ValueError("Unknown method.")

    node_types['image'] = 'start'
    return node_types
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

def count_specialized_edges_per_transition(graph, node_types, weight=True):
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
    cross_counts = []
    grasp_counts = []
    cls_counts = []
    for i in range(len(layers)-1):
        Ls = layers[i]
        Lt = layers[i+1]
        spec = 0
        cls = 0
        grasp = 0
        other = 0
        cross = 0
        for u,v,data in graph.edges(data=True):
            lu = node_layer(u)
            lv = node_layer(v)
            if lu == Ls and lv == Lt:
                t_u = node_types.get(u,'no_pref')
                t_v = node_types.get(v,'no_pref')
                # specialized edge: both specialized and same type (class->class OR grasp->grasp)
                if t_u == t_v and t_u in ('class','grasp'):
                    if weight: spec += abs(data['weight'])
                    else: spec += 1 
                    if t_u == "class":
                        if weight: cls += abs(data['weight'])
                        else: cls += 1
                    else:
                        if weight: grasp += abs(data['weight'])
                        else: grasp += 1
                else:
                    if weight: 
                        if (t_u == 'class' and t_v =='grasp') or (t_v == 'class' and t_u =='grasp'):
                            cross += abs(data['weight'])
                        other += abs(data['weight'])
                    else: 
                        if (t_u == 'class' and t_v =='grasp') or (t_v == 'class' and t_u =='grasp'):
                            cross += 1 
                        other += 1 
        transitions.append(f"{Ls}->{Lt}")
        spec_counts.append(spec)
        other_counts.append(other)
        cross_counts.append(cross)
        cls_counts.append(cls)
        grasp_counts.append(grasp)
    return transitions, spec_counts, other_counts, cross_counts, cls_counts, grasp_counts
# ...existing code...

from datetime import datetime

METHOD_DESCRIPTIONS = {
    'topk_each': "Per layer pick top-k kernels by SHAP for each task; overlaps (in both top-k) become no_pref. Simple rank-based selectivity.",
    'z_ratio':   "Requires high z-score for one task, low for the other, plus dominance ratio threshold; combines magnitude and relative emphasis.",
    'effect':    "Uses normalized difference (effect size style); assigns if |(c-g)/( |c|+|g| )| exceeds threshold; otherwise no_pref.",
    'mixture':   "Fits 3-component Gaussian Mixture to (class−grasp) differences; extreme components => specialized, middle => no_pref."
}
SIZES = [128, 32,64,64,64]

def _pair_sizes(layer_i):
  # Determine (src_size, tgt_size) for the given layer index.
  # layer_i == -1 corresponds to the first split layer: RGB(3) x 128
  if layer_i == -1:
    return 3, SIZES[0]
  return SIZES[layer_i], SIZES[layer_i + 1]

def convert_value_to_index(src, tgt, layer_i):
  src_size, tgt_size = _pair_sizes(layer_i)
  if not (isinstance(src, (int, np.integer)) and isinstance(tgt, (int, np.integer))):
    raise TypeError("src and tgt must be integers")
  if not (0 <= src < src_size and 0 <= tgt < tgt_size):
    raise ValueError(f"(src, tgt)=({src}, {tgt}) out of range for layer {layer_i} "
                     f"(src: 0..{src_size-1}, tgt: 0..{tgt_size-1})")
  return int(src * tgt_size + tgt)

def compare_specialized_edge_to_shapley():
    shap_values = np.load("shap_arrays/shap_values_depth.npy")
    layer_sizes = [128,32,64,64,64]
    # Per-layer z-score per task
    z = np.zeros_like(shap_values, dtype=float)
    # FIX: remove trailing commas (they made tuples)
    z_thresh = 0.15
    cross_max = 0.15
    for l in range(5):
        for t in range(2):
            v = shap_values[l,:layer_sizes[l],t]
            z[l,:layer_sizes[l],t] = (v - v.mean()) / (v.std() + 1e-8)

    per_layer_estimated_connection_scores = []
    conn_shap_values = np.load("shap_arrays/shap_values_connections.npy")
    sums = np.abs(np.sum(conn_shap_values, axis=1, keepdims=True))  # shape: (num_layers, 1, 2)
    print(sums)
    # Normalize per-layer connection shap (optional scaling factor keeps magnitudes readable)
    conn_shap_values = (conn_shap_values / (sums + 1e-8)) * 1000

    # Collect per-type lists for each layer:
    # 0=class, 1=grasp, 2=cross, 3=no_pref; entries are tuples: (cls_val, grasp_val)
    shap_scores_per_type = []
    average_shap_scores_per_type = np.zeros((4,4,2))

    for layer_i in range(4):
        shap_scores_per_type.append([[],[],[],[]])
        estimated_connection_scores = np.zeros((layer_sizes[layer_i], layer_sizes[layer_i+1]))
        for src in range(layer_sizes[layer_i]):
            for tgt in range(layer_sizes[layer_i+1]):
                estimated_connection_scores[src, tgt] = abs((z[layer_i, src, 0] - z[layer_i, src, 1])  \
                    * (z[layer_i+1, tgt, 0] - z[layer_i + 1, tgt, 1]))

                zc_src = z[layer_i,src,0]; zg_src = z[layer_i,src,1]
                zc_tgt = z[layer_i + 1,tgt,0]; zg_tgt = z[layer_i + 1,tgt,1]

                # Node typing for src/tgt
                if (zc_src >= z_thresh) and (zg_src <= cross_max):
                    src_type = 'class'
                elif (zg_src >= z_thresh) and (zc_src <= cross_max):
                    src_type = 'grasp'
                else:
                    src_type = 'no_pref'
                if (zc_tgt >= z_thresh) and (zg_tgt <= cross_max):
                    tgt_type = 'class'
                elif (zg_tgt >= z_thresh) and (zc_tgt <= cross_max):
                    tgt_type = 'grasp'
                else:
                    tgt_type = 'no_pref'

                conn_shap_index = convert_value_to_index(src, tgt, layer_i)
                conn_tuple = tuple(conn_shap_values[layer_i + 1, conn_shap_index, :])  # (cls, grasp)

                if "no_pref" in [src_type, tgt_type]:
                    shap_scores_per_type[layer_i][3].append(conn_tuple)
                elif src_type == tgt_type:
                    if src_type == "class":
                        shap_scores_per_type[layer_i][0].append(conn_tuple)
                    else:
                        shap_scores_per_type[layer_i][1].append(conn_tuple)
                else:
                    shap_scores_per_type[layer_i][2].append(conn_tuple)

        per_layer_estimated_connection_scores.append(estimated_connection_scores)

        # Layer-wise averages (kept as before)
        for type_i in range(len(shap_scores_per_type[layer_i])):
            for grasp_flag in range(2):
                vals = [score[grasp_flag] for score in shap_scores_per_type[layer_i][type_i]]
                average_shap_scores_per_type[layer_i, type_i, grasp_flag] = np.sum(vals) / (len(vals) + 1e-8)

    # Print sanity
    for i in range(4):
        print(f"Layer {i}->{i+1} -----------------")
        print(average_shap_scores_per_type[i,:,0])
        print(average_shap_scores_per_type[i,:,1])

    # ---------- Single figure: 4 layers stacked vertically, two bars per layer (cls, grasp) ----------
    import os
    import matplotlib.pyplot as plt
    os.makedirs("vis", exist_ok=True)

    type_names = ["Classification", "Grasp", "Cross", "No Preference"]
    colors = ["#d62728", "#1f77b4", "#9467bd", "#7f7f7f"]

    def mean_ci(vals):
        vals = np.array(vals, dtype=float)
        if vals.size == 0:
            return 0.0, 0.0
        m = float(vals.mean())
        s = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        se = s / np.sqrt(max(1, vals.size))
        half = 1.96 * se
        return m, half

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 12), sharey=False)

    for layer_i in range(4):
        # Separate values per connection type for each task
        vals_cls = [[tup[0] for tup in shap_scores_per_type[layer_i][t_idx]] for t_idx in range(4)]
        vals_gr  = [[tup[1] for tup in shap_scores_per_type[layer_i][t_idx]] for t_idx in range(4)]

        means_cls, halfs_cls, ns_cls = [], [], []
        means_gr,  halfs_gr,  ns_gr  = [], [], []
        for t_idx in range(4):
            m, h = mean_ci(vals_cls[t_idx]); means_cls.append(m); halfs_cls.append(h); ns_cls.append(len(vals_cls[t_idx]))
            m, h = mean_ci(vals_gr[t_idx]);  means_gr.append(m);  halfs_gr.append(h);  ns_gr.append(len(vals_gr[t_idx]))

        x = np.arange(4)
        axL = axes[layer_i, 0]
        axR = axes[layer_i, 1]

        # Classification subplot (left)
        axL.bar(x, means_cls, yerr=halfs_cls, color=colors, capsize=4)
        for xi, (m, h, n) in enumerate(zip(means_cls, halfs_cls, ns_cls)):
            axL.text(xi, m + h + 0.02 * (max(abs(m)+h for m, h in zip(means_cls, halfs_cls)) + 1e-8),
                     f"n={n}", ha='center', va='bottom', fontsize=8)
        axL.set_xticks(x)
        axL.set_xticklabels(type_names, rotation=0)
        if layer_i == 0:
            axL.set_title("Classification")
        axL.set_ylabel(f"Layer {layer_i} -> {layer_i+1}\nConnection Shapley Score")
        # Center y=0 in the plot and increase ylim by 20%
        maxval = max([abs(m)+h for m, h in zip(means_cls, halfs_cls)] + [1e-8])
        axL.set_ylim(-1.2*maxval, 1.2*maxval)
        axL.axhline(0, color='gray', linewidth=1)

        # Grasp subplot (right)
        axR.bar(x, means_gr, yerr=halfs_gr, color=colors, capsize=4)
        for xi, (m, h, n) in enumerate(zip(means_gr, halfs_gr, ns_gr)):
            axR.text(xi, m + h + 0.02 * (max(abs(m)+h for m, h in zip(means_gr, halfs_gr)) + 1e-8),
                     f"n={n}", ha='center', va='bottom', fontsize=8)
        axR.set_xticks(x)
        axR.set_xticklabels(type_names, rotation=0)
        if layer_i == 0:
            axR.set_title("Grasp")
        # Center y=0 in the plot and increase ylim by 20%
        maxval_gr = max([abs(m)+h for m, h in zip(means_gr, halfs_gr)] + [1e-8])
        axR.set_ylim(-1.2*maxval_gr, 1.2*maxval_gr)
        axR.axhline(0, color='gray', linewidth=1)

    fig.suptitle("Shapley Score by Connection Type per Layer")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_path_all = "vis/conn_type_bars_all_layers.png"
    fig.savefig(out_path_all, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path_all}")

    # ---------- t-tests between all 4 types (across all layers and tasks) ----------
    from scipy import stats
    # Pool across all layers and both tasks
    pooled_all = {name: [] for name in type_names}
    for layer_i in range(4):
        for t_idx, name in enumerate(type_names):
            for tup in shap_scores_per_type[layer_i][t_idx]:
                # Add both task values as separate observations
                pooled_all[name].append(float(tup[0]))
                pooled_all[name].append(float(tup[1]))

    # Summary stats
    summary_lines = []
    summary_lines.append("Connection-type Shapley t-tests (pooled across layers and tasks)\n")
    for name in type_names:
        arr = np.array(pooled_all[name], dtype=float)
        summary_lines.append(f"{name:>8}: n={arr.size}, mean={arr.mean():.6f}, std={arr.std(ddof=1):.6f}")

    # Pairwise Welch t-tests
    pairs = []
    for i in range(4):
        for j in range(i+1, 4):
            a = np.array(pooled_all[type_names[i]], dtype=float)
            b = np.array(pooled_all[type_names[j]], dtype=float)
            if a.size > 1 and b.size > 1:
                tstat, pval = stats.ttest_ind(a, b, equal_var=False)
            else:
                tstat, pval = np.nan, np.nan
            pairs.append((type_names[i], type_names[j], tstat, pval))

    summary_lines.append("\nPairwise Welch t-tests:")
    for (a, b, t, p) in pairs:
        summary_lines.append(f"{a:>8} vs {b:<8}: t={t:.4f}, p={p:.4g}")

    # Optional: FDR correction across the 6 tests
    try:
        from statsmodels.stats.multitest import multipletests
        pvals = [p for (_, _, _, p) in pairs]
        rej, p_adj, _, _ = multipletests(pvals, method='fdr_bh')
        summary_lines.append("\nFDR (BH) adjusted p-values:")
        for k, (a, b, _, _) in enumerate(pairs):
            summary_lines.append(f"{a:>8} vs {b:<8}: p_adj={p_adj[k]:.4g}, reject={bool(rej[k])}")
    except Exception as e:
        summary_lines.append(f"\nFDR correction skipped: {e}")

    # Write report
    os.makedirs("vis", exist_ok=True)
    report_path = "vis/ttest_shap_connection_types.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Wrote t-test summary to {report_path}")

def run_chi_square_specialization(graph_path='graphs/just_weights.pickle',
                                  refinedness=10,
                                  node_type_method='hand_picked',
                                  print_details=True,
                                  out_path='weighted_chi_square_specialization_results.txt',
                                  counts_on='base',
                                  weight=True):
    """
    Loads graph, refines, classifies nodes with multiple methods, builds contingency tables,
    runs chi-square + logistic trend, and saves all results to a text file.

    counts_on: 'base' or 'refined' - which graph to use for edge counting.
    """
    import numpy as np

    shap_values = np.load("shap_arrays/shap_values.npy")
    shap_values[:,:,0] /= 65
    shap_values[:,:,1] /= 81.5
    base_graph = pickle.load(open(graph_path,'rb'))
    refined = get_refined_graphs([base_graph], add_start=False, refinedness=refinedness)[0]

    graph_for_counts = base_graph if counts_on == 'base' else refined

    methods_to_run = ['topk_each','z_ratio','effect','mixture']

    all_results = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for m in methods_to_run:
        if print_details:
            print(f"\n=== Node typing method = {m} ===")
        node_types = build_node_types_advanced(shap_values, method=m)
        transitions, spec_counts, other_counts, cross_counts, cls_counts, grasp_counts = count_specialized_edges_per_transition(graph_for_counts, node_types, weight=weight)

        contingency = np.array([spec_counts, other_counts]).T  # rows: transitions, cols: [spec, other]
        chi2, p, dof, expected = stats.chi2_contingency(contingency)

        # Logistic trend (edge-level)
        logistic = {'slope': None, 'p': None, 'ok': False, 'error': None}
        try:
            X_layer = []
            y_spec = []
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
                try:
                    import statsmodels.api as sm
                    X = sm.add_constant(np.array(X_layer))
                    model = sm.Logit(np.array(y_spec), X).fit(disp=False)
                    logistic['slope'] = float(model.params[1])
                    logistic['p'] = float(model.pvalues[1])
                    logistic['ok'] = True
                except Exception as e:
                    logistic['error'] = f"Statsmodels failure: {e}"
            else:
                logistic['error'] = "All edges same class (no variance)."
        except Exception as e:
            logistic['error'] = f"Trend wrapper error: {e}"

        proportions = [ (sc / (sc+oc) if (sc+oc)>0 else 0.0) for sc,oc in zip(spec_counts, other_counts) ]

        if print_details:
            print("Layer transitions:", transitions)
            print("Specialized weighted counts:", spec_counts)
            print("Other weighted counts:", other_counts)
            print("Proportions specialized:", [f"{p_i:.3f}" for p_i in proportions])
            print("Contingency (rows=transitions, cols=[specialized, other]):")
            print(contingency)
            print(f"Chi-square={chi2:.3f}, dof={dof}, p-value={p:.4g}")
            if logistic['ok']:
                print(f"Logistic slope={logistic['slope']:.3f}, p={logistic['p']:.4g}")
            else:
                print(f"Logistic trend unavailable: {logistic['error']}")

        all_results.append({
            'method': m,
            'description': METHOD_DESCRIPTIONS.get(m, ''),
            'transitions': transitions,
            'specialized': spec_counts,
            'other': other_counts,
            'cross': cross_counts,
            'grasp': grasp_counts,
            'cls': cls_counts,
            'proportions': proportions,
            'contingency': contingency.tolist(),
            'expected': expected.tolist(),
            'chi2': float(chi2),
            'p': float(p),
            'dof': int(dof),
            'logistic': logistic
        })

    # Write results file
    try:
        with open(out_path, 'w') as f:
            f.write(f"Chi-square Weighted specialization analysis\nTimestamp: {timestamp}\n")
            f.write(f"Graph path: {graph_path}\nRefinedness: {refinedness}\nCounts on: {counts_on}\n")
            f.write(f"Methods run: {', '.join(methods_to_run)}\n")
            f.write("-"*70 + "\n\n")
            for res in all_results:
                f.write(f"Method: {res['method']}\n")
                f.write(f"Description: {res['description']}\n")
                f.write(f"Total Chi-square={res['chi2']:.3f}, dof={res['dof']}, p={res['p']:.4g}\n")
                f.write("Transition\t\tSpec\tNo_pref\tCross\tChi2\tCross Chi2\n")
                trans_chis, trans_ps = compute_per_transition_chi_p(res['specialized'], res['other'], res['cross'], remove_cross=True)
                cross_trans_chis, cross_trans_ps = compute_per_transition_chi_p(res['specialized'], res['other'], res['cross'], cross=True, remove_cross=True)
                for arr in [trans_chis, trans_ps, cross_trans_chis, cross_trans_ps]: arr.insert(0, 'N/A')
                for i, tr in enumerate(res['transitions']):
                    cross_val = res['cross'][i]
                    trans_chi_val = trans_chis[i]
                    trans_p_val = trans_ps[i]
                    cross_trans_chi_val = cross_trans_chis[i]
                    cross_trans_p_val = cross_trans_ps[i]
                    # Safely format floats, otherwise print as string
                    cross_str = f"{cross_val:.3f}" if isinstance(cross_val, (float, int)) else str(cross_val)
                    trans_chi_str = f"{trans_chi_val:.3f}" if isinstance(trans_chi_val, (float, int)) else str(trans_chi_val)
                    trans_p_str = f"{trans_p_val:.4g}" if isinstance(trans_p_val, (float, int)) else str(trans_p_val)
                    cross_trans_chi_str = f"{cross_trans_chi_val:.3f}" if isinstance(cross_trans_chi_val, (float, int)) else str(cross_trans_chi_val)
                    cross_trans_p_str = f"{cross_trans_p_val:.4g}" if isinstance(cross_trans_p_val, (float, int)) else str(cross_trans_p_val)
                    f.write(f"{tr}\t{res['specialized'][i]:.3f}\t{(res['other'][i] - res['cross'][i]):.3f}\t{cross_str}\t{trans_chi_str}, (p={trans_p_str})\t{cross_trans_chi_str}, (p={cross_trans_p_str})\t\n")
                f.write("\n" + "-"*70 + "\n\n")
        if print_details:
            print(f"\nAll results written to {out_path}")
    except Exception as e:
        print(f"Failed to write results file: {e}")

    return all_results

# ---------------- Run the analysis ----------------
# Example call (adjust as needed):
# ---------------- Run the analysis + visualize ----------------
weight = True
cross = False
remove_cross = True
compare_specialized_edge_to_shapley()
exit()
results = run_chi_square_specialization(
    graph_path='graphs/just_weights3.pickle',
    refinedness=32,
    node_type_method='topk_each',
    out_path='count_chi_square_specialization_results.txt',
    counts_on='base',
    weight=weight
)
plot_transition_chi(results, 
                    save_path=f'vis/chi2/remove_cross_{'weight' if weight else 'count'}_{'cross_' if cross else ''}chi_per_transition_significance.png', 
                    title=f"{'Weighted' if weight else 'Unweighted'}Per-transition Chi-sqaure", 
                    cross=cross, remove_cross=remove_cross)
# ...existing code...
