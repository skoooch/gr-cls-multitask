import networkx as nx
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from matplotlib import font_manager
from check_nr_agg import get_connections_first_average, get_model
import scipy.stats as stats

font_manager._load_fontmanager(try_read_cache=False)
font_path = 'ARIAL.TTF'  # Replace with the actual path
font_entry = font_manager.FontEntry(fname=font_path, name='MyCustomFontName')
font_manager.fontManager.ttflist.insert(0, font_entry) # Add to the beginning of the list
plt.rcParams['font.family'] = ['MyCustomFontName'] # Set as default
LAYERS = ['rgb_features.0', 'features.0', 'features.4', 'features.7', 'features.10']

def get_refined_graphs(graph_list, add_start=False, refinedness = 6, half_layer_2=False):
    shap_indices = np.load("shap_arrays/sort_shap_indices.npy")
    refined_graph_list = []
    for graph in graph_list:
        refined_graph_list.append(nx.DiGraph())
    for layer_idx in range(len(LAYERS) - 1):
        name_X = LAYERS[layer_idx]
        name_Y = LAYERS[layer_idx + 1]
        kernels_X = set()
        kernels_Y = set()
        for j, kernels in enumerate([kernels_X, kernels_Y]):
            i_c = 0
            i_g = 0
            grasp_flag = False
            while len(kernels) < (refinedness / (int(((layer_idx + j) == 1) and half_layer_2) + 1)):
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

def get_refined_graphs_dif(graph_list, add_start=False, refinedness = 6):
    shap_values = np.load("shap_arrays/shap_values.npy")
    shap_min = shap_values.min(axis=1, keepdims=True)
    shap_max = shap_values.max(axis=1, keepdims=True)
    shap_norm = (shap_values - shap_min) / (shap_max - shap_min + 1e-8)
    diffs = np.abs(shap_norm[:, :, 0] - shap_norm[:, :, 1])
    shap_inds = np.argsort(diffs[:, :], axis=1)[-refinedness:]
    refined_graph_list = []
    for graph in graph_list:
        refined_graph_list.append(nx.DiGraph())
    for layer_idx in range(len(LAYERS) - 1):
        name_X = LAYERS[layer_idx]
        name_Y = LAYERS[layer_idx + 1]
        kernels_X = set()
        kernels_Y = set()
        for j, kernels in enumerate([kernels_X, kernels_Y]):
            grasp_flag = False
            for i in range(refinedness):
                kernels.add(shap_inds[layer_idx + j, i])
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

def normalize_edges_per_layer(graph):
    node_to_layer = {}
    for n in graph.nodes():
        if n == 'image':
            node_to_layer[n] = 'start'
        elif '_k' in n:
            node_to_layer[n] = n.rsplit('_k', 1)[0]
        else:
            node_to_layer[n] = n
    
    for layer_idx in range(len(LAYERS) - 1):
        layer_x = LAYERS[layer_idx]
        layer_y = LAYERS[layer_idx + 1]
        edge_list = [(u, v, d) for u, v, d in graph.edges(data=True) 
                    if node_to_layer.get(u) == layer_x and node_to_layer.get(v) == layer_y]
        if not edge_list:
            continue
        
        weights = np.array([abs(d['weight']) for _, _, d in edge_list])
        w_min, w_max = weights.min(), weights.max()
        norm_weights = np.ones_like(weights) if w_max - w_min < 1e-8 else (weights - w_min) / (w_max - w_min)
        
        for (u, v, d), nw in zip(edge_list, norm_weights):
            graph[u][v]['weight'] = nw
            
def visualize_graph_discrete(refinedness=10, weight_threshold=0.0):
    # Set font to arial for all text elements
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial'] + plt.rcParams['font.sans-serif']
    
    shap_values = np.load('shap_arrays/shap_values.npy')
    graph = pickle.load(open('graphs/just_weights.pickle', 'rb'))
    normalize_edges_per_layer(graph)
    graph = get_refined_graphs([graph], refinedness=refinedness, add_start=False)[0]
    normalize_edges_per_layer(graph)
    
    # Determine node types
    node_types = {}

    # Min-max norm per layer for ratio computations
    mm = (shap_values - shap_values.min(axis=1, keepdims=True)) / \
         (shap_values.max(axis=1, keepdims=True) - shap_values.min(axis=1, keepdims=True) + 1e-8)
    effect_thresh = 0.3
    ratio_high = 0.5
    ratio_low = 0.3
    for node in graph.nodes():
        if node == 'image':
            node_types[node] = 'start'
            continue
        layer_name = node.split('_k')[0]
        l = int(LAYERS.index(layer_name))
        k = int(node.rsplit('_k', 1)[1])
        rc = mm[l,k,0] / (mm[l,k,0] + mm[l,k,1] + 1e-8)
        c = shap_values[l, k, 0]; g = shap_values[l, k, 1]
        d = (c - g) / (abs(c) + abs(g) + 1e-8)
        if d >= effect_thresh and (rc >= ratio_high):
            node_types[node] = 'class'
        elif d <= -effect_thresh and (rc <= ratio_low):
            node_types[node] = 'grasp'
        else:
            node_types[node] = 'no_pref'
    
    # Map nodes to layers
    node_to_layer = {}
    for n in graph.nodes():
        if n == 'image':
            node_to_layer[n] = 'start'
        elif '_k' in n:
            node_to_layer[n] = n.rsplit('_k', 1)[0]
        else:
            node_to_layer[n] = n
    
    # Position nodes
    layer_positions, pos = {}, {}
    x_spacing, y_spacing = 2.6, 0.8
    min_y = float('inf')
    
    for layer_idx, layer in enumerate(LAYERS):
        nodes = [n for n in graph.nodes() if n != 'image' and node_to_layer.get(n) == layer]
        
        # Verify we have exactly refinedness nodes per layer
        if len(nodes) != refinedness:
            print(f"Warning: Layer {layer} has {len(nodes)} nodes instead of {refinedness}")
        
        nodes.sort(key=lambda n: ({'grasp': 2, 'no_pref': 1, 'class': 0}[node_types.get(n, 'no_pref')], 
                                  int(n.split('_k')[-1])))
        layer_positions[layer] = nodes
        x = layer_idx * x_spacing
        
        if nodes:
            y_start = -(len(nodes) - 1) * y_spacing / 2
            for i, n in enumerate(nodes):
                y_pos = y_start + i * y_spacing
                pos[n] = (x, y_pos)
                min_y = min(min_y, y_pos)
    layer1_nodes = layer_positions[LAYERS[0]]
    n1 = len(layer1_nodes)
    image_y = 0 if n1 == 0 else (-(n1 - 1) * y_spacing / 2 + (n1 - 1) * y_spacing / 2)
    pos['image'] = (-x_spacing, image_y)
    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 12))
    # node_color_map = {'grasp': "#0391CE", 'class': '#F24236', 'no_pref': '#404040'}
    # edge_color_map = {'grasp_grasp': '#0391CE', 'class_class': '#F24236', 'class_grasp': '#8E44AD', 'no_pref': '#404040'}
    node_color_map = {'grasp': "blue", 'class': 'red', 'no_pref': 'gray'}
    edge_color_map = {'grasp_grasp': 'blue', 'class_class': 'red', 'class_grasp': 'purple', 'no_pref': 'gray'}
    # Draw edges from "Image" to Layer 1 nodes as solid gray
    weights = get_connections_first_average(get_model())
    imp_weights = {}
    for n in layer1_nodes:
        imp_weights[int(n.split('_k')[-1])] = weights[int(n.split('_k')[-1])]
    # Normalize imp_weights to [0, 1]
    imp_vals = np.array(list(imp_weights.values()))
    min_w, max_w = imp_vals.min(), imp_vals.max()
    for k in imp_weights:
        if max_w - min_w < 1e-8:
            imp_weights[k] = 1.0
        else:
            imp_weights[k] = (imp_weights[k] - min_w) / (max_w - min_w)
    for n in layer1_nodes:
        w = imp_weights[int(n.split('_k')[-1])]
        ax.plot([pos['image'][0], pos[n][0]], [pos['image'][1], pos[n][1]],
                color='#888888', alpha=1.0, linewidth=(1 + 4*(w)), zorder=1)
    # Draw edges
    for u, v, d in graph.edges(data=True):
        if u == 'image' or v == 'image':
            continue
        
        w = abs(d['weight'])
        if w < weight_threshold:
            continue
        
        t1, t2 = node_types.get(u, 'no_pref'), node_types.get(v, 'no_pref')
        zorder = 1
        if t1 == 'class' and t2 == 'class':
            color = edge_color_map['class_class']
        elif t1 == 'grasp' and t2 == 'grasp':
            color = edge_color_map['grasp_grasp']
        elif {'class', 'grasp'} <= {t1, t2}:
            zorder = 0.5
            color = edge_color_map['class_grasp']
        else:
            zorder = 0
            color = edge_color_map['no_pref']
        
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                color=color, alpha=w, linewidth=(1 + 4*w), zorder=zorder)
    
    # Draw nodes (including "Image")
    node_size = 1000
    for ntype, color in node_color_map.items():
        nodes = [n for n in graph.nodes() if n != 'image' and node_types.get(n) == ntype]
        if nodes:
            xs, ys = zip(*[pos[n] for n in nodes])
            ax.scatter(xs, ys, c=color, s=node_size, edgecolors='black',
                       linewidths=1, zorder=2, alpha=1.0)
    # Draw the "Image" node
    ax.scatter([pos['image'][0]], [pos['image'][1]], c='#888888', s=node_size,
               edgecolors='black', linewidths=1, zorder=2)
    ax.text(pos['image'][0], pos['image'][1], "Image", fontsize=8, ha='center', va='center',
            color='white', fontweight='bold', zorder=3)
    
    # Add node labels
    for n, (x, y) in pos.items():
        if n != 'image':
            label = n.split('_k')[-1] if '_k' in n else n
            ax.text(x, y, label, fontsize=16, ha='center', va='center', 
                   color='white', fontweight='bold', zorder=3)
    
    # Add layer labels
    label_y = min_y - 1.5
    for layer_idx, layer in enumerate(LAYERS):
        x = layer_idx * x_spacing
        ax.text(x, label_y, f'Layer {layer_idx + 1}', fontsize=20, ha='center', 
               va='center', fontweight='bold', zorder=3)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=node_color_map['grasp'], 
               markeredgecolor='black', markersize=22, label='Dorsal Feature Map'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=node_color_map['no_pref'], 
               markeredgecolor='black', markersize=22, label='Unspecific Feature Map'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=node_color_map['class'], 
               markeredgecolor='black', markersize=22, label='Ventral Feature Map'),
        Line2D([0], [0], color=edge_color_map['grasp_grasp'], linewidth=6, label='Dorsal ↔ Dorsal connection'),
        Line2D([0], [0], color=edge_color_map['class_class'], linewidth=6, label='Ventral ↔ Ventral connection'),
        Line2D([0], [0], color=edge_color_map['class_grasp'], linewidth=6, label='Cross connection'),
        Line2D([0], [0], color=edge_color_map['no_pref'], linewidth=6, label='Unspecific connection')
    ]
    
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
             fontsize=20, frameon=False, fancybox=False, shadow=False)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('vis/chi2/graph_discrete_integrated.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved visualization: vis/chi2/graph_discrete_integrated.png')
    
    # Print verification
    print(f"Verification - Nodes per layer:")
    for layer_idx, layer in enumerate(LAYERS):
        nodes = [n for n in graph.nodes() if n != 'image' and node_to_layer.get(n) == layer]
        print(f"  Layer {layer_idx + 1} ({layer}): {len(nodes)} nodes")


if __name__ == '__main__':
    visualize_graph_discrete()
