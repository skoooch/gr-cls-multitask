import networkx as nx
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

layers = ['rgb_features.0', 'features.0', 'features.4', 'features.7', 'features.10']

grasp_kernels = [[62, 17], [19, 2, 7], [42, 13, 22, 5, 50], [41, 22, 21, 13, 12], [1, 45, 42, 20]]
cls_kernels = [[20], [17], [40, 47, 32, 28], [58, 59, 54, 45], [13, 10, 12, 6, 4]]

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

def normalize_edges_per_layer(graph):
    node_to_layer = {}
    for n in graph.nodes():
        if n == 'image':
            node_to_layer[n] = 'start'
        elif '_k' in n:
            node_to_layer[n] = n.rsplit('_k', 1)[0]
        else:
            node_to_layer[n] = n
    
    for layer_idx in range(len(layers) - 1):
        layer_x = layers[layer_idx]
        layer_y = layers[layer_idx + 1]
        edge_list = [(u, v, d) for u, v, d in graph.edges(data=True) 
                    if node_to_layer.get(u) == layer_x and node_to_layer.get(v) == layer_y]
        if not edge_list:
            continue
        
        weights = np.array([abs(d['weight']) for _, _, d in edge_list])
        w_min, w_max = weights.min(), weights.max()
        norm_weights = np.ones_like(weights) if w_max - w_min < 1e-8 else (weights - w_min) / (w_max - w_min)
        
        for (u, v, d), nw in zip(edge_list, norm_weights):
            graph[u][v]['weight'] = nw

def visualize_graph_discrete(refinedness=64, weight_threshold=0.0):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial'] + plt.rcParams['font.sans-serif']
    
    shap_values = np.load('shap_arrays/shap_values.npy')
    graph = pickle.load(open('graphs/just_weights.pickle', 'rb'))
    normalize_edges_per_layer(graph)
    graph = get_refined_graphs([graph], refinedness=refinedness, add_start=False)[0]
    normalize_edges_per_layer(graph)
    
    node_types = {}
    top_class, top_grasp, no_pref = set(), set(), set()
    
    for layer_idx, layer in enumerate(layers):
        for idx in cls_kernels[layer_idx]:
            top_class.add(f'{layer}_k{idx}')
        for idx in grasp_kernels[layer_idx]:
            top_grasp.add(f'{layer}_k{idx}')
    
    for node in graph.nodes():
        if node == 'image':
            node_types[node] = 'start'
        elif node in top_class and node in top_grasp:
            node_types[node] = 'no_pref'
        elif node in top_class:
            node_types[node] = 'class'
        elif node in top_grasp:
            node_types[node] = 'grasp'
        else:
            node_types[node] = 'no_pref'
    
    node_to_layer = {}
    for n in graph.nodes():
        if n == 'image':
            node_to_layer[n] = 'start'
        elif '_k' in n:
            node_to_layer[n] = n.rsplit('_k', 1)[0]
        else:
            node_to_layer[n] = n
    
    layer_positions, pos = {}, {}
    x_spacing, y_spacing = 256.0, 32.0
    min_y = float('inf')
    
    for layer_idx, layer in enumerate(layers):
        nodes = [n for n in graph.nodes() if n != 'image' and node_to_layer.get(n) == layer]
        
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
    
    fig, ax = plt.subplots(figsize=(64, 24))
    node_color_map = {'grasp': "#0391CE", 'class': '#F24236', 'no_pref': '#404040'}
    edge_color_map = {'grasp_grasp': '#0391CE', 'class_class': '#F24236', 'class_grasp': '#8E44AD', 'no_pref': '#404040'}
    
    for u, v, d in graph.edges(data=True):
        if u == 'image' or v == 'image':
            continue
        
        w = abs(d['weight'])
        if w < weight_threshold:
            continue
        
        t1, t2 = node_types.get(u, 'no_pref'), node_types.get(v, 'no_pref')
        if t1 == 'class' and t2 == 'class':
            color = edge_color_map['class_class']
        elif t1 == 'grasp' and t2 == 'grasp':
            color = edge_color_map['grasp_grasp']
        elif {'class', 'grasp'} <= {t1, t2}:
            color = edge_color_map['class_grasp']
        else:
            color = edge_color_map['no_pref']
        
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                color=color, alpha=w * 0.3, linewidth=(0.5 + 3*w), zorder=1)
    
    node_size = 100
    for ntype, color in node_color_map.items():
        nodes = [n for n in graph.nodes() if n != 'image' and node_types.get(n) == ntype]
        if nodes:
            xs, ys = zip(*[pos[n] for n in nodes])
            ax.scatter(xs, ys, c=color, s=node_size, edgecolors='black', 
                      linewidths=1, zorder=2, alpha=1.0)
    
    label_y = min_y - 30.0
    for layer_idx, layer in enumerate(layers):
        x = layer_idx * x_spacing
        ax.text(x, label_y, f'Layer {layer_idx + 1}', fontsize=16, ha='center', 
               va='center', fontweight='bold', zorder=3)
    
    # Add legend
    legend_elements = [
        Patch(facecolor=node_color_map['grasp'], edgecolor='black', label='Grasp kernel'),
        Patch(facecolor=node_color_map['no_pref'], edgecolor='black', label='No preference kernel'),
        Patch(facecolor=node_color_map['class'], edgecolor='black', label='Classification kernel'),
        Line2D([0], [0], color=edge_color_map['class_class'], linewidth=2, label='Class ↔ Class connection'),
        Line2D([0], [0], color=edge_color_map['grasp_grasp'], linewidth=2, label='Grasp ↔ Grasp connection'),
        Line2D([0], [0], color=edge_color_map['class_grasp'], linewidth=2, label='Class ↔ Grasp connection'),
        Line2D([0], [0], color=edge_color_map['no_pref'], linewidth=2, label='No preference connection')
    ]
    
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
             fontsize=20, frameon=False, fancybox=False, shadow=False)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('graph_discrete_integrated_all.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved visualization: graph_discrete_integrated_all.png')
    
    print(f"Verification - Nodes per layer:")
    for layer_idx, layer in enumerate(layers):
        nodes = [n for n in graph.nodes() if n != 'image' and node_to_layer.get(n) == layer]
        print(f"  Layer {layer_idx + 1} ({layer}): {len(nodes)} nodes")

if __name__ == '__main__':
    visualize_graph_discrete()
