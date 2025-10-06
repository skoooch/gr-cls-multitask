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
from datetime import datetime

METHOD_DESCRIPTIONS = {
    'topk_each': "Per layer pick top-k kernels by SHAP for each task; overlaps (in both top-k) become no_pref. Simple rank-based selectivity.",
    'z_ratio':   "Requires high z-score for one task, low for the other, plus dominance ratio threshold; combines magnitude and relative emphasis.",
    'effect':    "Uses normalized difference (effect size style); assigns if |(c-g)/( |c|+|g| )| exceeds threshold; otherwise no_pref.",
    'mixture':   "Fits 3-component Gaussian Mixture to (class−grasp) differences; extreme components => specialized, middle => no_pref."
}
import matplotlib.pyplot as plt
def compute_per_transition_chi_p(spec_counts, other_counts):
    """
    For each transition i build 2x2 table:
        [[spec_i, other_i],
         [spec_rest, other_rest]]
    Returns (chi_list, p_list)
    """
    spec_counts = np.array(spec_counts, dtype=float)
    other_counts = np.array(other_counts, dtype=float)
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
        chi2, p, dof, exp = stats.chi2_contingency(table, correction=False)
        chi_list.append(chi2)
        p_list.append(p)
    return chi_list, p_list

def plot_transition_chi(all_results,
                        save_path='chi_per_transition_significance.png',
                        title='Per-transition Chi-square',
                        show=False):
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
    transitions = all_results[0]['transitions'][1:]
    x = np.arange(len(transitions))  # 0..3
    plt.figure(figsize=(7,4))
    for res in all_results:
        chi_vals, p_vals = compute_per_transition_chi_p(res['specialized'], res['other'])
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
    plt.xticks(x, transitions, rotation=0)
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


import scipy.stats as stats
def build_node_types_advanced(shap_values,
                              method='z_ratio',
                              z_thresh=0.5,
                              cross_max=0.,
                              ratio_high=0.6,
                              ratio_low=0.4,
                              effect_thresh=0.15,
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
    for l, layer in enumerate(layers):
        # Determine actual channel count if variable; else assume K_max
        k_count = shap_values[l,:,0].shape[0]
        vals_class = shap_values[l,:k_count,0]
        vals_grasp = shap_values[l,:k_count,1]

        if method == 'topk_each':
            top_c = set(np.argsort(-vals_class)[:top_k])
            top_g = set(np.argsort(-vals_grasp)[:top_k])

        if method == 'mixture':
            from sklearn.mixture import GaussianMixture
            diff = vals_class - vals_grasp
            diff = diff.reshape(-1,1)
            gmm = GaussianMixture(n_components=3, random_state=0)
            gmm.fit(diff)
            # Order components by mean
            means = gmm.means_.flatten()
            order = np.argsort(means)
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
                if (zc >= z_thresh) and (zg <= cross_max) and (rc >= ratio_high):
                    node_types[name] = 'class'
                elif (zg >= z_thresh) and (zc <= cross_max) and (rc <= ratio_low):
                    node_types[name] = 'grasp'
                else:
                    node_types[name] = 'no_pref'

            elif method == 'effect':
                c = vals_class[k]; g = vals_grasp[k]
                d = (c - g) / (abs(c) + abs(g) + 1e-8)
                if d >= effect_thresh:
                    node_types[name] = 'class'
                elif d <= -effect_thresh:
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
                    if weight: spec += data['weight']
                    else: spec += 1 
                else:
                    if weight: other += data['weight']
                    else: other += 1 
        transitions.append(f"{Ls}->{Lt}")
        spec_counts.append(spec)
        other_counts.append(other)
    return transitions, spec_counts, other_counts
# ...existing code...

from datetime import datetime

METHOD_DESCRIPTIONS = {
    'topk_each': "Per layer pick top-k kernels by SHAP for each task; overlaps (in both top-k) become no_pref. Simple rank-based selectivity.",
    'z_ratio':   "Requires high z-score for one task, low for the other, plus dominance ratio threshold; combines magnitude and relative emphasis.",
    'effect':    "Uses normalized difference (effect size style); assigns if |(c-g)/( |c|+|g| )| exceeds threshold; otherwise no_pref.",
    'mixture':   "Fits 3-component Gaussian Mixture to (class−grasp) differences; extreme components => specialized, middle => no_pref."
}

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
        transitions, spec_counts, other_counts = count_specialized_edges_per_transition(graph_for_counts, node_types, weight=weight)

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
                f.write(f"Chi-square={res['chi2']:.3f}, dof={res['dof']}, p={res['p']:.4g}\n")
                logi = res['logistic']
                if logi['ok']:
                    f.write(f"Logistic trend: slope={logi['slope']:.4f}, p={logi['p']:.4g}\n")
                else:
                    f.write(f"Logistic trend: unavailable ({logi['error']})\n")
                f.write("Transition\tSpec\tOther\tProp_spec\tExp_spec\tExp_other\n")
                for i,tr in enumerate(res['transitions']):
                    exp_spec, exp_other = res['expected'][i]
                    f.write(f"{tr}\t{res['specialized'][i]}\t{res['other'][i]}\t"
                            f"{res['proportions'][i]:.3f}\t{exp_spec:.2f}\t{exp_other:.2f}\n")
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
results = run_chi_square_specialization(
    graph_path='graphs/just_weights3.pickle',
    refinedness=32,
    node_type_method='topk_each',
    out_path='count_chi_square_specialization_results.txt',
    counts_on='base',
    weight=weight
)
plot_transition_chi(results, save_path=f'{'weight' if weight else 'count'}_chi_per_transition_significance.png', title=f"{'Weighted' if weight else 'Unweighted'}Per-transition Chi-sqaure")
# ...existing code...