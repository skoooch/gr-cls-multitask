import numpy as np
import sys
import glob
from scipy.stats import pearsonr
import pandas as pd
from sympy import *
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from svarm_run1.true_corr_ci import corr_true_ci_from_mse_bounds
import os
import torch
from utils.parameters import Params
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3

LAYERS = ['rgb_features.0', 'features.0','features.4','features.7','features.10']

def get_model(device='cpu'):
    params = Params()
    model = Multi_AlexnetMap_v3()
    weights_dir = params.MODEL_PATH
    model_name = params.MODEL_NAME
    ckpt = os.path.join(weights_dir, model_name, model_name + '_final.pth')
    sd = torch.load(ckpt, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model.to(device)

def extract_conv_weight_mats(model):
    """
    Return dict: layer_name -> 2D np.array of shape [out_ch, in_ch]
    using mean absolute weights across spatial dims.
    """
    mats = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and hasattr(mod, 'weight') and mod.weight is not None:
            W = mod.weight.data.detach()  # [out, in, kH, kW]
            if W.ndim == 4:
                W2 = torch.mean(torch.abs(W), dim=(2, 3)).cpu().numpy()  # [out, in]
                mats[name] = W2
    return mats

def plot_all_layer_scatter(X_arrays, Y_arrays, r_values, confidence_intervals):
    '''
    plot correlation in neuron shapley values between tasks across layers
    plot shapley graphs for each layer under the correlation graph
    '''

    fig, ax = plt.subplots(figsize=(10, 9))
    x_values = np.arange(1, len(r_values) + 1)
    n = 64 # can change number neurons in each layer

    # confidence_intervals = [confidence_interval(r, n) for r in r_values]
    lower_bounds = [ci[0] for ci in confidence_intervals]
    upper_bounds = [ci[1] for ci in confidence_intervals]

    ax.errorbar(
        x_values, r_values, 
        yerr=[np.abs(np.array(r_values) - np.array(lower_bounds)), 
              np.abs(np.array(upper_bounds) - np.array(r_values))],
        fmt='o', capsize=5, linestyle='--', color="black"
    )

    ax.set_xticks(x_values)
    ax.xaxis.set_major_locator(plt.FixedLocator([1, 2, 3, 4, 5]))
    ax.axhline(0, color='gray', linewidth=0.7, linestyle='-')
    ax.set_ylabel('Correlation (r-value)', labelpad=2, fontsize=22)
    ax.set_xlabel('Feature extraction layer', labelpad=1.5, fontsize=22)
    #ax.set_title('Correlation in Connection Shapley Values Between Tasks Across Layers', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=18)

    # add inset axes below x-axis
    fig.subplots_adjust(bottom=0.38)
    for i in range(len(X_arrays)):
        x, y, r = X_arrays[i], Y_arrays[i], r_values[i]
        inset_width = 0.12
        inset_height = 0.167
        left = 0.13 + i * (0.83 / len(X_arrays))  # even horizontal spacing
        bottom = 0.08

        inset_ax = fig.add_axes([left, bottom, inset_width, inset_height])
        inset_ax.scatter(x, y, s=(rcParams['lines.markersize']**2)/4, alpha=0.7, color="black")
        inset_ax.set_title(f'r = {r:.2f}', fontsize=16)
    plt.savefig('vis/shap/connections/combined_below_graph_black.png', dpi=300)

def merge_welford(m1,n1,M2_1,m2,n2,M2_2):
    if n1 == 0: return m2, n2, M2_2
    if n2 == 0: return m1, n1, M2_1
    delta = m2 - m1
    n = n1 + n2
    m = m1 + delta * (n2 / n)
    M2 = M2_1 + M2_2 + delta*delta * (n1*n2)/n
    return m, n, M2

def get_connections(model: nn.Module, layer="features.0") -> nn.Module:
    """
    Return the mean weight magnitude of each kernel in layer.
    """
    with torch.no_grad():
        # Directly access the layer's weight
        layer_module = dict(model.named_modules())[layer]
        W = layer_module.weight
        mean_weights = []
        for src in range(W.data.shape[1]):
            for tgt in range(W.data.shape[0]):
                mean_weights.append(torch.mean(torch.abs(W.data[tgt, src, :, :])))
    return mean_weights

def get_connections_first(model: nn.Module) -> nn.Module:
    """
    Return the mean weight magnitude of each kernel in layer 1.
    """
    with torch.no_grad():
        # Directly access the layer's weight
        rgb_layer_module = dict(model.named_modules())['rgb_features.0']
        depth_layer_module = dict(model.named_modules())['d_features.0']
        W_rgb = rgb_layer_module.weight
        W_d = depth_layer_module.weight
        # Store original weights
        mean_weights = []
        for i in range(3):
            for j in range(128):
                src,tgt = i,j
                if tgt >= 64:
                    tgt -= 64
                    mean_weights.append(torch.mean(torch.abs(W_d[tgt, src, :, :])))
                else:
                    mean_weights.append(torch.mean(torch.abs(W_rgb[tgt, src, :,:])))       
    return mean_weights

def get_connections_first_average(model: nn.Module) -> nn.Module:
    with torch.no_grad():
        # Directly access the layer's weight
        rgb_layer_module = dict(model.named_modules())['rgb_features.0']
        depth_layer_module = dict(model.named_modules())['d_features.0']
        W_rgb = rgb_layer_module.weight
        W_d = depth_layer_module.weight
        # Store original weights
        mean_weights = []
        for j in range(128):
            tgt = j
            if tgt >= 64:
                tgt -= 64
                mean_weights.append(torch.mean(torch.abs(W_d[tgt, :, :, :])))
            else:
                mean_weights.append(torch.mean(torch.abs(W_rgb[tgt, :, :,:])))       
    return mean_weights

if __name__ == "__main__":
    SIZES=[128,32,64,64,64]
    SHAP_ARRAY_SIZES = [3*128, 32*128, 32*64, 64*64, 64*64]
    x_values = []
    y_values = []
    r_values = []
    conf_intervals = []
    # NEW: store weight correlations per layer
    r_w_cls = []
    r_w_grasp = []
    w_layer_names = []

    # Load model + weight matrices once
    _device = 'cpu'
    _model = get_model(_device)
    shap_array = np.zeros((5, 4096, 2))
    for layer in range(-1, 4):
        shap_arrays = []
        error_arrays = []
        for task_i, task in enumerate(["cls", "grasp"]):
            plus_mean_files = sorted(glob.glob(f'logs/svarm_run1/plus_mean_worker*_{task}_{layer}.npy'))
            if not plus_mean_files:
                raise RuntimeError("No worker files found.")
            # Assume all shapes equal
            shape = np.load(plus_mean_files[0]).shape

            agg_plus_mean  = np.zeros(shape)

            agg_plus_n     = np.zeros(shape, dtype=int)
            agg_plus_M2    = np.zeros(shape)
            agg_minus_mean = np.zeros(shape)
            agg_minus_n    = np.zeros(shape, dtype=int)
            agg_minus_M2   = np.zeros(shape)
            T = 0
            for f in plus_mean_files:
                wid = f.split('worker')[-1].split('.npy')[0]
                pm = np.load(f)
                pn = np.load(f.replace('plus_mean','plus_n'))
                pM2= np.load(f.replace('plus_mean','plus_M2'))
                mm = np.load(f.replace('plus_mean','minus_mean'))
                mn = np.load(f.replace('plus_mean','minus_n'))
                mM2= np.load(f.replace('plus_mean','minus_M2'))
                csv_file = f.replace('plus_mean','svarm_progress').replace('.npy', '.csv').replace('_worker', '')
                df = pd.read_csv(csv_file)
                used_row = df['used']
                if not used_row.empty:
                    final_value = used_row.iloc[-1]

                else:
                    final_value = None
                T += final_value
                # elementwise merge (vectorized)
                # Need loop because Welford merge not linear; do per index
                it = np.ndindex(shape)
                for idx in it:
                    m_p, n_p, M2_p = merge_welford(agg_plus_mean[idx], agg_plus_n[idx], agg_plus_M2[idx],
                                                    pm[idx], pn[idx], pM2[idx])
                    agg_plus_mean[idx], agg_plus_n[idx], agg_plus_M2[idx] = m_p, n_p, M2_p
                    m_m, n_m, M2_m = merge_welford(agg_minus_mean[idx], agg_minus_n[idx], agg_minus_M2[idx],
                                                    mm[idx], mn[idx], mM2[idx])
                    agg_minus_mean[idx], agg_minus_n[idx], agg_minus_M2[idx] = m_m, n_m, M2_m
            
            # Variances
            var_plus  = np.where(agg_plus_n  > 1, agg_plus_M2  / (agg_plus_n  - 1), np.inf)
            var_minus = np.where(agg_minus_n > 1, agg_minus_M2 / (agg_minus_n - 1), np.inf)
            H_n = harmonic(shape[0])
            err = 2 * H_n * (var_plus + var_minus) / T 
            error_arrays.append(err)
            #print(f"Estimator error  <= {err}")
            # Shapley
            shap = agg_plus_mean - agg_minus_mean
            shap_array[layer + 1, :len(shap), task_i] = shap
            shap_arrays.append(shap)
            var_phi = var_plus/agg_plus_n + var_minus/agg_minus_n


            var_shap = shap.var()
            est_var = var_plus /agg_plus_n + var_minus /agg_minus_n
            M_emp_t = np.mean(est_var) 
            nr = M_emp_t/var_shap
            print(f"N = {np.mean(agg_plus_n)}")
            print(f"Noise ratio = {nr}")
            if len(shap_arrays) == 2:
                flat0 = shap_arrays[0].flatten()
                flat1 = shap_arrays[1].flatten()
                total_shap = np.abs(flat0) + np.abs(flat1)
                
                # Compute correlation only on these indices
                x_values.append(flat0)
                y_values.append(flat1)
                res = corr_true_ci_from_mse_bounds(flat0,  flat1, error_arrays[0], error_arrays[1], K=1, n_boot=100, seed=42)
                print(res["interval"])
                # print("Observed r:", res["r_obs"])
                r_values.append(res["r_obs"])
                # print("Identified interval from bounds:", res["interval"])
                # print("95% CI (bootstrap + bounds):", res["ci"])
                conf_intervals.append((res['ci'][0], res['ci'][1]))
                # ---- NEW: correlate Shapley with weights for this layer ----
                shape = shap_arrays[0].shape
                matched_name = None
                matched_W = None
                transposed = False
                
                w_flat = get_connections_first(_model) if layer == -1 else get_connections(_model, layer=LAYERS[layer+1])
                r_cls, _ = pearsonr(np.abs(flat0) + np.abs(flat1), w_flat)
                r_gr,  _ = pearsonr(flat1, w_flat)
                r_w_cls.append(r_cls)
                r_w_grasp.append(r_gr)
                w_layer_names.append(f"{matched_name}{'.T' if transposed else ''}")
            else:
                pass
    plot_all_layer_scatter(x_values, y_values, r_values, conf_intervals)
    # ---- NEW: summary printout ----
    print("\nCorrelation with weights per layer:")
    for i, layer in enumerate(range(-1, 4)):
        print(f" layer {layer:>2}: r_cls={r_w_cls[i]: .3f}, r_grasp={r_w_grasp[i]: .3f}")
