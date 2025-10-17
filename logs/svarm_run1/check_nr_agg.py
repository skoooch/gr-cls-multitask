import numpy as np
import sys
import glob
from scipy.stats import pearsonr
import pandas as pd
from sympy import *
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

from true_corr_ci import corr_true_ci_from_mse_bounds
def plot_all_layer_scatter(X_arrays, Y_arrays, r_values, confidence_intervals):
    '''
    plot correlation in neuron shapley values between tasks across layers
    plot shapley graphs for each layer under the correlation graph
    '''

    
    fig, ax = plt.subplots(figsize=(10, 8))
    x_values = np.arange(1, len(r_values) + 1)
    n = 64 # can change number neurons in each layer

    # confidence_intervals = [confidence_interval(r, n) for r in r_values]
    lower_bounds = [ci[0] for ci in confidence_intervals]
    upper_bounds = [ci[1] for ci in confidence_intervals]

    ax.errorbar(x_values, r_values, 
                yerr=[np.abs(np.array(r_values) - np.array(lower_bounds)), 
                      np.abs(np.array(upper_bounds) - np.array(r_values))],
                fmt='o', capsize=5, linestyle='--', color="black")

    ax.set_xticks(x_values)
    ax.xaxis.set_major_locator(plt.FixedLocator([1, 3, 5]))
    # ax.set_xticklabels([str(i) for i in x_values if i % 2 == 1])  # Show only odd layers for clarity
    ax.axhline(0, color='gray', linewidth=0.7, linestyle='-')
    ax.set_ylabel('Correlation (r-value)', labelpad=2, fontsize=14)
    ax.set_xlabel('Feature extraction layer', labelpad=1.5, fontsize=14)
    ax.set_title('Correlation in Connection Shapley Values Between Tasks Across Layers', fontsize=18)

    # add inset axes below x-axis
    fig.subplots_adjust(bottom=0.35)
    for i in range(len(X_arrays)):
        x, y, r = X_arrays[i], Y_arrays[i], r_values[i], 
        inset_width = 0.12
        inset_height = 0.18
        left = 0.13 + i * (0.83 / len(X_arrays))  # even horizontal spacing
        bottom = 0.08

        inset_ax = fig.add_axes([left, bottom, inset_width, inset_height])
        inset_ax.scatter(x, y,s=(rcParams['lines.markersize']**2)/4 , alpha=0.7, color="black")
        inset_ax.set_title(f'r = {r:.2f}', fontsize=8)
        inset_ax.set_xlabel(f'Classification', fontsize=6, labelpad=1)
        inset_ax.set_ylabel(f'Grasp', fontsize=6, labelpad=1)
        inset_ax.tick_params(axis='both', which='major', labelsize=6)

    plt.savefig('a_vis/combined_below_graph_black.png', dpi=300)

def merge_welford(m1,n1,M2_1,m2,n2,M2_2):
    if n1 == 0: return m2, n2, M2_2
    if n2 == 0: return m1, n1, M2_1
    delta = m2 - m1
    n = n1 + n2
    m = m1 + delta * (n2 / n)
    M2 = M2_1 + M2_2 + delta*delta * (n1*n2)/n
    return m, n, M2
SIZES=[128,32,64,64,64]

x_values = []
y_values = []
r_values = []
conf_intervals = []
for layer in range(-1, 4):
    shap_arrays = []
    error_arrays = []
    for task in ["cls", "grasp"]:
        plus_mean_files = sorted(glob.glob(f'plus_mean_worker*_{task}_{layer}.npy'))
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
        print(f"Estimator error  <= {err}")
        # Shapley
        shap = agg_plus_mean - agg_minus_mean
        shap_arrays.append(shap)
        var_phi = var_plus/agg_plus_n + var_minus/agg_minus_n


        var_shap = shap.var()

        est_var = var_plus /agg_plus_n + var_minus /agg_minus_n
        M_emp_t = np.mean(est_var) 
        nr = M_emp_t/var_shap
        print(f"N = {np.mean(agg_plus_n)}")
        print(f"Noise ratio = {nr}")
        if len(shap_arrays) ==2:
            # Flatten and get indices of top 1000 absolute values for each array
            flat0 = shap_arrays[0].flatten()
            flat1 = shap_arrays[1].flatten()
            # Compute correlation only on these indices
            x_values.append(flat0)
            y_values.append(flat1)
            res = corr_true_ci_from_mse_bounds(flat0, flat1, error_arrays[0], error_arrays[1], K=1, n_boot=100, seed=42)
            print("Observed r:", res["r_obs"])
            r_values.append(res["r_obs"])
            print("Identified interval from bounds:", res["interval"])
            print("95% CI (bootstrap + bounds):", res["ci"])
            conf_intervals.append((res['ci'][0], res['ci'][1]))
        else:
            print("Expected two shap arrays to compute correlation.")
plot_all_layer_scatter(x_values, y_values, r_values, conf_intervals)