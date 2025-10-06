import numpy as np
import sys
import glob
from scipy.stats import pearsonr
def merge_welford(m1,n1,M2_1,m2,n2,M2_2):
    if n1 == 0: return m2, n2, M2_2
    if n2 == 0: return m1, n1, M2_1
    delta = m2 - m1
    n = n1 + n2
    m = m1 + delta * (n2 / n)
    M2 = M2_1 + M2_2 + delta*delta * (n1*n2)/n
    return m, n, M2
SIZES=[128,32,64,64,64]
layer = sys.argv[1]
shap_arrays = []
for task in ["grasp", "cls"]:
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

    for f in plus_mean_files:
        wid = f.split('worker')[-1].split('.npy')[0]
        pm = np.load(f)
        pn = np.load(f.replace('plus_mean','plus_n'))
        pM2= np.load(f.replace('plus_mean','plus_M2'))
        mm = np.load(f.replace('plus_mean','minus_mean'))
        mn = np.load(f.replace('plus_mean','minus_n'))
        mM2= np.load(f.replace('plus_mean','minus_M2'))

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
    if len(shap_arrays) == 2:
        # Flatten and get indices of top 1000 absolute values for each array
        flat0 = shap_arrays[0].flatten()
        flat1 = shap_arrays[1].flatten()
        idx0 = np.argpartition(np.abs(flat0), -1000)[-1000:]
        idx1 = np.argpartition(np.abs(flat1), -1000)[-1000:]
        # Union of indices
        top_idx = np.unique(np.concatenate([idx0, idx1]))
        # Compute correlation only on these indices
        corr, p = pearsonr(flat0, flat1)
        print(f"Correlation between shap values: {corr}, p={p}")

        # Print indices of the biggest 5 shap values for each array
        top5_idx0 = np.argpartition(np.abs(flat0), -5)[-5:]
        top5_idx1 = np.argpartition(np.abs(flat1), -5)[-5:]
        print(f"Indices of biggest 5 shap values grASP (layer {int(layer)}): {top5_idx0// SIZES[int(layer)+1]}")
        print(f"Indices of biggest 5 shap values grasp(layer {int(layer) + 1}): {top5_idx0 % SIZES[int(layer)]}")
        print(f"Indices of biggest 5 shap values cls (layer {int(layer)}): {top5_idx1// SIZES[int(layer)+1]}")
        print(f"Indices of biggest 5 shap values cls(layer {int(layer) + 1}): {top5_idx1 % SIZES[int(layer)]}")
    else:
        print("Expected two shap arrays to compute correlation.")