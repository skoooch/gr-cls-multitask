import numpy as np, glob, math

def merge_welford(m1,n1,M2_1,m2,n2,M2_2):
    if n1 == 0: return m2, n2, M2_2
    if n2 == 0: return m1, n1, M2_1
    delta = m2 - m1
    n = n1 + n2
    m = m1 + delta * (n2 / n)
    M2 = M2_1 + M2_2 + delta*delta * (n1*n2)/n
    return m, n, M2

def aggregate(layer_dir, alpha=0.05):
    plus_mean_files  = sorted(glob.glob(f'{layer_dir}/plus_mean_worker*.npy'))
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
    shapley = agg_plus_mean - agg_minus_mean
    z = 1.96 if alpha == 0.05 else \
        float(np.abs(np.round(np.sqrt(2)*math.erfcinv(alpha))))
    var_phi = var_plus/agg_plus_n + var_minus/agg_minus_n
    half = z * np.sqrt(var_phi)
    ci_low  = shapley - half
    ci_high = shapley + half

    np.save(f'{layer_dir}/agg_shapley.npy', shapley)
    np.save(f'{layer_dir}/agg_ci_low.npy', ci_low)
    np.save(f'{layer_dir}/agg_ci_high.npy', ci_high)
    np.save(f'{layer_dir}/agg_ci_half.npy', half)
    np.save(f'{layer_dir}/agg_plus_n.npy', agg_plus_n)
    np.save(f'{layer_dir}/agg_minus_n.npy', agg_minus_n)
    print("Done. Max half-width:", np.nanmax(half))

if __name__ == "__main__":
    import sys
    layer_dir = f'shap/connections/{sys.argv[1]}'  # e.g. shap/connections/1
    aggregate(layer_dir)