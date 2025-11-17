import numpy as np
import sys
import glob
from scipy.stats import pearsonr
import pandas as pd
from sympy import * 

import numpy as np
import numpy as np

def corr_true_from_two_reps(x1, x2, y1, y2, n_boot=1000, seed=0):
    x1 = np.asarray(x1); x2 = np.asarray(x2)
    y1 = np.asarray(y1); y2 = np.asarray(y2)
    xb = 0.5*(x1+x2); yb = 0.5*(y1+y2); dx = x1-x2; dy = y1-y2

    var_dx = np.var(dx, ddof=1); var_dy = np.var(dy, ddof=1)
    var_eps_x = 0.5*var_dx; var_eps_y = 0.5*var_dy
    cov_dd = np.cov(dx, dy, ddof=1)[0,1]
    cov_err = 0.5*cov_dd

    var_xb = np.var(xb, ddof=1); var_yb = np.var(yb, ddof=1); cov_bb = np.cov(xb, yb, ddof=1)[0,1]

    cov_true = cov_bb - 0.25*cov_dd
    var_x_true = max(var_xb - 0.25*var_dx, 0.0)
    var_y_true = max(var_yb - 0.25*var_dy, 0.0)
    rho_true = cov_true / np.sqrt(max(var_x_true,1e-12)*max(var_y_true,1e-12))

    # Diagnostics as correlations
    def corr(u,v): return np.cov(u,v,ddof=1)[0,1] / np.sqrt(np.var(u,ddof=1)*np.var(v,ddof=1))
    diag = dict(
        r_err = cov_err / np.sqrt(var_eps_x*var_eps_y + 1e-18),
        r_dx_xb = corr(dx, xb), r_dy_yb = corr(dy, yb),
        r_dx_yb = corr(dx, yb), r_dy_xb = corr(dy, xb),
    )

    # Bootstrap CI (resample units)
    rng = np.random.default_rng(seed); n = len(x1); boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb_b = xb[idx]; yb_b = yb[idx]; dx_b = dx[idx]; dy_b = dy[idx]
        var_dx_b = np.var(dx_b, ddof=1); var_dy_b = np.var(dy_b, ddof=1)
        cov_true_b = np.cov(xb_b, yb_b, ddof=1)[0,1] - 0.25*np.cov(dx_b, dy_b, ddof=1)[0,1]
        var_x_true_b = max(np.var(xb_b, ddof=1) - 0.25*var_dx_b, 0.0)
        var_y_true_b = max(np.var(yb_b, ddof=1) - 0.25*var_dy_b, 0.0)
        boots.append(cov_true_b / np.sqrt(max(var_x_true_b,1e-12)*max(var_y_true_b,1e-12)))
    lo, hi = np.quantile(boots, [0.025, 0.975])

    return dict(rho_true=rho_true, ci=(lo,hi), cov_true=cov_true,
                var_x_true=var_x_true, var_y_true=var_y_true, diag=diag,
                var_eps_x=var_eps_x, var_eps_y=var_eps_y, cov_err=cov_err)
    
def true_corr_bounds_from_per_point_mse(xhat, yhat, tx_i, ty_i):
    """
    xhat, yhat: arrays of observed data (same length n)
    tx_i, ty_i: per-point MSE bounds E[(Xi - Xhat_i)^2] and E[(Yi - Yhat_i)^2]
    Returns: (lower_bound, upper_bound, details_dict)
    """
    xhat = np.asarray(xhat); yhat = np.asarray(yhat)
    tx_i = np.asarray(tx_i); ty_i = np.asarray(ty_i)
    assert xhat.shape == yhat.shape == tx_i.shape == ty_i.shape

    r_obs = np.corrcoef(xhat, yhat)[0, 1]
    s2_xhat = np.var(xhat, ddof=1)
    s2_yhat = np.var(yhat, ddof=1)

    T_x = float(np.mean(tx_i))
    T_y = float(np.mean(ty_i))

    r_x_min = max(0.0, 1.0 - (T_x / s2_xhat if s2_xhat > 0 else np.inf))
    r_y_min = max(0.0, 1.0 - (T_y / s2_yhat if s2_yhat > 0 else np.inf))

    if r_obs == 0.0:
        return 0.0, 0.0, dict(r_obs=r_obs, r_x_min=r_x_min, r_y_min=r_y_min, T_x=T_x, T_y=T_y)

    sign = np.sign(r_obs)
    denom = np.sqrt(max(r_x_min, 1e-12) * max(r_y_min, 1e-12))
    upper_mag = min(abs(r_obs) / denom, 1.0)
    lower = sign * abs(r_obs)
    upper = sign * upper_mag
    lo, hi = (lower, upper) if lower <= upper else (upper, lower)
    return lo, hi, dict(r_obs=r_obs, r_x_min=r_x_min, r_y_min=r_y_min, T_x=T_x, T_y=T_y)

def correlation_interval(phi_cls, phi_grasp,
                         nr_cls, nr_gr):

    # 2. Observed correlation
    r_obs, _ = pearsonr(phi_cls, phi_grasp)

    # 3. Attenuation factor
    if (nr_cls < 1) and (nr_gr < 1):
        A = np.sqrt((1 - nr_cls)*(1 - nr_gr))
    else:
        A = np.nan

    r_corr = np.clip(r_obs / A, -1, 1) if A>0 else np.nan

    # 4–5. CI via Fisher z
    n = len(phi_cls)
    if n > 5 and A>0:
        SE_obs = 1/np.sqrt(n - 3)
        SE_true = SE_obs / A
        zc = np.arctanh(np.clip(r_corr, -0.999999, 0.999999))
        half = 1.96 * SE_true
        lo = np.tanh(zc - half)
        hi = np.tanh(zc + half)
    else:
        lo = hi = np.nan

    # Deterministic attenuation bound
    if A>0:
        if r_obs >= 0:
            det_lo, det_hi = r_obs, r_obs / A
        else:
            det_lo, det_hi = r_obs / A, r_obs
    else:
        det_lo = det_hi = np.nan

    return {
        "r_obs": r_obs,
        "nr_cls": nr_cls,
        "nr_grasp": nr_gr,
        "attenuation_factor_A": A,
        "r_corrected": r_corr,
        "ci95": (lo, hi),
        "attenuation_bound": (det_lo, det_hi)
    }

def svarm_rep_diag(x1, x2, y1, y2):
    xb = 0.5*(np.asarray(x1)+np.asarray(x2))
    yb = 0.5*(np.asarray(y1)+np.asarray(y2))
    dx = np.asarray(x1)-np.asarray(x2)
    dy = np.asarray(y1)-np.asarray(y2)
    cov_err = 0.5 * np.cov(dx, dy, ddof=1)[0,1]  # estimate of Cov(eps_x, eps_y)
    stats = dict(
        cov_err=cov_err,
        cov_dx_xb=np.cov(dx, xb, ddof=1)[0,1],
        cov_dy_yb=np.cov(dy, yb, ddof=1)[0,1],
        cov_dx_yb=np.cov(dx, yb, ddof=1)[0,1],
        cov_dy_xb=np.cov(dy, xb, ddof=1)[0,1],
        var_eps_x=0.5*np.var(dx, ddof=1),
        var_eps_y=0.5*np.var(dy, ddof=1),
    )
    return stats
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

for i, task in enumerate(["cls", "grasp"]):
    plus_mean_files = sorted(glob.glob(f'plus_mean_worker*_{task}_{layer}.npy'))
    if not plus_mean_files:
        raise RuntimeError("No worker files found.")
    # Assume all shapes equal
    
    T = 0
    s_h_e = (0,len(plus_mean_files)//2, len(plus_mean_files))
    for i in range(2):
        shape = np.load(plus_mean_files[0]).shape

        agg_plus_mean  = np.zeros(shape)

        agg_plus_n     = np.zeros(shape, dtype=int)
        agg_plus_M2    = np.zeros(shape)
        agg_minus_mean = np.zeros(shape)
        agg_minus_n    = np.zeros(shape, dtype=int)
        agg_minus_M2   = np.zeros(shape)
        for f in plus_mean_files[s_h_e[i]:s_h_e[i+1]]:
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
        shap_arrays.append(agg_plus_mean - agg_minus_mean)
print(len(shap_arrays))
print(svarm_rep_diag(shap_arrays[0], shap_arrays[1], shap_arrays[2], shap_arrays[3]))