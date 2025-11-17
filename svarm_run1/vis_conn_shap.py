import numpy as np
from scipy.stats import pearsonr
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