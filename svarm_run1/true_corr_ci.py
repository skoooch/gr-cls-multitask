import numpy as np

def _interval_from_bounds(r_obs, s2x, s2y, Txb, Tyb):
    """
    Compute the disattenuation-based interval [L, U] for the true correlation
    using reliability lower bounds from theoretical MSE caps.
    """
    # Reliability lower bounds
    rx_min = max(0.0, 1.0 - (Txb / s2x if s2x > 0 else np.inf))
    ry_min = max(0.0, 1.0 - (Tyb / s2y if s2y > 0 else np.inf))

    if r_obs == 0.0:
        return 0.0, 0.0, rx_min, ry_min

    sign = np.sign(r_obs)
    # Upper magnitude under minimum reliability; clip to 1
    denom = np.sqrt(max(rx_min, 1e-12) * max(ry_min, 1e-12))
    upper_mag = min(abs(r_obs) / denom, 1.0)

    # Sign-preserving attenuation (Cov is unchanged under assumptions)
    lower = sign * abs(r_obs)
    upper = sign * upper_mag
    lo, hi = (lower, upper) if lower <= upper else (upper, lower)
    return lo, hi, rx_min, ry_min


def corr_true_ci_from_mse_bounds(
    xhat,
    yhat,
    tx_i,
    ty_i,
    K=1,
    n_boot=2000,
    seed=0,
):
    """
    95% CI for the true Pearson correlation using theoretical MSE bounds and bootstrap.

    Inputs
    - xhat, yhat: arrays of observed estimates (same length n). If you already averaged K runs, pass the averages here.
    - tx_i, ty_i: per-item single-run MSE bounds (scalars or arrays length n) for X and Y.
      If you already averaged K independent runs, set K accordingly; bounds are scaled by 1/K internally.
    - K: number of independent runs that were averaged to produce xhat/yhat (default 1).
    - n_boot: bootstrap replicates (default 2000).
    - seed: RNG seed.

    Returns
    - result dict with:
      r_obs: observed correlation
      interval: (L, U) identified interval at the full sample from bounds
      ci: (L_CI, U_CI) bootstrap 95% CI combining sampling + bounds
      rx_min, ry_min: reliability lower bounds at the full sample
      details: misc diagnostics
    """
    xhat = np.asarray(xhat)
    yhat = np.asarray(yhat)
    if xhat.shape != yhat.shape:
        raise ValueError("xhat and yhat must have the same shape")
    n = xhat.size

    # Broadcast bounds if scalars
    tx = np.asarray(tx_i)
    ty = np.asarray(ty_i)
    if tx.ndim == 0:
        tx = np.full(n, float(tx))
    if ty.ndim == 0:
        ty = np.full(n, float(ty))
    if tx.shape != (n,) or ty.shape != (n,):
        raise ValueError("tx_i and ty_i must be scalars or arrays of length n matching xhat/yhat")

    # Scale theoretical bounds for K-run averages
    Txb = float(np.mean(tx)) / max(int(K), 1)
    Tyb = float(np.mean(ty)) / max(int(K), 1)

    # Full-sample stats
    r_obs = float(np.corrcoef(xhat, yhat)[0, 1])
    s2x = float(np.var(xhat, ddof=1))
    s2y = float(np.var(yhat, ddof=1))

    L, U, rx_min, ry_min = _interval_from_bounds(r_obs, s2x, s2y, Txb, Tyb)

    # Bootstrap over items
    rng = np.random.default_rng(seed)
    Ls, Us = [], []
    for test_i in range(int(n_boot)):
        idx = rng.integers(0, n, n)  # resample items
        x_b = xhat[idx]
        y_b = yhat[idx]
        # Recompute sample stats on the resampled set
        r_b = float(np.corrcoef(x_b, y_b)[0, 1]) if np.var(x_b, ddof=1) > 0 and np.var(y_b, ddof=1) > 0 else 0.0
        s2x_b = float(np.var(x_b, ddof=1))
        s2y_b = float(np.var(y_b, ddof=1))
        # Use per-item theoretical bounds on the resampled set (still scaled by K)
        Txb_b = float(np.mean(tx[idx])) / max(int(K), 1)
        Tyb_b = float(np.mean(ty[idx])) / max(int(K), 1)

        L_b, U_b, _, _ = _interval_from_bounds(r_b, s2x_b, s2y_b, Txb_b, Tyb_b)
        Ls.append(L_b)
        Us.append(U_b)

    L_ci = float(np.quantile(Ls, 0.025))
    U_ci = float(np.quantile(Us, 0.975))

    # Ensure ordering
    if L_ci > U_ci:
        L_ci, U_ci = U_ci, L_ci

    return dict(
        r_obs=r_obs,
        interval=(L, U),
        ci=(max(-1.0, L_ci), min(1.0, U_ci)),
        rx_min=rx_min,
        ry_min=ry_min,
        details=dict(n=n, K=K, Txb=Txb, Tyb=Tyb, s2x=s2x, s2y=s2y),
    )