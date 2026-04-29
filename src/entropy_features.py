import numpy as np
from scipy.spatial import cKDTree
from numpy.lib.stride_tricks import sliding_window_view


def coarse_grain_time(
    time: np.ndarray, flux: np.ndarray,
    scale: int, cadence: float, min_fill_fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    W = scale * cadence
    t_start = time[0]
    n_windows = int(np.floor((time[-1] - t_start) / W))
    if n_windows < 1:
        return np.array([]), np.array([]), np.array([], dtype=bool) 
    edges = t_start + np.arange(n_windows + 1) * W
    t_cg = 0.5 * (edges[:-1] + edges[1:]) 
    bin_idx = np.searchsorted(edges[1:], time, side='left')
    np.clip(bin_idx, 0, n_windows - 1, out=bin_idx) 
    # bin_starts[k] = first index i such that bin_idx[i] >= k. Length n_windows+1.
    bin_starts = np.searchsorted(bin_idx, np.arange(n_windows + 1))
    counts = np.diff(bin_starts)
    nonempty = counts > 0 
    sums = np.zeros(n_windows)
    if nonempty.any():
        sums[nonempty] = np.add.reduceat(flux, bin_starts[:-1][nonempty]) 
    # Fill-fraction check
    min_pts = max(1, int(np.ceil(min_fill_fraction * scale)))
    fill_ok = counts >= min_pts 
    # Internal-gap check
    if len(time) > 1:
        is_big_gap = np.diff(time) > 1.5 * cadence
        cum = np.concatenate([[0], np.cumsum(is_big_gap)])
        ends = np.maximum(bin_starts[1:] - 1, bin_starts[:-1])
        gaps_in_window = cum[ends] - cum[bin_starts[:-1]]
        gap_ok = gaps_in_window == 0
    else:
        gap_ok = np.ones(n_windows, dtype=bool) 
    valid = fill_ok & gap_ok & nonempty 
    x_cg = np.full(n_windows, np.nan)
    np.divide(sums, counts, out=x_cg, where=(counts > 0))
    x_cg[~valid] = np.nan 
    return t_cg, x_cg, valid
 
def build_template_validity(window_valid: np.ndarray, m: int) -> np.ndarray:
    N = len(window_valid)
    if N <= m:
        return np.zeros(N, dtype=bool)
    cum = np.concatenate([[0], np.cumsum(window_valid.astype(np.int8))])
    rolling_sum = cum[m + 1:] - cum[:N - m]
    template_valid = np.zeros(N, dtype=bool)
    template_valid[:N - m] = rolling_sum == (m + 1)
    return template_valid

def sample_entropy_fast(x: np.ndarray, m: int, r: float, valid_mask: np.ndarray | None = None) -> float:
    T = sliding_window_view(x, m + 1).astype(np.float64)
    N = len(T)
    idx = np.where(valid_mask[:N])[0] if valid_mask is not None else np.arange(N)
    if len(idx) < 2:
        return np.nan
    T = T[idx]
    Nv = len(T)
    # For m>1 query neighbors is several times faster than count neighbours
    # On the other hand count neighbors is more memory efficient at m = 1
    if m == 1:
        tree_m  = cKDTree(T[:, :1])
        tree_m1 = cKDTree(T)
        A = (tree_m.count_neighbors(tree_m,  r, p=np.inf) - Nv) // 2
        B = (tree_m1.count_neighbors(tree_m1, r, p=np.inf) - Nv) // 2
    else: 
        tree_m = cKDTree(T[:, :m]) 
        pairs = tree_m.query_pairs(r, p=np.inf, output_type='ndarray')
        if len(pairs) == 0:
            return np.nan
        A = len(pairs)
        diffs = np.abs(T[pairs[:, 0], m] - T[pairs[:, 1], m])
        B = int((diffs < r).sum())
    if B == 0:
        B = 1
    return float(-np.log(B / A)) if A > 0 else np.nan

def mse_gap_aware(
    time: np.ndarray,
    flux: np.ndarray,
    scales: list[int],
    m: int = 1,
    tolerance_factor: float = 0.1,    
    min_fill_fraction: float = 0.9,
) -> np.ndarray:   
    cadence = np.median(np.diff(time))
    min_templates = 10 ** m
    mse_values = [] 
    r = None
    for scale in scales:
        time_cg, flux_cg, win_valid = coarse_grain_time(time, flux, scale, cadence, min_fill_fraction)
        if r is None:
            r = tolerance_factor * np.std(flux_cg[win_valid])
        template_valid = build_template_validity(win_valid, m)
        n_valid = template_valid.sum()
        if n_valid < min_templates:
            mse_values.append(np.nan)
        else:
            se = sample_entropy_fast(flux_cg, m, r, valid_mask=template_valid)
            mse_values.append(se)
    return np.array(mse_values)

def entropy_features(time: np.ndarray, flux: np.ndarray, sector: int, m: int, tolerance_factor: float) -> dict[str, float]:
    scales = list(range(1, 11)) if sector < 27 else list(range(1*3, 11*3, 3))
    mse = mse_gap_aware(time, flux, scales, m=m, tolerance_factor=tolerance_factor)
    return {f'mse_{k}': e for k, e in enumerate(mse)}
    
