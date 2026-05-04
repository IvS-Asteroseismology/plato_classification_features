import numpy as np
from scipy.signal import find_peaks

def acf_gapped(
    time: np.ndarray,
    flux: np.ndarray,
    max_lag_days: float,
    center: str = "median",
    scale: str = "mad",
    min_pair_frac: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # Uniform grid with guard against duplicates
    dt = np.median(np.diff(time))
    n_grid = int(np.round((time[-1] - time[0]) / dt)) + 1
    idx = np.round((time - time[0]) / dt).astype(int) 
    idx = np.clip(idx, 0, n_grid - 1)
 
    f = np.zeros(n_grid, dtype=np.float64)
    w = np.zeros(n_grid, dtype=np.float64)
     
    # if two samples fall in the same bin, average them
    np.add.at(f, idx, flux)
    np.add.at(w, idx, 1.0)
    occupied = w > 0
    f[occupied] /= w[occupied]
    w[occupied] = 1.0
 
    obs = f[occupied]
    if center == "median":
        c = np.median(obs)
    else:
        c = np.mean(obs)
    f[occupied] -= c 
    if scale == "mad":
        mad = np.median(np.abs(f[occupied]))
        s = 1.4826 * mad if mad > 0 else 1.0
    else:
        s = float(np.std(f[occupied])) or 1.0
    f[occupied] /= s
    f[~occupied] = 0.0 
    
    # FFT autocorrelation
    max_lag = min(int(np.round(max_lag_days / dt)), n_grid - 1)
    nfft = 1 << (2 * n_grid - 1).bit_length() 
    F = np.fft.rfft(f, n=nfft)
    W = np.fft.rfft(w, n=nfft) 
    num = np.fft.irfft(F * np.conj(F), n=nfft)[:max_lag + 1]
    den = np.fft.irfft(W * np.conj(W), n=nfft)[:max_lag + 1] 
    # Unbiased ratio
    min_pairs = max(1.0, min_pair_frac * den[0])
    valid = den >= min_pairs 
    ratio = np.full(max_lag + 1, np.nan, dtype=np.float64)
    ratio[valid] = num[valid] / den[valid]
    if np.isnan(ratio[0]) or ratio[0] <= 0:
        raise ValueError("ACF lag-0 is non-positive; cannot normalise")
    rho = ratio / ratio[0] 
    lag = np.arange(max_lag + 1) * dt
    den_frac = den / den[0] # For diagnostics 
    return lag, rho, den_frac, dt

def _hann_smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w < 3:
        return x.copy()
    if w % 2 == 0:
        w += 1
    win = np.hanning(w)
    win /= win.sum()
    y_valid = np.convolve(x, win, mode="valid")
    y = np.full_like(x, np.nan, dtype=np.float64)
    half = w // 2
    y[half : half + y_valid.size] = y_valid
    return y 
 
def _smooth_window_samples(smooth_days: float, dt: float) -> int:
    n = int(np.round(smooth_days / dt))
    n = max(n, 3)
    if n % 2 == 0:
        n += 1
    return n

def _interp_rho(lag: np.ndarray, rho: np.ndarray, t: float) -> float:
    valid = ~np.isnan(rho)
    if valid.sum() < 2:
        return np.nan
    return float(np.interp(t, lag[valid], rho[valid]))

def _parabolic_refine(y: np.ndarray, i: int) -> tuple[float, float]:
    if i <= 0 or i >= len(y) - 1 or np.isnan(y[i - 1]) or np.isnan(y[i + 1]):
        return float(i), float(y[i])
    a, b, c = y[i - 1], y[i], y[i + 1]
    denom = a - 2 * b + c
    if denom == 0:
        return float(i), float(b)
    di = 0.5 * (a - c) / denom
    yi = b - 0.25 * (a - c) * di
    return i + di, float(yi)

def _interpolated_crossing(lag: np.ndarray, rho: np.ndarray, level: float) -> float:
    for i in range(len(rho) - 1):
        if np.isnan(rho[i]) or np.isnan(rho[i + 1]):
            continue
        if rho[i] >= level > rho[i + 1]:
            # linear interpolation
            frac = (rho[i] - level) / (rho[i] - rho[i + 1])
            return float(lag[i] + frac * (lag[i + 1] - lag[i]))
    return float(lag[~np.isnan(rho)][-1]) if np.any(~np.isnan(rho)) else np.nan
    
def _detect_peaks(
    lag: np.ndarray,
    rho: np.ndarray,
    dt: float,
    min_peak_height: float = 0.10,
    min_peak_prominence: float = 0.01,
    smooth_days: float = 0.02,
    wlen_days: float = 1.0,
    min_lag_days: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sw = _smooth_window_samples(smooth_days, dt)
    smooth_rho = _hann_smooth(rho, sw)
    wlen_samp = max(3, int(np.round(wlen_days / dt)))
    if wlen_samp % 2 == 0: # wlen must be odd for find_peaks
        wlen_samp += 1
    sr = np.where(np.isnan(smooth_rho), -1.0, smooth_rho) 
    peak_idx, props = find_peaks(
        sr,
        height=min_peak_height,
        prominence=min_peak_prominence,
        wlen=wlen_samp,
    )
    # Discard peaks to close to zero lag
    min_lag_samp = max(1, int(np.round(min_lag_days / dt)))     
    peak_idx = peak_idx[peak_idx >= min_lag_samp] 
    # Discard peaks that sit in the last point
    valid_idx = np.where(~np.isnan(rho))[0]
    if len(valid_idx) > 0:
        last_valid = valid_idx[-1]
        peak_idx = peak_idx[peak_idx < last_valid - 1]
    # discard peaks that sit on the initial descent from rho=1
    keep = []
    for pi in peak_idx:
        preceding = rho[1:pi]  # skip lag-0 which is always 1
        valid_preceding = preceding[~np.isnan(preceding)]
        if len(valid_preceding) > 0 and np.min(valid_preceding) < min_peak_height:
            keep.append(pi)
    peak_idx = np.array(keep, dtype=int)
    if len(peak_idx) == 0:
        return np.array([]), np.array([]), np.array([], dtype=int)
 
    # refine each peak parabolically
    refined_lags = np.empty(len(peak_idx))
    refined_vals = np.empty(len(peak_idx))
    for j, pi in enumerate(peak_idx):
        ri, rv = _parabolic_refine(rho, pi)
        refined_lags[j] = ri * dt
        refined_vals[j] = rv
    order = np.argsort(refined_lags)
    return refined_lags[order], refined_vals[order], peak_idx[order]

def autocorrelation_features(
    time: np.ndarray,
    flux: np.ndarray,
    max_lag_days: float,
    center: str = "median",
    scale: str = "mad",
    min_peak_height: float = 0.10,
    min_peak_prominence: float = 0.01,
    smooth_days: float = 0.015, # 20 mins
    wlen_days: float = 1.0,
    min_pair_frac: float = 0.05,
) -> dict[str, float]:
    lag, rho, den_frac, dt_used = acf_gapped(
        time, flux,
        max_lag_days=max_lag_days,
        center=center,
        scale=scale,
        min_pair_frac=min_pair_frac,
    ) 
    features: dict[str, float] = {}
    # ACF at 30 mins
    features["acf_rho_at_30m"] = _interp_rho(lag, rho, 0.02)
    # Decorrelation timescales
    features["acf_tau_half"] = _interpolated_crossing(lag, rho, 0.5)
    features["acf_tau_e"] = _interpolated_crossing(lag, rho, 1.0 / np.e)
    features["acf_tau_zero"] = _interpolated_crossing(lag, rho, 0.0)
    # Peak structure
    peak_lags, peak_vals, peak_idx = _detect_peaks(
        lag, rho, dt_used,
        min_peak_height=min_peak_height,
        min_peak_prominence=min_peak_prominence,
        smooth_days=smooth_days,
        wlen_days=wlen_days,
    ) 
    n_peaks = len(peak_lags)
    features["acf_n_peaks"] = float(n_peaks)
    # The first two peaks
    features["acf_peak1_lag"] = float(peak_lags[0]) if n_peaks >= 1 else max_lag_days
    features["acf_peak1_val"] = float(peak_vals[0]) if n_peaks >= 1 else 0.0
    features["acf_peak2_lag"] = float(peak_lags[1]) if n_peaks >= 2 else max_lag_days
    features["acf_peak2_val"] = float(peak_vals[1]) if n_peaks >= 2 else 0.0
    # FWHM of the first peak
    features["acf_peak1_fwhm"] = max_lag_days
    if n_peaks >= 1:
        half_height = peak_vals[0] / 2.0
        pi = peak_idx[0]
        # left HWHM
        left_hw = np.nan
        for j in range(pi, 0, -1):
            if np.isnan(rho[j]) or np.isnan(rho[j - 1]):
                continue
            if rho[j] >= half_height > rho[j - 1]:
                frac = (rho[j] - half_height) / (rho[j] - rho[j - 1])
                left_hw = peak_lags[0] - (lag[j] - frac * dt_used)
                break
        # right HWHM
        right_hw = np.nan
        for j in range(pi, len(rho) - 1):
            if np.isnan(rho[j]) or np.isnan(rho[j + 1]):
                continue
            if rho[j] >= half_height > rho[j + 1]:
                frac = (rho[j] - half_height) / (rho[j] - rho[j + 1])
                right_hw = (lag[j] + frac * dt_used) - peak_lags[0]
                break
        if not (np.isnan(left_hw) or np.isnan(right_hw)):
            features["acf_peak1_fwhm"] = left_hw + right_hw        
    # Harmonic ratio between first peak and the peak at twice that lag
    if n_peaks >= 1:
        rho_2p = _interp_rho(lag, rho, 2 * peak_lags[0])
        features["acf_harmonic_ratio"] = rho_2p / peak_vals[0] 
    else:
        features["acf_harmonic_ratio"] = 0.0 
    # Peak spacing regularity for sources with at least 3 peaks
    if n_peaks >= 3:
        spacings = np.diff(peak_lags)
        features["acf_peak_spacing_mean"] = float(np.mean(spacings))
        features["acf_peak_spacing_cv"] = float(np.std(spacings) / np.mean(spacings))
    else:
        features["acf_peak_spacing_mean"] = max_lag_days
        features["acf_peak_spacing_cv"] = max_lag_days 
    # Integrated absolute correlation
    valid_mask = ~np.isnan(rho)
    rho_valid = rho[valid_mask]
    lag_valid = lag[valid_mask]
    span = lag_valid[-1] - lag_valid[0]
    if len(rho_valid) > 1 and span > 0:        
        features["acf_int_abs_rho"] = float(np.trapezoid(np.abs(rho_valid), lag_valid) / span)
    else:
        features["acf_int_abs_rho"] = 0.0
    return features
