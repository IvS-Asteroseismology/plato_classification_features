import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt


def flux_statistics(flux: np.ndarray) -> dict[str, float]:
    N = len(flux)
    mean = np.mean(flux)
    std = np.std(flux, ddof=1) # unbiased estimators
    flux_std = (flux - mean) / std
    moments = {
        'flux_std': std,
        'flux_skewness': np.sum(flux_std ** 3) * N /((N-1)*(N-2)),
        'flux_kurtosis': np.sum(flux_std ** 4) * N * (N+1) / ((N-1)*(N-2)*(N-3)) - 3 * (N-1)**2 / ((N-2)*(N-3)),
    }
    ps = np.array([5., 10., 17.5, 25., 32.5, 40., 50., 60., 67.5, 75., 82.5, 90., 95.])    
    p5, p10, p17_5, p25, p32_5, p40, med, p60, p67_5, p75, p82_5, p90, p95 = np.percentile(flux, ps)
    iqr = p75 - p25
    pdiff_95_5 = p95 - p5
    abs_flux_minus_med = np.abs(flux - med)
    robust = {
        'flux_iqr': iqr,
        'flux_amplitude': 0.5 * pdiff_95_5,
        'flux_mad': np.median(abs_flux_minus_med),
        'flux_largest_departure': np.amax(abs_flux_minus_med)/med,
        'flux_percent_difference': pdiff_95_5 / med,
    }    
    ratios = {
        "flux_ratio_mid20": (p60 - p40) / pdiff_95_5,
        "flux_ratio_mid45": (p67_5 - p32_5) / pdiff_95_5,
        "flux_ratio_mid50": iqr / pdiff_95_5,
        "flux_ratio_mid65": (p82_5 - p17_5) / pdiff_95_5,
        "flux_ratio_mid80": (p90 - p10)  / pdiff_95_5,
    }
    all_features = {**moments, **robust, **ratios}
    return {k : v.item() for k, v in all_features.items()}

def max_slope(time: np.ndarray, flux: np.ndarray) -> dict[str, float]:
    dflux = flux[1:] - flux[:-1]
    dtime = time[1:] - time[:-1]
    assert np.all(dtime>0.0), "Times are not unique"
    return {'max_slope': np.amax(dflux/dtime).item()}

def _acf_fft_segment(x: np.ndarray, max_lag: int, center: str = 'median', scale: str = 'mad') -> np.ndarray:
    n = x.size
    if center == 'median':
        x = x - np.median(x)
    elif center == 'mean':
        x = x - np.mean(x)
    if scale == 'mad':
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad > 0:
            x = x / mad
    elif scale == 'std':
        s = np.std(x)
        if s > 0:
            x = x / s
    if max_lag >= n:
        max_lag = n - 1
    # FFT autocorrelation
    nfft = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(X * np.conj(X), n=nfft)[:n]
    ac = ac[:max_lag + 1]
    if ac[0] <= 0:
        return None
    return ac / ac[0]

def _split_by_time_gaps(t: np.ndarray, gap_factor: float) -> tuple[list[tuple[int, int]], float]:
    dt = np.diff(t)
    dt_nom = np.median(dt[np.isfinite(dt)])
    if not np.isfinite(dt_nom) or dt_nom <= 0:
        return [(0, t.size)]
    breaks = np.where(dt > gap_factor * dt_nom)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, t.size]
    return list(zip(starts, ends)), dt_nom

def acf_fft_from_time_flux(time, flux, max_lag_days, gap_factor=15, center='median', scale='mad'):
    segments, dt_nom = _split_by_time_gaps(time, gap_factor=gap_factor)
    if np.all(np.array([time[end-1]-time[start] for start, end in segments]) < max_lag_days):
        raise ValueError('Cannot compute acf, all segments are shorter than the desired maximum lag')
    max_lag = int(np.round(max_lag_days / dt_nom))
    rhos = []
    weights = []
    for start, end in segments:
        segment_flux = flux[start:end]
        if segment_flux.size < max_lag + 1:
            continue
        rho = _acf_fft_segment(segment_flux, max_lag=max_lag, center=center, scale=scale)
        if rho is None:
            continue
        rhos.append(rho)
        weights.append(segment_flux.size)
    if not rhos:
        raise ValueError("Cannot compute acf, no valid segments found")
    rhos = np.vstack(rhos)
    w = np.asarray(weights, dtype=float)
    w /= w.sum()
    rho_avg = (rhos.T @ w)
    return np.arange(0, len(rho_avg), 1)*dt_nom, rho_avg, dt_nom

def smooth_hann(x: np.ndarray, w: int) -> np.ndarray:
    if w % 2 == 0:
        raise ValueError("Window size has to be odd")
    win = np.hanning(w)
    win /= win.sum()
    y_valid = np.convolve(x, win, mode="valid")
    y = np.full_like(x, np.nan, dtype=float)
    half = w // 2
    y[half:half + y_valid.size] = y_valid
    return y

def first_two_peaks(
    lag: np.ndarray,
    rho: np.ndarray,
    min_peak_height: float,
    smooth_window_len: int = 5,
    min_peak_prominence: float = 0.01
) -> tuple[tuple[float, float], tuple[float, float]]:
    max_lag = lag[-1].item()
    # To decrease the change of bogus peaks we first smooth the acf with a Hann window
    smooth_rho = smooth_hann(rho, w=smooth_window_len)
    peak_idx, _ = find_peaks(smooth_rho, height=min_peak_height, prominence=min_peak_prominence, wlen=10)
    # If no peaks are found we simply output the maximum lag
    if len(peak_idx) == 0:
        return (max_lag, 0.0), (max_lag, 0.0)
    elif len(peak_idx) == 1:
        i1 = peak_idx[0]
        return (lag[i1].item(), rho[i1].item()), (max_lag, 0.0)
    else:
        i1, i2 = peak_idx[np.argsort(lag[peak_idx])[:2]]
        return (lag[i1].item(), rho[i1].item()), (lag[i2].item(), rho[i2].item())

def autocorrelation_features(
    time: np.ndarray,
    flux: np.ndarray,
    max_lag_days: float = 4.0,
    min_peak_height: float = 1/8,
    min_peak_prominence: float = 0.01
) -> dict[str, float]:
    # ACF is computed per segment and then averaged
    lag, rho, dt = acf_fft_from_time_flux(time, flux, max_lag_days=max_lag_days, gap_factor=15)
    # Lag at which correlation first goes below 0.5
    lag_first = lag[rho < 0.5][0]
    basic_descriptors = {'acf_lag_first_below_05': lag_first.item(), 'acf_val_lag1': rho[1].item()}
    # Lags and values for the first and second local maxima
    p1, p2 = first_two_peaks(lag, rho, min_peak_height=min_peak_height, min_peak_prominence=min_peak_prominence)
    peak_structure = {'acf_lag_peak1': p1[0], 'acf_val_peak1': p1[1], 'acf_lag_peak2': p2[0], 'acf_val_peak2': p2[1]}
    return {**basic_descriptors, **peak_structure}
