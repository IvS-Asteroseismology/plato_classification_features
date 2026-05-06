import numpy as np
from scipy.signal import savgol_filter

from light_curves import find_segments

"""
Features from Richards et al. 2012, Nun et al. 2015 and references therein
"""

def flux_statistics(flux: np.ndarray) -> dict[str, float]:
    N = len(flux)
    mean = np.mean(flux)
    std = np.std(flux, ddof=1) # unbiased estimators
    flux_std = (flux - mean) / std
    moments = {
        'std': std,
        'skewness': np.sum(flux_std ** 3) * N /((N-1)*(N-2)),
        'kurtosis': np.sum(flux_std ** 4) * N * (N+1) / ((N-1)*(N-2)*(N-3)) - 3 * (N-1)**2 / ((N-2)*(N-3)),
        'stetson_K': (np.mean(np.abs(flux - mean)) / np.sqrt(np.mean((flux - mean)**2))), # unweighted
    }
    ps = np.array([5., 10., 17.5, 25., 32.5, 40., 50., 60., 67.5, 75., 82.5, 90., 95.])    
    p5, p10, p17_5, p25, p32_5, p40, med, p60, p67_5, p75, p82_5, p90, p95 = np.percentile(flux, ps)
    iqr = p75 - p25
    pdiff_95_5 = p95 - p5
    abs_flux_minus_med = np.abs(flux - med)
    amplitude = 0.5 * pdiff_95_5
    robust = {
        'iqr': iqr,
        'amplitude': amplitude,
        'mad': np.median(abs_flux_minus_med),
        'asymmetry_p': ((p95 - med) - (med - p5)) / (p95 - p5),
        'largest_departure': np.amax(abs_flux_minus_med)/med,
        'percent_difference': pdiff_95_5 / med,
        'median_bpr': np.mean(np.abs(flux - med) < 0.1 * amplitude),
        # 'eta': np.sum(np.diff(flux)**2) / ((N-1) * np.var(flux, ddof=1)), # Shows dependency on cadence
    }    
    ratios = {
        "ratio_mid20": (p60 - p40) / pdiff_95_5,
        "ratio_mid45": (p67_5 - p32_5) / pdiff_95_5,
        "ratio_mid50": iqr / pdiff_95_5,
        "ratio_mid65": (p82_5 - p17_5) / pdiff_95_5,
        "ratio_mid80": (p90 - p10)  / pdiff_95_5,
    }
    all_features = {**moments, **robust, **ratios}
    return {'flux_'+k : v.item() for k, v in all_features.items()}

def time_flux_statistics(
    time: np.ndarray, flux: np.ndarray,
    derivative_window: float = 0.0625, 
    eps: float = 1e-30,
) -> dict[str, float]:
    median_dt = np.median(time[1:] - time[:-1])
    window_pts = int(derivative_window / median_dt)
    if window_pts % 2 == 0:
        window_pts += 1
    window_pts = max(window_pts, 5)
    slopes = savgol_filter(flux, window_pts, polyorder=2, deriv=1, delta=median_dt)

    slope_p25, slope_med, slope_p75= np.percentile(slopes, np.array([25., 50., 75.]))
    slope_mad = np.median(np.abs(slopes - slope_med))
    slope_iqr = slope_p75 - slope_p25    

    return {
        'time_flux_slope_abs_max': np.abs(slopes).max().item(),
        'time_flux_slope_mad': slope_mad.item(),
        'time_flux_slope_iqr': slope_iqr.item(),
        'time_flux_slope_skewness': (np.mean((slopes - slope_med)**3) / (slope_mad**3 + eps)).item(),        
    }