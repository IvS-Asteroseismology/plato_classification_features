import numpy as np
  
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
