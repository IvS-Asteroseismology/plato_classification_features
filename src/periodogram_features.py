import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from nifty_ls import finufft
from astropy.timeseries import LombScargle


def frequency_grid(fmin: float, fmax: float, fres: float) -> np.ndarray:
    Nf = int((fmax - fmin) / fres)
    return fmin + np.arange(Nf) * fres

def lombscargle_periodogram(
    time: np.ndarray, flux: np.ndarray, 
    fmin: float, fmax: float, fres: float, 
    normalization: str = 'standard'
) -> np.ndarray:
    return finufft.lombscargle(
        t=time, y=flux, nthreads=1, 
        fmin=fmin, df=fres, Nf=int((fmax-fmin)/fres), 
        normalization=normalization
    )

def baluev_false_alarm_probability(
    power_level: float, 
    time: np.ndarray, flux: np.ndarray, 
    fmin: float, fmax: float, oversampling_factor: float,
    normalization: str = 'standard',
) -> np.ndarray:
    ls = LombScargle(time, flux, normalization=normalization)
    return ls.false_alarm_probability(
        power_level, method='baluev', 
        minimum_frequency=fmin, maximum_frequency=fmax, 
        samples_per_peak=oversampling_factor,
    )

def baluev_fap_level(
        level: float,
        time: np.ndarray, flux: np.ndarray,
        fmin: float, fmax: float, oversampling_factor: float,
        normalization: str = 'standard',
) -> float:
    ls = LombScargle(time, flux, normalization=normalization)
    return ls.false_alarm_level(
            level, method="baluev",
            minimum_frequency=fmin, maximum_frequency=fmax,
            samples_per_peak=oversampling_factor,
    )

def find_top_k_peaks(x: np.ndarray, k: int, w: int) -> np.ndarray:
    n = x.size
    if n < 2 * w + 1 or k <= 0:
        return np.array([], dtype=int)
    win = sliding_window_view(x, 2 * w + 1)
    center = x[w : n - w]
    is_peak = center == win.max(axis=1)
    peak_idx = np.arange(w, n - w)[is_peak]
    if peak_idx.size == 0:
        return np.array([], dtype=int)
    order = np.argsort(x[peak_idx])[::-1]
    return peak_idx[order[:k]]

def compute_periodogram(
    time: np.ndarray, flux: np.ndarray, 
    oversampling_factor: float = 10.0, 
    normalization: str = 'standard',
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    delta_time = np.median(time[1:] - time[:-1])
    time_span = time[-1] - time[0]
    fmin = 1 / time_span
    fmax = 0.5 / delta_time
    fres = 1 / (oversampling_factor * time_span)
    freqs = frequency_grid(fmin, fmax, fres)
    power = lombscargle_periodogram(time, flux, fmin, fmax, fres, normalization=normalization)
    return freqs, power, (fmin, fmax, fres)

def fine_tune_peak(
    time: np.ndarray, flux: np.ndarray, 
    fpeak: float, fmin: float, fmax: float, fres: float, 
    finetune_factor: float = 10.,
):
    f_lo = max(fmin, fpeak - fres)
    f_hi = min(fmax, fpeak + fres)
    fres_ft = fres / finetune_factor
    # Compute refined local periodogram
    power = lombscargle_periodogram(time, flux, f_lo, f_hi, fres_ft)
    freqs = frequency_grid(f_lo, f_hi, fres_ft)
    jmax = int(np.argmax(power))
    return freqs[jmax], power[jmax]    

def spectral_entropy(power: np.ndarray, eps: float = 1e-12) -> float:
    pnorm = power / (power.sum() + eps)
    H = -np.sum(pnorm * np.log(pnorm + eps))
    return H / np.log(len(pnorm) + eps)

def spectral_flatness(power: np.ndarray, eps: float = 1e-12) -> float:
    return np.exp(np.mean(np.log(power + eps))) / (power.mean() + eps)

def peak_density(
    freqs: np.ndarray, power: np.ndarray,
    snr_threshold: float,
    fmin: float, fmax: float,
    oversampling_factor: float,
) -> dict[str, float]:
    peak_idx = find_top_k_peaks(power, k=len(power), w=int(2 * oversampling_factor))
    edges = [fmin, 1.5, 5.0, fmax]
    names = ["low", "mid", "high"]
    feats = {}
    for name, lo, hi in zip(names, edges[:-1], edges[1:]):
        band = (freqs >= lo) & (freqs < hi)
        if band.sum() == 0:
            feats[f"ls_peak_density_{name}"] = 0.0
            continue
        local_median = np.median(power[band])
        in_band = (peak_idx,)  # filter peaks into this band
        band_peaks = peak_idx[(freqs[peak_idx] >= lo) & (freqs[peak_idx] < hi)]
        count = np.sum(power[band_peaks] >= snr_threshold * local_median)
        feats[f"ls_peak_density_{name}"] = (count / (hi - lo)).item()
    return feats


SUBHARMONIC_LABELS = {
    4: "quarter",
    2: "half",
}

def compute_subharmonic_amplitude_ratios(
    frequency: np.ndarray, power: np.ndarray, 
    dominant: float, min_power_level: float, local_window: int = 10, local_elevation_factor: float = 1.5,
) -> dict[str, float]:
    dominant_power = np.interp(dominant, frequency, power)
    features = {}
    for divisor, label in SUBHARMONIC_LABELS.items():
        sub_freq = dominant / divisor
        sub_idx = np.argmin(np.abs(frequency - sub_freq))
        sub_power = np.interp(sub_freq, frequency, power)
        
        lo = max(0, sub_idx - local_window)
        hi = min(len(power) - 1, sub_idx + local_window)
        local_median = np.median(power[lo:hi + 1])
        is_locally_elevated = sub_power > local_elevation_factor * local_median

        features[f"ls_amplitude_ratio_{label}_dominant"] = (sub_power / dominant_power).item()
        features[f"ls_amplitude_is_significant_{label}_dominant"] = sub_power > min_power_level
        features[f"ls_is_local_max_{label}_dominant"] = bool(is_locally_elevated)
    return features

def periodogram_features(
    time: np.ndarray, flux: np.ndarray,
    oversampling_factor: float = 10.,
    finetune_factor: float = 10.,
    peak_significance: float = 0.01,
    peak_snr_threshold: float = 5.
) -> dict[str, float]:
    freqs, power, (fmin, fmax, fres) = compute_periodogram(time, flux, oversampling_factor)
    # Fine tune ten highest peaks
    peak_idx = find_top_k_peaks(power, k=10, w=int(2 * oversampling_factor))
    for idx in peak_idx:
        freqs[idx], power[idx] = fine_tune_peak(time, flux, freqs[idx], fmin, fmax, fres, finetune_factor)
    dominant_idx = peak_idx[np.argmax(power[peak_idx])]
    freq_dominant, power_dominant = freqs[dominant_idx], power[dominant_idx], 
    fap_dominant = baluev_false_alarm_probability(power_dominant, time, flux, fmin, fmax, oversampling_factor)
    eps = np.finfo(float).tiny
    fap_clipped = eps if fap_dominant < eps else fap_dominant
    minimum_peak_power = baluev_fap_level(peak_significance, time, flux, fmin, fmax, oversampling_factor)
    features = {
        'ls_frequency_dominant': freq_dominant.item(), 
        'ls_amplitude_dominant': power_dominant.item(),
        'ls_amplitude_is_significant_dominant': power_dominant > minimum_peak_power,
        'ls_fap_dominant': fap_clipped.item(),
        'ls_entropy': spectral_entropy(power).item(),
        'ls_flatness': spectral_flatness(power).item(),
    }
    features.update(compute_subharmonic_amplitude_ratios(freqs, power, freq_dominant, minimum_peak_power))    
    features.update(peak_density(freqs, power, peak_snr_threshold, fmin, 24., oversampling_factor))
    return features
