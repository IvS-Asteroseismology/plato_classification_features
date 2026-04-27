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
    minimum_peak_power: float,
    fmin: float,  fmax: float, 
    oversampling_factor: float,
) -> dict[str, float]:
    peak_idx = find_top_k_peaks(power, k=len(power), w=int(2 * oversampling_factor))
    peak_freqs = freqs[peak_idx[power[peak_idx] >= minimum_peak_power]]
    edges = [fmin, 1.5, 5.0, fmax]
    names = ["low", "mid", "high"]
    feats = {}
    for name, lo, hi in zip(names, edges[:-1], edges[1:]):
        width = hi - lo
        count = np.sum((peak_freqs >= lo) & (peak_freqs < hi))
        feats[f"ls_peak_density_{name}"] = (count / width).item()
    return feats

def periodogram_features(
    time: np.ndarray, flux: np.ndarray,
    oversampling_factor: float = 10.,
    finetune_factor: float = 10.,
    peak_density_maximum_fap: float = 0.01,
) -> dict[str, float]:
    freqs, power, (fmin, fmax, fres) = compute_periodogram(time, flux, oversampling_factor)
    # Fine tune ten highest peaks
    peak_idx = find_top_k_peaks(power, k=10, w=int(2 * oversampling_factor))
    for idx in peak_idx:
        freqs[idx], power[idx] = fine_tune_peak(time, flux, freqs[idx], fmin, fmax, fres, finetune_factor)
    dominant_idx = peak_idx[np.argmax(power[peak_idx])]
    power_dominant = power[dominant_idx]
    fap_dominant = baluev_false_alarm_probability(power_dominant, time, flux, fmin, fmax, oversampling_factor)
    eps = np.finfo(float).tiny
    fap_clipped = eps if fap_dominant < eps else fap_dominant   
    half_dominant_idx = np.argmin(np.abs(freqs - freqs[dominant_idx]/2))
    features = {
        'ls_frequency_dominant': freqs[dominant_idx].item(), 
        'ls_amplitude_dominant': power_dominant.item(),
        'ls_fap_dominant': fap_clipped.item(),
        'ls_amplitude_ratio_half_dominant': (power[half_dominant_idx]/power_dominant).item(),
        'ls_entropy': spectral_entropy(power).item(),
        'ls_flatness': spectral_flatness(power).item(),
    }
    minimum_peak_power = baluev_fap_level(peak_density_maximum_fap, time, flux, fmin, fmax, oversampling_factor)
    features.update(peak_density(freqs, power, minimum_peak_power, fmin, fmax, oversampling_factor))
    return features
