import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from nifty_ls import finufft
from astropy.timeseries import LombScargle

def frequency_grid(fmin: float, fmax: float, fres: float) -> np.ndarray:
    Nf = int((fmax - fmin) / fres)
    return fmin + np.arange(Nf) * fres

def lombscargle_periodogram(time: np.ndarray, flux: np.ndarray, fmin: float, fmax: float, fres: float, normalization: str = 'standard') -> np.ndarray:
    return finufft.lombscargle(t=time, y=flux, nthreads=1, fmin=fmin, df=fres, Nf=int((fmax-fmin)/fres), normalization=normalization)

def baluev_false_alarm_probability(power_level: float, time: np.ndarray, flux: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    ls = LombScargle(time, flux, normalization='standard')
    return ls.false_alarm_probability(power_level, method='baluev', minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=10)

def find_top_k_peaks(x: np.ndarray, k: int) -> np.ndarray:
    local_maxima = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
    idx = np.argsort(x[1:-1][local_maxima])[::-1][:k]
    return np.arange(1, len(x)-1)[local_maxima][idx]

def find_top_k_peaks_window_fast(x: np.ndarray, k: int, w: int) -> np.ndarray:
    n = x.size
    if n == 0 or k <= 0:
        return np.array([], dtype=int)
    if w < 1:
        raise ValueError("w must be >= 1")
    if n < 2 * w + 1:
        return np.array([], dtype=int)
    win = sliding_window_view(x, 2 * w + 1)
    center = x[w:n - w]
    wmax = win.max(axis=1)
    is_peak = center == wmax
    # is_peak &= (win == wmax[:, None]).sum(axis=1) == 1
    peak_idx = (np.arange(w, n - w))[is_peak]
    if peak_idx.size == 0:
        return np.array([], dtype=int)
    order = np.argsort(x[peak_idx])[::-1]
    return peak_idx[order[:k]]

def compute_lombscargle_periodogram(time: np.ndarray, flux: np.ndarray, oversampling_factor: float = 10.0, normalization: str = 'standard') -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    delta_time = np.median(time[1:] - time[:-1])
    time_span = time[-1] - time[0]
    fmin, fmax, fres = 1/time_span, 0.5/delta_time, 1/(oversampling_factor * time_span)
    freqs = frequency_grid(fmin, fmax, fres)
    ampl = lombscargle_periodogram(time, flux, fmin, fmax, fres, normalization=normalization)
    return freqs, ampl, (fmin, fmax, fres)

def fine_tune_lombscargle_peak(time: np.ndarray, flux: np.ndarray, fcenter: float, fmin: float, fmax: float, fres: float, finetune_oversampling_factor: float = 10):
    f_lo = max(fmin, fcenter - fres)
    f_hi = min(fmax, fcenter + fres)
    fres_ft = fres / finetune_oversampling_factor
    # Compute refined local periodogram
    ampl_ref = lombscargle_periodogram(time, flux, f_lo, f_hi, fres_ft)
    freqs_ref = frequency_grid(f_lo, f_hi, fres_ft)
    jmax = int(np.argmax(ampl_ref))
    amax = ampl_ref[jmax]
    fmax_local = freqs_ref[jmax]
    return fmax_local, amax

def spectral_entropy(power: np.ndarray, eps: float = 1e-12) -> float:
    total_power = power.sum()
    pnorm = power / (total_power + eps)
    H = -np.sum(pnorm * np.log(pnorm + eps))
    Hnorm = H / np.log(len(pnorm) + eps) 
    return Hnorm

def spectral_flatness(power: np.ndarray, eps: float = 1e-12) -> float:
    arithmetic_mean = power.mean()    
    geometric_mean = np.exp(np.mean(np.log(power + eps)))
    return (geometric_mean / (arithmetic_mean + eps))

def find_harmonic_relation(freq_test: float, freq_ref: float, max_harmonic: int = 5) -> tuple[float, float]:
    kf = np.concatenate([np.arange(2, max_harmonic, 1), 1/np.arange(2, max_harmonic, 1)])
    distance = np.abs(freq_test - kf * freq_ref)
    closest_harmonic = np.argmin(distance)
    return kf[closest_harmonic], distance[closest_harmonic]   

def periodogram_features(
    time: np.ndarray,
    flux: np.ndarray,
    oversampling_factor: float = 10.,
    finetune_oversampling_factor: float = 10.,
    #num_peaks_for_peak_structure: int = 3
) -> dict[str, float]:
    freqs, ampl, (fmin, fmax, fres) = compute_lombscargle_periodogram(time, flux, oversampling_factor=oversampling_factor)
    # Fine tune highest peaks
    peak_idx = find_top_k_peaks_window_fast(ampl, k=10, w=int(2*oversampling_factor))
    for idx in peak_idx:
        fmax_local, amax = fine_tune_lombscargle_peak(time, flux, freqs[idx], fmin, fmax, fres, finetune_oversampling_factor)
        ampl[idx] = amax
        freqs[idx] = fmax_local
    periodogram_metrics = {
        'ls_entropy': spectral_entropy(ampl).item(),
        'ls_flatness': spectral_flatness(ampl).item(),
    }
    peak_idx = peak_idx[np.argsort(ampl[peak_idx])[::-1]] # Sort in amplitude decreasing order
    fundamental_idx = peak_idx[0]
    ampl_fundamental = ampl[fundamental_idx]
    fundamental_fap = baluev_false_alarm_probability(ampl_fundamental, time, flux, fmin, fmax)
    eps = np.finfo(float).tiny
    fap_clipped = eps if fundamental_fap < eps else fundamental_fap   
    half_dominant_idx = np.argmin(np.abs(freqs - freqs[fundamental_idx]/2))
    dominant_frequency = {
        'ls_frequency_dominant': freqs[fundamental_idx].item(), 
        'ls_amplitude_dominant': ampl_fundamental.item(),
        'ls_fap_dominant': fap_clipped.item(),
        'ls_amplitude_ratio_half_dominant': (ampl[half_dominant_idx]/ampl_fundamental).item(),
    }    
    #for i, p in enumerate(peak_idx[1:num_peaks_for_peak_structure+1]):
    #    harm_k, harm_dist = find_harmonic_relation(freqs[p], freqs[fundamental_idx])
    #    peak_structure[f'ls_frequency_top{i+2}'] = freqs[p].item()
    #    peak_structure[f'ls_amplitude_ratio_top{i+2}_fundamental'] = (ampl[p]/ampl_fundamental).item()
    #    peak_structure[f'ls_distance_to_fundamental_harmonic_top{i+2}'] = harm_dist.item()
    #    peak_structure[f'ls_closest_harmonic_top{i+2}'] = harm_k.item()
    return {**dominant_frequency, **periodogram_metrics}

def fourier_series_design_matrix(time: np.ndarray, fundamental_frequency: float, num_components: int) -> np.ndarray:
    omegat = 2 * np.pi * fundamental_frequency * time
    cos_terms = [np.cos(omegat*j) for j in range(1, num_components+1)]
    sin_terms = [np.sin(omegat*j) for j in range(1, num_components+1)]
    return np.concatenate([[np.ones_like(time)] + cos_terms + sin_terms]).T

def fit_fourier_series(time: np.ndarray, flux: np.ndarray, fundamental_frequency: float, num_components: int):
    assert num_components>0, "At least one sinusoidal  component needed"
    X = fourier_series_design_matrix(
        time, fundamental_frequency=fundamental_frequency, num_components=num_components, 
    )
    theta, sum_squared_residuals, rank, s = np.linalg.lstsq(X, flux, rcond=None)
    # Representation for a model A0 + sum_j Aj cos( j omega time - PHj)
    theta_cos, theta_sin = np.array_split(theta[1:], 2)
    amplitudes = np.sqrt(theta_cos ** 2 + theta_sin ** 2)
    phases = np.arctan2(theta_sin, theta_cos)
    return theta, amplitudes, phases, sum_squared_residuals

def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2*np.pi) - np.pi

def fourier_series_features(
    time: np.ndarray, 
    flux: np.ndarray, 
    fundamental_frequency: float, 
    num_components: int, 
    adjust_with_harmonic_ratio: bool = True, 
    return_relative_amplitudes: bool = True,
) -> dict[str, float]:    
    _, amplitudes, phases, sum_squared_residuals = fit_fourier_series(
        time, flux, fundamental_frequency, num_components, 
    )
    # First phase will be reference
    if adjust_with_harmonic_ratio: # As in Debosscher et al. 2007
        k = np.arange(2, len(phases) + 1)
        relative_phases = wrap_to_pi(phases[1:] - k*phases[0])
    else: # As in Richards et al. 2012
        relative_phases = wrap_to_pi(phases[1:] - phases[0])
    r2 = 1. - sum_squared_residuals[0].item()/np.sum((flux - np.mean(flux))**2)
    output_r2 = {'fourier_r2_fit': r2.item()}
    output_amplitudes = {'fourier_amplitude_1': amplitudes[0].item()}    
    if num_components > 1:
        if not return_relative_amplitudes:
            output_amplitudes = output_amplitudes | {f'fourier_amplitude_{j+2}': a.item() for j, a in enumerate(amplitudes[1:])}
        else:
            output_amplitudes = output_amplitudes | {f'fourier_amplitude_{j+2}': (a/amplitudes[0]).item() for j, a in enumerate(amplitudes[1:])}
        output_phases = {f'fourier_relative_phase_{j+2}': p.item() for j, p in enumerate(relative_phases)}
    return {**output_amplitudes, **output_phases, **output_r2}
