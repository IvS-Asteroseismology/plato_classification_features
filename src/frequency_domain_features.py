import numpy as np
from nifty_ls import finufft
from astropy.timeseries import LombScargle

def frequency_grid(fmin: float, fmax: float, fres: float) -> np.ndarray:
    Nf = int((fmax - fmin) / fres)
    return fmin + np.arange(Nf) * fres

def lombscargle_periodogram(time: np.ndarray, flux: np.ndarray, fmin: float, fmax: float, fres: float) -> np.ndarray:
    return finufft.lombscargle(t=time, y=flux, nthreads=1, fmin=fmin, df=fres, Nf=int((fmax-fmin)/fres))

def baluev_false_alarm_probability(power_level: float, time: np.ndarray, flux: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    ls = LombScargle(time, flux, normalization='standard')
    return ls.false_alarm_probability(power_level, method='baluev', minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=10)

def find_top_k_peaks(x: np.ndarray, k: int) -> np.ndarray:
    local_maxima = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
    idx = np.argsort(x[1:-1][local_maxima])[::-1][:k]
    return np.arange(1, len(x)-1)[local_maxima][idx]

def periodogram_features(time: np.ndarray, flux: np.ndarray, oversampling_factor: float = 10., peaks_to_finetune: int = 0) -> dict[str, float]:
    delta_time = np.median(time[1:] - time[:-1])
    time_span = time[-1] - time[0]
    fmin, fmax, fres = 1/time_span, 0.5/delta_time, 1/(oversampling_factor * time_span)
    freqs = frequency_grid(fmin, fmax, fres)
    ampl = lombscargle_periodogram(time, flux, fmin, fmax, fres)
    if peaks_to_finetune>0:
        peak_idx = find_top_k_peaks(ampl, k=peaks_to_finetune)
        fres_ft = fres / 10.0  # 10 times finer than coarse grid
        for idx in peak_idx:
            f0 = freqs[idx]
            f_lo = max(fmin, f0 - fres)
            f_hi = min(fmax, f0 + fres)
            # Compute refined local periodogram
            ampl_ref = lombscargle_periodogram(time, flux, f_lo, f_hi, fres_ft)
            freqs_ref = frequency_grid(f_lo, f_hi, fres_ft)
            jmax = int(np.argmax(ampl_ref))
            amax = float(ampl_ref[jmax])
            fmax_local = float(freqs_ref[jmax])
            ampl[idx] = amax
            freqs[idx] = fmax_local
    fundamental_idx = np.argmax(ampl)
    fundamental_frequency = np.log10(freqs[fundamental_idx])
    fundamental_fap = baluev_false_alarm_probability(ampl[fundamental_idx], time, flux, fmin, fmax)
    eps = np.finfo(float).tiny
    fap_clipped = eps if fundamental_fap < eps else fundamental_fap 
    return {'ls_fundamental_log_10_frequency': fundamental_frequency.item(), 'ls_fundamental_log_10_fap': np.log10(fap_clipped).item()}

def fourier_series_design_matrix(time: np.ndarray, fundamental_frequency: float, num_components: int) -> np.ndarray:
    omegat = 2 * np.pi * fundamental_frequency * time
    cos_terms = [np.cos(omegat*j) for j in range(1, num_components+1)]
    sin_terms = [np.sin(omegat*j) for j in range(1, num_components+1)]
    return np.concatenate([[np.ones_like(time)] + cos_terms + sin_terms]).T

def fit_fourier_series(time: np.ndarray, flux: np.ndarray, fundamental_frequency: float, num_components: int):
    X = fourier_series_design_matrix(time, fundamental_frequency=fundamental_frequency, num_components=num_components)
    theta, sum_squared_residuals, rank, s = np.linalg.lstsq(X, flux, rcond=None)
    theta_cos, theta_sin = np.array_split(theta[1:], 2)
    # Representation for a model sum_j Aj cos( j omega time - PHj)
    theta_cos, theta_sin = np.array_split(theta[1:], 2)
    amplitudes = np.sqrt(theta_cos ** 2 + theta_sin ** 2)
    phases = np.arctan2(theta_sin, theta_cos)
    return theta, amplitudes, phases, sum_squared_residuals

def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2*np.pi) - np.pi

def fourier_series_features(time, flux, fundamental_frequency: float, num_components: int = 4, adjust_with_harmonic_ratio: bool = False) -> dict[str, float]:
    _, amplitudes, phases, sum_squared_residuals = fit_fourier_series(time, flux, fundamental_frequency, num_components)
    # First phase will be reference
    if adjust_with_harmonic_ratio: # As in Debosscher et al. 2007
        relative_phases = wrap_to_pi(phases - np.arange(1, len(phases)+1)*phases[0])[1:]
    else: # As in Richards et al. 2012
        relative_phases = wrap_to_pi(phases - phases[0])[1:]
    output_amplitudes = {f'fourier_amplitude_{j+1}': a.item() for j, a in enumerate(amplitudes)}
    output_phases = {f'fourier_relative_phase_{j+2}': p.item() for j, p in enumerate(relative_phases)}
    output_r2 = {'fourier_r2_fit': 1. - sum_squared_residuals[0].item()/np.sum((flux - np.mean(flux))**2)}
    return {**output_amplitudes, **output_phases, **output_r2}

