from dataclasses import dataclass
import numpy as np
from matplotlib import pyplot as plt

from periodogram_features import periodogram_features, compute_periodogram, baluev_fap_level, SUBHARMONIC_LABELS


def fourier_design_matrix(
    time: np.ndarray, frequency: float, num_components: int,
) -> np.ndarray:
    omegat = 2 * np.pi * frequency * time
    k = np.arange(1, num_components + 1)
    return np.column_stack(
        [np.ones_like(time)]
        + [np.cos(omegat * j) for j in k]
        + [np.sin(omegat * j) for j in k]
    )

def fit_fourier_series(
    time: np.ndarray, 
    flux: np.ndarray, 
    frequency: float, 
    num_components: int,
    l2_reg: float = 0.0,
) -> tuple[np.ndarray, float]:
    assert num_components>0, "At least one sinusoidal  component needed"
    X = fourier_design_matrix(time, frequency, num_components)
    if l2_reg > 0: # ridge regression with lstsq
        n_cols = X.shape[1]
        reg_mat = np.sqrt(l2_reg) * np.eye(n_cols)
        reg_mat[0, 0] = 0.0  # don't regularize the mean
        X = np.vstack([X, reg_mat])
        flux = np.concatenate([flux, np.zeros(n_cols)])
    theta, *_ = np.linalg.lstsq(X, flux, rcond=None)
    residuals = flux[:len(time)] - X[:len(time)] @ theta
    sum_squared_residuals = residuals @ residuals
    return theta, sum_squared_residuals

def resolve_fundamental(periodogram_results: dict[str, float], ratio_threshold: float) -> float:
    dominant = periodogram_results["ls_frequency_dominant"]
    for divisor, label in SUBHARMONIC_LABELS.items():
        subharmonic_amplitude = periodogram_results[f"ls_amplitude_ratio_{label}_dominant"]
        subharmonic_is_significant = periodogram_results[f"ls_amplitude_is_significant_{label}_dominant"]
        subharmonic_is_local_max = periodogram_results[f"ls_is_local_max_{label}_dominant"]
        if subharmonic_amplitude > ratio_threshold and subharmonic_is_significant and subharmonic_is_local_max:
            return dominant / divisor
    return dominant

@dataclass
class PrewhiteningComponent:
    frequency: float
    theta: np.ndarray
    variance_explained: float

def prewhiten(
    time: np.ndarray, flux: np.ndarray, 
    num_components: int, num_iterations: int, ratio_threshold: float, 
    oversampling_factor: float, finetune_factor: float, peak_density_maximum_fap: float,
    l2_reg: float = 0.0,
) -> tuple[list[PrewhiteningComponent], list[dict[str, float]]]:    
    total_variance = np.var(flux)
    residual = flux.copy()    
    components, pfs = [], []
    for _ in range(num_iterations):
        pf = periodogram_features(
                time, residual, 
                oversampling_factor=oversampling_factor,
                finetune_factor=finetune_factor, 
                peak_density_maximum_fap=peak_density_maximum_fap,
        )
        pfs.append(pf)
        freq = resolve_fundamental(pf, ratio_threshold) 
        theta, _ = fit_fourier_series(time, residual, freq, num_components, l2_reg)
        variance_before = np.var(residual)
        residual = residual - np.dot(fourier_design_matrix(time, freq, num_components), theta)
        variance_after = np.var(residual)
        components.append(PrewhiteningComponent(
            frequency=freq,
            theta=theta,
            variance_explained=((variance_before - variance_after) / total_variance).item(),
        ))
    return components, pfs

def cartesian_to_polar(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Representation for a model A0 + sum_j Aj cos( j omega time - PHj)
    theta_cos, theta_sin = np.array_split(theta[1:], 2)
    amplitudes = np.sqrt(theta_cos ** 2 + theta_sin ** 2)
    phases = np.arctan2(theta_sin, theta_cos)
    return amplitudes, phases

def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi

def prewhitening_features(
    components: list[PrewhiteningComponent],
) -> dict[str, float]:
    feats = {}
    n = len(components)
    variances = [c.variance_explained for c in components]
    total = sum(variances)
    feats["pw_total_variance_explained"] = total
    feats["pw_variance_ratio_1_to_total"] = variances[0] / total
    feats["pw_variance_ratio_2_1"] = variances[1] / variances[0] if variances[0] > 0 else np.nan
    for i, c in enumerate(components):
        feats[f"pw_variance_explained_{i + 1}"] = c.variance_explained
        feats[f"pw_frequency_{i + 1}"] = c.frequency
        amplitudes, phases = cartesian_to_polar(c.theta)
        feats[f"pw_{i + 1}_amplitude_1"] = amplitudes[0].item()
        sum_ratio_sq = 0.0
        for j in range(1, len(amplitudes)):
            ratio = (amplitudes[j] / amplitudes[0]).item()
            feats[f"pw_{i + 1}_amplitude_ratio_{j + 1}_1"] = ratio
            rel_phase = wrap_to_pi(phases[j] - (j + 1) * phases[0])
            feats[f"pw_{i + 1}_relative_phase_{j + 1}"] = rel_phase.item()
            sum_ratio_sq += ratio ** 2
        feats[f"pw_{i + 1}_harmonicity"] = 1.0 / (1.0 + sum_ratio_sq)
    for i in range(1, n):
        feats[f"pw_freq_ratio_{i + 1}_1"] = components[i].frequency / components[0].frequency
        feats[f"pw_amplitude_ratio_comp_{i + 1}_1"] = (
            cartesian_to_polar(components[i].theta)[0][0]
            / cartesian_to_polar(components[0].theta)[0][0]
        ).item()
    return feats

def plot_prewhitening(
    time: np.ndarray,
    flux: np.ndarray,
    components: list[PrewhiteningComponent],
    n_components: int,
    freq_xlim: float = 5.0,
    oversampling_factor: float = 5.0,
    normalization='standard',
    fap_level: float = 0.01,
) -> plt.Figure:
    n_iter = len(components)
    fig, axes = plt.subplots(n_iter + 1, 2, figsize=(14, 3 * (n_iter + 1)), constrained_layout=True)
    residual = flux.copy()
    phase_time = np.linspace(0, 1, 500)

    for i, comp in enumerate(components):
        model = np.dot(fourier_design_matrix(time, comp.frequency, n_components), comp.theta)
        period = 1.0 / comp.frequency
        phase = (time * comp.frequency) % 1.0
        phase_model = np.dot(
            fourier_design_matrix(phase_time / comp.frequency, comp.frequency, n_components),
            comp.theta,
        )
        freqs, power, (fmin, fmax, fres) = compute_periodogram(time, residual, oversampling_factor, normalization=normalization)
        power_threshold = baluev_fap_level(fap_level, time, flux, fmin, fmax, oversampling_factor, normalization)
        ax_pg = axes[i, 0]
        mask = freqs < freq_xlim
        ax_pg.plot(freqs[mask], power[mask], lw=0.5)
        ax_pg.axhline(power_threshold, ls='--', c='r', lw=1, label=f"FAP={fap_level:.3f}")
        ax_pg.axvline(comp.frequency, ls="--", c="g", lw=1, label=f"f={comp.frequency:.4f}")
        #ax_pg.set_xlim(0, freq_xlim)
        ax_pg.set_ylabel("Power")
        ax_pg.set_title(f"Iteration {i + 1}: var explained = {comp.variance_explained:.2%}")
        ax_pg.legend(fontsize=8)
        ax_ph = axes[i, 1]
        ax_ph.scatter(phase, residual, s=1, alpha=0.3, c="grey")
        order = np.argsort(phase_time)
        ax_ph.plot(phase_time[order], phase_model[order], c="r", lw=1.5)
        ax_ph.set_ylabel("Flux")
        ax_ph.set_title(f"P = {period:.4f} d")
        residual = residual - model
    # final residual
    freqs, power, _ = compute_periodogram(time, residual, oversampling_factor, normalization=normalization)
    axes[-1, 0].plot(freqs, power, lw=0.5)
    axes[-1, 0].set_xlim(0, freq_xlim)
    axes[-1, 0].set_xlabel("Frequency [1/d]")
    axes[-1, 0].set_ylabel("Power")
    axes[-1, 0].set_title("Residual periodogram")

    axes[-1, 1].scatter(time, residual, s=1, alpha=0.3, c="grey")
    axes[-1, 1].set_xlabel("Time [d]")
    axes[-1, 1].set_ylabel("Flux")
    residual_std = np.std(residual)
    axes[-1, 1].set_title(f"Residual (std = {residual_std:.4f})")
    return fig
