import numpy as np


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

def fit_fourier_series(time: np.ndarray, flux: np.ndarray, frequency: float, num_components: int) -> tuple[np.ndarray, float]:
    assert num_components>0, "At least one sinusoidal  component needed"
    X = fourier_design_matrix(time, frequency, num_components)
    theta, sum_squared_residuals, _, _ = np.linalg.lstsq(X, flux, rcond=None)
    return theta, sum_squared_residuals

def cartesian_to_polar(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Representation for a model A0 + sum_j Aj cos( j omega time - PHj)
    theta_cos, theta_sin = np.array_split(theta[1:], 2)
    amplitudes = np.sqrt(theta_cos ** 2 + theta_sin ** 2)
    phases = np.arctan2(theta_sin, theta_cos)
    return amplitudes, phases

def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi

def fourier_series_features(
    time: np.ndarray, 
    flux: np.ndarray, 
    fundamental_frequency: float, 
    num_components: int, 
    adjust_with_harmonic_ratio: bool = True, 
    return_relative_amplitudes: bool = True,
) -> dict[str, float]:
    theta, sum_squared_residuals = fit_fourier_series(time, flux, fundamental_frequency, num_components)
    amplitudes, phases = cartesian_to_polar(theta)    
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
