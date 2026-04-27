from dataclasses import dataclass

from time_domain_features import max_slope, flux_statistics, autocorrelation_features
from barbara_kepler_features import barbara_features
from periodogram_features import periodogram_features
from fourier_series_features import fourier_series_features

@dataclass(frozen=True)
class FeatureExtractorConfig:
    oversampling_factor: float = 5.0
    n_harmonics: int = 4
    max_lag_days: float = 4.0
    return_relative_fourier_amplitudes = True

feature_types = float | str

def extract_features(time, flux, cfg: FeatureExtractorConfig) -> dict[str, feature_types]:
    # NOTE: This assumes that time and flux have already been sanitized
    # First order features: No dependencies
    fof = {}
    fof.update(flux_statistics(flux))
    fof.update(max_slope(time, flux))
    fof.update(autocorrelation_features(time, flux, cfg.max_lag_days))
    fof.update(barbara_features(time, flux))
    fof.update(periodogram_features(time, flux, oversampling_factor=cfg.oversampling_factor))
    # Second order features: Depend on first order results
    f0 = fof["ls_frequency_dominant"]
    sof = {}
    sof.update(fourier_series_features(
        time, flux, fundamental_frequency=f0, 
        num_components=cfg.n_harmonics, return_relative_amplitudes=cfg.return_relative_fourier_amplitudes,
    ))
    return {**fof, **sof}
