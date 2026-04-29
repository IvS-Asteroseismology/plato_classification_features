from dataclasses import dataclass
import numpy as np

from time_domain_features import max_slope, flux_statistics, autocorrelation_features
from barbara_kepler_features import barbara_features
from periodogram_features import periodogram_features
from prewhitening_features import prewhiten, prewhitening_features
from entropy_features import entropy_features

@dataclass(frozen=True)
class FeatureExtractorConfig:
    # Periodogram config
    oversampling_factor: float = 5.0
    finetune_factor: float = 10.0
    peak_max_fap_level: float = 0.01
    # Autocorrelation config
    max_lag_days: float = 4.0
    # Multi scale ntropy config
    m: int = 1
    tolerance_factor: float = 0.2
    # Prewhitening config
    num_harmonics: int = 10
    num_prewithening_iterations: int = 3
    half_frequency_detection_threshold: float = 0.025


def extract_features(
        time: np.ndarray, flux: np.ndarray, 
        sector: int, cfg: FeatureExtractorConfig
) -> dict[str, float]:
    # NOTE: This assumes that time and flux have already been sanitized
    feats = {}
    feats.update(flux_statistics(flux))
    feats.update(max_slope(time, flux))
    feats.update(autocorrelation_features(time, flux, cfg.max_lag_days))
    feats.update(barbara_features(time, flux))
    feats.update(entropy_features(time, flux, sector, cfg.m, cfg.tolerance_factor))
    prewhitening_result, periodogram_features = prewhiten(
        time, flux,
        num_components=cfg.num_harmonics, num_iterations=cfg.num_prewithening_iterations, ratio_threshold=cfg.half_frequency_detection_threshold,
        oversampling_factor=cfg.oversampling_factor, finetune_factor=cfg.finetune_factor, peak_density_maximum_fap=cfg.peak_max_fap_level,
    )
    feats.update(periodogram_features[0]) # The full periodogram features of the first component
    feats.update(prewhitening_features(prewhitening_result))
    return feats
