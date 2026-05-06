from dataclasses import dataclass
from functools import partial
import numpy as np

from time_domain_features import flux_statistics, time_flux_statistics
from autocorrelation_features import autocorrelation_features
from barbara_kepler_features import barbara_features
from periodogram_features import periodogram_features
from prewhitening_features import prewhiten, prewhitening_features
from entropy_features import entropy_features


@dataclass(frozen=True)
class FeatureExtractorConfig:
    # Periodogram config
    oversampling_factor: float = 5.0
    finetune_factor: float = 10.0
    peak_significance: float = 0.01
    peak_snr_threshold: float = 5
    # Autocorrelation config
    max_lag_days: float = 27.4/5
    # Multi scale ntropy config
    m: int = 1
    tolerance_factor: float = 0.2
    # Prewhitening config
    num_harmonics: int = 10
    num_prewithening_iterations: int = 3
    half_frequency_detection_threshold: float = 0.025
    l2_regularization: float = 1e-3

def extract_features(
        time: np.ndarray, flux: np.ndarray, 
        sector: int, cfg: FeatureExtractorConfig
) -> dict[str, float]:
    # NOTE: This assumes that time and flux have already been sanitized
    feats = {}
    feats.update(flux_statistics(flux))
    feats.update(time_flux_statistics(time, flux))
    feats.update(autocorrelation_features(time, flux, cfg.max_lag_days))
    # feats.update(barbara_features(time, flux))
    feats.update(entropy_features(time, flux, sector, cfg.m, cfg.tolerance_factor))
    # Configure the feature extractor and pass it to the prewhitening routine
    periodogram_feature_extractor = partial(
        periodogram_features, 
        oversampling_factor=cfg.oversampling_factor, finetune_factor=cfg.finetune_factor, 
        peak_significance=cfg.peak_significance, peak_snr_threshold=cfg.peak_snr_threshold,
    )
    prewhitening_result, per_features = prewhiten(
        time, flux, periodogram_feature_extractor,
        num_components=cfg.num_harmonics, num_iterations=cfg.num_prewithening_iterations, 
        ratio_threshold=cfg.half_frequency_detection_threshold, l2_reg=cfg.l2_regularization,
    )
    feats.update(per_features[0]) # The full periodogram features of the first component
    feats.update(prewhitening_features(prewhitening_result))
    return feats
