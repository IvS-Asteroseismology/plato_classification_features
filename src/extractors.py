from dataclasses import dataclass
from typing import Any, Protocol
from pathlib import Path
import numpy as np
import polars as pl

from time_domain_features import max_slope, flux_statistics, autocorrelation_features
from barbara_kepler_features import barbara_features
from frequency_domain_features import periodogram_features, fourier_series_features
from light_curves import read_fits_lc, preprocess_lc

class ExtractorFn(Protocol):
    def __call__(
        self,
        row: dict[str, Any],
    ) -> dict[str, Any]: ...

@dataclass(frozen=True)
class FeatureExtractorConfig:
    oversampling_factor: float = 5.0
    num_peaks_for_peak_structure: int = 3
    n_harmonics: int = 4
    max_lag_days: float = 4.0
    return_relative_fourier_amplitudes = True

feature_types = float | str

def extract_features(time, flux, cfg: FeatureExtractorConfig = FeatureExtractorConfig()) -> dict[str, feature_types]:
    # NOTE: This assumes that time and flux have already been sanitized
    # First order features: No dependencies
    fof = {}
    fof.update(flux_statistics(flux))
    fof.update(max_slope(time, flux))
    fof.update(autocorrelation_features(time, flux, cfg.max_lag_days))
    fof.update(barbara_features(time, flux))
    fof.update(periodogram_features(
        time, flux, 
        oversampling_factor=cfg.oversampling_factor, num_peaks_for_peak_structure=cfg.num_peaks_for_peak_structure
    ))
    # Second order features: Depend on first order results
    f0 = fof["ls_frequency_fundamental"]
    sof = {}
    sof.update(fourier_series_features(
        time, flux, fundamental_frequency=f0, 
        num_components=cfg.n_harmonics, return_relative_amplitudes=cfg.return_relative_fourier_amplitudes,
    ))
    return {**fof, **sof}

def extract_features_from_light_curve(time: np.ndarray, flux: np.ndarray, sector: int, path: Path | str) -> dict[str, Any]:
    try:
        time, flux = preprocess_lc(time, flux, sector)
    except Exception as e:
        return {"path": str(path), "_error": f"preprocessing error: {e}"}
    try:
        feats = extract_features(time, flux)
    except Exception as e:
        return {"path": str(path), "_error": f"feature error: {e}"}
    feats["path"] = str(path)
    feats["num_negative_fluxes"] = np.sum(flux<0.0).item()
    return feats

"""
def extract_features_from_path(path: Path | str) -> dict[str, feature_types]:
    if isinstance(path, str):
        path = Path(path)
    try:
        time, flux, flux_error, sector = read_fits_lc(path)
    except Exception as e:
        return {"path": str(path), "_error": f"parsing error: {e}"}
    return extract_features_from_light_curve(time, flux, sector, path)
"""

def extract_features_from_row(row: dict[str, Any], uncalibrated_flux: bool = False) -> dict[str, Any]:
    time = np.asarray(row['time'])
    if uncalibrated_flux:
        flux = np.asarray(row['raw_flux'])
    else:
        flux = np.asarray(row['cal_flux'])
    sector = row['sector']
    path = row['path']
    return extract_features_from_light_curve(time, flux, sector, path)

def extract_light_curve_from_path(path: Path | str, flux_type: str = "aper") -> dict[str, Any]:
    try:
        lc = read_fits_lc(path, flux_type=flux_type)
    except Exception as e:
        return {'path': str(path), '_error': f"parsing error {e}"}
    return {
        'path'      : str(path),
        'sector'    : lc.sector,
        'flux_type' : lc.flux_type,
        'time'      : lc.time,
        'raw_flux'  : lc.raw_flux,
        'cal_flux'  : lc.cal_flux,
        'raw_error' : lc.raw_error,
        'cal_error' : lc.cal_error,
    }

def extract_light_curve_from_row(row: dict[str, Any], flux_type: str) -> dict[str, Any]:
    return extract_light_curve_from_path(row['path'] if 'path' in row else row['FilePath'], flux_type=flux_type)
