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
    peaks_to_finetune: int = 3
    n_harmonics: int = 4
    max_lag_days: float = 4.0

feature_types = float | str

def extract_features(time, flux, cfg: FeatureExtractorConfig = FeatureExtractorConfig()) -> dict[str, feature_types]:
    # NOTE: This assumes that time and flux have already been sanitized
    # First order features: No dependencies
    fof = {}
    fof.update(flux_statistics(flux))
    fof.update(max_slope(time, flux))
    fof.update(autocorrelation_features(time, flux, cfg.max_lag_days))
    fof.update(barbara_features(time, flux))
    fof.update(periodogram_features(time, flux, oversampling_factor=cfg.oversampling_factor, peaks_to_finetune=cfg.peaks_to_finetune))
    # Second order features: Depend on first order results
    f0 = fof["ls_fundamental_frequency"]
    sof = {}
    sof.update(fourier_series_features(time, flux, fundamental_frequency=f0, num_components=cfg.n_harmonics))
    return {**fof, **sof}

def extract_features_from_light_curve(time: np.ndarray, flux: np.ndarray, sector: int) -> dict[str, Any]:
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

def extract_features_from_path(path: Path | str) -> dict[str, feature_types]:
    if isinstance(path, str):
        path = Path(path)
    try:
        time, flux, sector = read_fits_lc(path)
    except Exception as e:
        return {"path": str(path), "_error": f"parsing error: {e}"}
    return extract_features_from_light_curve(time, flux, sector)

def extract_features_from_row(row: dict[str, Any]) -> dict[str, Any]:
    time, flux, sector = row['time'], row['flux'], row['sector']
    return extract_features_from_light_curve(time, flux, sector)

def extract_light_curve_from_path(path: Path | str) -> dict[str, Any]:
    try: 
        time, flux, sector = read_fits_lc(path)
    except Exception as e:
        return {'path': str(path), "_error": f"parsing error {e}"}
    return {'path': str(path), 'sector': sector, 'time': time, 'flux': flux}

def extract_light_curve_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return extract_light_curve_from_path(row['path'] if 'path' in row else row['FilePath'])


if __name__ == '__main__':

    data = pl.read_csv('data/random5.csv').with_columns(
        pl.col("time").str.json_decode(dtype=pl.List(pl.Float64)),
        pl.col("flux").str.json_decode(dtype=pl.List(pl.Float64)),
    )
    def get_lc(row_idx: int = 0) -> tuple[str, np.ndarray, np.ndarray]:
        row = data.slice(row_idx, 1)
        path = row['path'][0]
        time = row['time'][0].to_numpy()
        flux = row['flux'][0].to_numpy()
        return path, time, flux
    
    features = []
    for k in range(data.height):
        path, time, flux = get_lc(k)
        features.append({'path': path} | extract_features(time, flux))
    pl.DataFrame(features).write_parquet('features.parquet')
        
