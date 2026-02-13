from dataclasses import dataclass
from typing import Iterator, Any
from pathlib import Path
import numpy as np
import polars as pl
from joblib import Parallel, delayed

from time_domain_features import max_slope, flux_statistics, autocorrelation_features
from barbara_kepler_features import barbara_features
from frequency_domain_features import periodogram_features, fourier_series_features
from light_curves import read_fits_lc

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

def extract_features_from_path(path: Path | str) -> dict[str, feature_types]:
    if isinstance(path, str):
        path = Path(path)
    try:
        t, f = read_fits_lc(path)
    except Exception as e:
        return {"path": str(path), "_error": f"parsing error: {e}"}
    try:
        feats = extract_features(t, f)
    except Exception as e:
        return {"path": str(path), "_error": f"feature error: {e}"}
    feats["path"] = str(path)
    return feats

def iter_batches(seq: list[Any], batch_size: int) -> Iterator[list[Any]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]

def extract_features_parallel(
    paths: list[str],
    *,
    n_jobs: int,
    backend: str,
    worker_batch_size: int,
) -> list[dict[str, feature_types]]:
    def _extract_many(ps: list[str]) -> list[dict[str, feature_types]]:
        return [extract_features_from_path(p) for p in ps]

    lists = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_extract_many)(sub) for sub in iter_batches(paths, worker_batch_size)
    )
    return [row for sub in lists for row in sub]


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
        
