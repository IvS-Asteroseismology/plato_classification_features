from dataclasses import dataclass
import numpy as np
import polars as pl

from time_domain_features import max_slope, flux_statistics
from frequency_domain_features import periodogram_features, fourier_series_features

@dataclass(frozen=True)
class FeatureExtractorConfig:
    oversampling_factor: float = 10.0
    peaks_to_finetune: int = 10
    n_harmonics: int = 4

def extract_features(time, flux, cfg: FeatureExtractorConfig = FeatureExtractorConfig()) -> dict[str, float]:
    # NOTE: This assumes that time and flux have already been sanitized
    # First order features: No dependencies
    td = {}
    td.update(flux_statistics(flux))
    td.update(max_slope(time, flux))
    # Stage 2: periodogram
    pg = periodogram_features(time, flux, oversampling_factor=cfg.oversampling_factor, peaks_to_finetune=cfg.peaks_to_finetune)
    # Second order features: Depend on first order results
    f0 = 10 ** pg["ls_fundamental_log_10_frequency"]
    fs = fourier_series_features(time, flux, fundamental_frequency=f0, num_components=cfg.n_harmonics)
    return {**td, **pg, **fs}


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
        
