from typing import Any, Protocol
from pathlib import Path
import numpy as np
import polars as pl

from features import extract_features
from light_curves import read_fits_lc, preprocess_lc
from config import PipelineConfig


class ExtractorFn(Protocol):
    def __call__(
        self,
        df_slice: pl.DataFrame,
    ) -> dict[str, Any]: ...

def extract_features_from_light_curve(time: np.ndarray, flux: np.ndarray, sector: int, cfg: PipelineConfig) -> dict[str, Any]:
    try:
        time, flux = preprocess_lc(time, flux, sector, cfg.preprocessing)
    except Exception as e:
        return {"_error": f"preprocessing error: {e}"}
    try:
        feats = extract_features(time, flux, sector, cfg.features)
    except Exception as e:
        return {"_error": f"feature error: {e}"}
    feats["lc_num_obs_after_preprocessing"] = len(time)
    feats["lc_num_negative_fluxes"] = np.sum(flux<0.0).item()
    return feats

def _flux_col(cfg: PipelineConfig) -> str:
    return "cal_flux" if cfg.preprocessing.use_calibrated_flux else "raw_flux"

def extract_features_from_slice(df_slice: pl.DataFrame, cfg: PipelineConfig) -> dict[str, Any]:
    time = df_slice['time'].to_numpy()[0]
    flux = df_slice[_flux_col(cfg)].to_numpy()[0]
    return {'path': df_slice['path'].item()} | extract_features_from_light_curve(time, flux, df_slice['sector'].item(), cfg)

def extract_features_from_path(
    path: Path | str, cfg: PipelineConfig, flux_type: str = "aper",
) -> dict[str, Any]:
    lc = extract_light_curve_from_path(path, flux_type)
    if "_error" in lc:
        return lc    
    feats = extract_features_from_light_curve(lc["time"], lc[_flux_col(cfg)], lc["sector"], cfg)
    return {'path': lc['path']} | feats

def extract_light_curve_from_slice(df_slice: pl.DataFrame, flux_type: str) -> dict[str, Any]:
    path = df_slice['path'].item()
    return extract_light_curve_from_path(path, flux_type=flux_type)

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
