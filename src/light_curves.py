from pathlib import Path
from dataclasses import dataclass
import fitsio
import numpy as np
from scipy.ndimage import gaussian_filter1d
from astropy.stats import sigma_clip


FLUX_COLUMNS = {
    "aper": ("aperture_flux", "cal_aper_flux"),
    "psf" : ("psf_flux",      "cal_psf_flux"),
}

ERROR_KEYS = {
        'cal_aper_flux': 'CAPE_ERR',
        'aperture_flux': 'APER_ERR',
        'cal_psf_flux' : 'CPSF_ERR',
        'psf_flux'     : 'PSF_ERR',
}

@dataclass
class LightCurve:
    time        : np.ndarray
    raw_flux    : np.ndarray
    cal_flux    : np.ndarray
    raw_error   : float
    cal_error   : float
    sector      : int
    flux_type   : str # "aper" or "psf"

def read_fits_lc(
        path: Path | str, 
        flux_type: str  = "aper",   # "aper" | "psf"
        apply_quality_masks: bool = True, 
        
) -> LightCurve:
    if flux_type not in FLUX_COLUMNS:
        raise ValueError(f"flux_type must be one of {list(FLUX_COLUMNS)}; got {flux_type!r}")
    raw_col, cal_col = FLUX_COLUMNS[flux_type]
    with fitsio.FITS(path) as f:
        metadata = f[0].read_header()
        sector = int(metadata['SECTOR'])
        hdu_data = f[1]
        header = hdu_data.read_header()
        def _parse_error(key):
            val = header[ERROR_KEYS[key]]
            return np.nan if val == "NaN" else float(val)
        raw_error = _parse_error(raw_col)
        cal_error = _parse_error(cal_col)
        time = hdu_data.read_column("time").astype('float64')
        raw_flux = hdu_data.read_column(raw_col).astype('float64')
        cal_flux = hdu_data.read_column(cal_col).astype('float64')
        if apply_quality_masks:
            tess_flags = hdu_data.read_column('TESS_flags')
            tglc_flags = hdu_data.read_column('TGLC_flags')
    if apply_quality_masks:
        good_quality_mask = np.logical_and(tess_flags == 0, tglc_flags == 0)
        time = time[good_quality_mask]
        raw_flux = raw_flux[good_quality_mask]
        cal_flux = cal_flux[good_quality_mask]
        
    return LightCurve(
        time      = time,
        raw_flux  = raw_flux,
        cal_flux  = cal_flux,
        raw_error = raw_error,
        cal_error = cal_error,
        sector    = sector,
        flux_type = flux_type,
    )

def preprocess_lc(
    time: np.ndarray,
    flux: np.ndarray,
    sector: int,
    min_non_nan_points: int = 10,
    detrend_sigma: float = 100
) -> tuple[np.ndarray, np.ndarray]:
    good_mask = ~np.isnan(flux) & ~np.isinf(flux)
    if np.sum(good_mask) < min_non_nan_points:
        raise ValueError("A light curve contains too many NaN values.")
    time = time[good_mask]
    flux = flux[good_mask]
    if sector is not None and sector >= 27 and len(flux) >= 3:
       n = (len(flux) // 3) * 3
       time = time[:n].reshape(-1, 3).mean(axis=1)
       flux = flux[:n].reshape(-1, 3).mean(axis=1)
    outlier_mask = sigma_clip(flux, sigma_lower=10.0, sigma_upper=5.0).mask
    time, flux = time[~outlier_mask], flux[~outlier_mask]
    if len(flux) == 0:
        raise ValueError("Light curve is empty after removing outliers.")
    time, flux = detrend_light_curve(time, flux, detrend_sigma=detrend_sigma, mode='reflect')
    return time, flux

def smooth_time_series(time_series: np.ndarray, sigma: float, mode: str = 'nearest') -> np.ndarray:
    '''
    A helper function to do Gaussian smoothing with a filter.
    '''
    smoothed_series = gaussian_filter1d(time_series, sigma=sigma, mode=mode)
    return smoothed_series

def detrend_light_curve(time: np.ndarray, flux: np.ndarray, detrend_sigma, mode: str = 'nearest') -> tuple[np.ndarray, np.ndarray]:
    '''
    A helper function to processes the light curve by detrending with a Gaussian filter.
    '''    
    smoothed = smooth_time_series(flux, sigma=detrend_sigma, mode=mode)
    detrended_flux = flux[:len(smoothed)] - smoothed
    detrended_time = time[:len(detrended_flux)]
    return detrended_time, detrended_flux + np.mean(flux)
