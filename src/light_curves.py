from pathlib import Path
import fitsio
import numpy as np
from scipy.ndimage import gaussian_filter1d
from astropy.stats import sigma_clip


def read_fits_lc(
        path: Path | str, 
        hdu_index: int = 1, 
        flux_column: str = "cal_aper_flux", 
        apply_quality_masks: bool = True, 
        
) -> tuple[np.ndarray, np.ndarray, str | None]:
    with fitsio.FITS(path) as f:
        header = f[0].read_header()
        sector = header['SECTOR'] if 'SECTOR' in header else None
        hdu_data = f[hdu_index]
        time = hdu_data.read_column("time").astype('float64')
        flux = hdu_data.read_column(flux_column).astype('float64')
        if apply_quality_masks:
            tess_flags = hdu_data.read_column('TESS_flags')
            tglc_flags = hdu_data.read_column('TGLC_flags')
    if apply_quality_masks:
        good_quality_mask = np.logical_and(tess_flags == 0, tglc_flags == 0)
        time = time[good_quality_mask]
        flux = flux[good_quality_mask]
    return time, flux, sector

def preprocess_lc(
    time: np.ndarray,
    flux: np.ndarray,
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
