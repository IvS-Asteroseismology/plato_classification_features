from pathlib import Path
import fitsio
import numpy as np
from scipy.ndimage import gaussian_filter1d


def read_fits_lc(path: Path | str, hdu: int = 1, flux_col: str = "cal_aper_flux", apply_masks: bool = True) -> tuple[np.ndarray, np.ndarray]:
    with fitsio.FITS(path) as f:
        data = f[hdu]
        time = data.read_column("time").astype('float64')
        flux = data.read_column(flux_col).astype('float64')
        if apply_masks:
            tess_flags = data.read_column('TESS_flags')
            tglc_flags = data.read_column('TGLC_flags')
            mask_good = np.logical_and(tess_flags == 0, tglc_flags == 0)
            time = time[mask_good]
            flux = flux[mask_good]
    time, flux = preprocess_light_curve(time, flux, sigma_threshold=10, detrend_sigma=100)
    return time, flux

def smooth_time_series(time_series: np.ndarray, sigma: float) -> np.ndarray:
    '''
    A helper function to do Gaussian smoothing with a filter.
    '''
    smoothed_series = gaussian_filter1d(time_series, sigma=sigma, mode='nearest')
    return smoothed_series


def apply_quality_mask(time: np.ndarray, flux: np.ndarray, quality_mask: np.ndarray, nan_threshold: int = 10) -> tuple[np.ndarray, np.ndarray]:
    '''
    A helper function to apply a quality mask, remove NaN values, and check for sufficient data.
    '''
    good_quality = quality_mask == 0
    time = time[good_quality]
    flux = flux[good_quality]
    
    mask = ~np.isnan(flux)
    if np.sum(mask) < nan_threshold:
        raise ValueError("A light curve contains too many NaN values.")
    return time[mask], flux[mask]

def remove_outliers(time: np.ndarray, flux: np.ndarray, sigma_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    '''
    A helper function to remove outliers from flux based on a sigma threshold.
    '''
    flux_mean = np.mean(flux)
    flux_std = np.std(flux)
    mask = np.abs(flux - flux_mean) < sigma_threshold * flux_std
    return time[mask], flux[mask]

def preprocess_light_curve(time, flux, sigma_threshold, detrend_sigma):
    '''
    A helper function to processes the light curve by detrending with a Gaussian filter.
    '''
    time, flux = remove_outliers(time, flux, sigma_threshold)
    #if time is None or flux is None:
    #    raise ValueError(f"The light curve {file} is empty or invalid.")
    smoothed = smooth_time_series(flux, sigma=detrend_sigma)
    detrended_flux = flux[:len(smoothed)] - smoothed
    detrended_time = time[:len(detrended_flux)]
    return detrended_time, detrended_flux + np.mean(flux)

"""
def get_data(file, pipeline, ap=False):
    '''
    Opens a FITS file and detrends a light curve.
    Returns detrended flux and corresponding time points.
    '''
    if not os.path.exists(file):
        raise FileNotFoundError(f"The file '{file}' does not exist.")
    
    if pipeline == 'QLP':
        lc = lk.io.qlp.read_qlp_lightcurve(file)
        if lc is None:
            raise ValueError(f"Failed to read the light curve data from the file '{file}'.")
        time, flux = apply_quality_mask(
            np.array(lc['time'].value),
            np.array(lc['sap_flux'].value),
            lc.quality
        )
        if time is None or flux is None:
            raise ValueError(f"The light curve data for pipeline '{pipeline}' is empty or invalid.")
        return process_light_curve(time, flux, sigma_threshold=10, detrend_sigma=61)

    elif pipeline == 'TGLC':
        with fits.open(file) as hdul:
            if 'TESS_flags' not in hdul[1].data.columns.names or 'TGLC_flags' not in hdul[1].data.columns.names:
                raise ValueError(f"The FITS file '{file}' does not contain required columns 'TESS_flags' or 'TGLC_flags'.")
            flags = np.logical_and(
                hdul[1].data['TESS_flags'] == 0,
                hdul[1].data['TGLC_flags'] == 0
            )
            time = np.array(hdul[1].data['time'][flags])
            if ap == True:
                flux = np.array(hdul[1].data['cal_aper_flux'][flags])
            else:
                flux = np.array(hdul[1].data['cal_psf_flux'][flags])
            
            time, flux = apply_quality_mask(time, flux, quality_mask=np.zeros(len(time)))
            if time is None or flux is None:
                raise ValueError(f"The light curve data for pipeline '{pipeline}' is empty or invalid.")
            return process_light_curve(time, flux, sigma_threshold=10, detrend_sigma=100)

    else:
        raise ValueError(
            f"Pipeline '{pipeline}' is not supported. Supported pipelines: 'QLP', 'TGLC'."
        )
"""
