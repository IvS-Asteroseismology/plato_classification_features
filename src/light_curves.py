from pathlib import Path
import fitsio
import numpy as np


def read_fits_lc(path: Path | str, hdul: int = 1) -> tuple[np.ndarray, np.ndarray]:
    with fitsio.FITS(path) as f:
        time = f[hdul].read_column("time").astype('float64')
        flux = f[hdul].read_column("psf_flux").astype('float64')
    return time, flux


def preprocess_lc(time: np.ndarray, flux: np.ndarray):
    raise NotImplementedError()
