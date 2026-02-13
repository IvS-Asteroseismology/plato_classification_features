import numpy as np

def SP_Summaries_welch_rect_peakPower_5(y):
    L = int(min(256, y.size))
    if L < 2:
        return np.nan
    window = np.ones(L, dtype=float)
    # MATLAB default when noverlap = []:
    noverlap = L // 2
    step = L - noverlap
    if step <= 0:
        return np.nan
    if y.size < L:
        segs = np.zeros((1, L), dtype=float)
        segs[0, :y.size] = y
    else:
        nseg = 1 + (y.size - L) // step
        idx = np.arange(L)[None, :] + step * np.arange(nseg)[:, None]
        segs = y[idx]
    segs = segs * window[None, :]
    X = np.fft.rfft(segs, n=L, axis=1)
    S = (np.abs(X) ** 2).mean(axis=0)  # average periodograms
    k = int(np.argmax(S))
    lo = max(0, k - 2)
    hi = min(S.size, k + 3)
    return float(np.mean(S[lo:hi]))


def ST_LocalExtrema_n100_medianmax(x: np.ndarray, n_windows: int = 100) -> float:
    N = x.size
    if N < 2 or n_windows < 1:
        return np.nan
    window_len = N // n_windows
    if window_len <= 1:
        return np.nan
    num = N // window_len
    x_trunc = x[:num * window_len]
    windows = x_trunc.reshape(num, window_len)
    loc_max = windows.max(axis=1)
    return np.median(loc_max).item()

def barbara_features(
    time: np.ndarray,
    flux: np.ndarray,
) -> dict[str, float]:
    output = {}
    x = (flux - np.mean(flux))/np.std(flux)
    output['AC_nl_001'] = np.corrcoef(x[:-1] ** 3, x[1:])[0, 1].item()
    output['CO_trev_3_num'] = np.mean((x[3:] - x[:-3])**3).item()
    output['ST_LocalExtrema_n100_medianmax'] = ST_LocalExtrema_n100_medianmax(x)
    output['SP_Summaries_welch_rect_peakPower_5'] = SP_Summaries_welch_rect_peakPower_5(flux - np.mean(flux))
    # PENDING: MF_steps_ahead_ar_best_6_mabserr_5 
    return output