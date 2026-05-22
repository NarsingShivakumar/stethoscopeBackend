"""utils/dsp.py — reusable DSP primitives used across all services."""
from __future__ import annotations
import numpy as np
import scipy.signal as sps


def bandpass(
    x: np.ndarray, sr: int,
    low: float, high: float,
    order: int = 4
) -> np.ndarray:
    """Butterworth bandpass filter (zero-phase sosfiltfilt)."""
    nyq  = sr / 2.0
    low  = max(low, 1.0)
    high = min(high, nyq * 0.99)
    if low >= high:
        return x.copy()
    sos = sps.butter(order, [low / nyq, high / nyq],
                     btype="bandpass", output="sos")
    return sps.sosfiltfilt(sos, x.astype(np.float64)).astype(np.float32)


def hilbert_envelope(
    x: np.ndarray,
    smooth_hz: float,
    sr: int
) -> np.ndarray:
    """Compute analytic envelope via Hilbert transform, then lowpass-smooth."""
    env  = np.abs(sps.hilbert(x.astype(np.float64))).astype(np.float32)
    nyq  = sr / 2.0
    freq = min(smooth_hz, nyq * 0.95)
    sos  = sps.butter(4, freq / nyq, btype="low", output="sos")
    return sps.sosfiltfilt(sos, env.astype(np.float64)).astype(np.float32)


def rms(x: np.ndarray) -> float:
    """Root-mean-square energy."""
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def find_peaks_physiological(
    env: np.ndarray, sr: int,
    min_bpm: float, max_bpm: float,
    min_height_factor: float,
    min_width_ms: float, max_width_ms: float
) -> np.ndarray:
    """
    Peak detection constrained to physiological heart rate range.
    Returns array of peak sample indices.
    """
    min_dist = int(60.0 / max_bpm * sr)
    min_w    = int(min_width_ms * sr / 1000.0)
    max_w    = int(max_width_ms * sr / 1000.0)
    med      = float(np.median(env)) + 1e-9
    try:
        peaks, _ = sps.find_peaks(
            env.astype(np.float64),
            height     = med * min_height_factor,
            distance   = max(1, min_dist),
            width      = (max(1, min_w), max_w),
            prominence = med * 0.8,
        )
    except Exception:
        return np.array([], dtype=int)
    return peaks