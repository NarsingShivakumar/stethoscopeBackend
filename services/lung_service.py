"""
services/lung_service.py
=========================
Lung sound processing: keep full waveform + optional classification
into normal / wheeze / crackles.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import scipy.signal as sps

log = logging.getLogger("steth.lung")


class LungService:
    """Process separated lung audio and classify lung sounds."""

    def analyze(self, lung_audio: np.ndarray, sr: int) -> Dict:
        """
        Parameters
        ----------
        lung_audio : float32 (N,) separated lung channel

        Returns
        -------
        {
            "lung_classification": "normal" | "wheeze" | "crackles" | "mixed",
            "wheeze_confidence":   float,
            "crackle_confidence":  float,
        }
        """
        wheeze_conf  = self._detect_wheeze(lung_audio, sr)
        crackle_conf = self._detect_crackles(lung_audio, sr)

        if wheeze_conf > 0.5 and crackle_conf > 0.5:
            classification = "mixed"
        elif wheeze_conf > 0.5:
            classification = "wheeze"
        elif crackle_conf > 0.5:
            classification = "crackles"
        else:
            classification = "normal"

        log.info(
            "Lung: %s  wheeze=%.3f  crackles=%.3f",
            classification, wheeze_conf, crackle_conf
        )

        return {
            "lung_classification": classification,
            "wheeze_confidence":   round(wheeze_conf,  3),
            "crackle_confidence":  round(crackle_conf, 3),
        }

    def _detect_wheeze(self, audio: np.ndarray, sr: int) -> float:
        """
        Wheeze: continuous, high-pitched (400–1000 Hz), tonal sound.
        High spectral flatness in the wheeze band indicates tonality.
        """
        band = _bandpass(audio, sr, 400, 1000, order=4)
        n    = len(band)
        if n < 256:
            return 0.0
        X    = np.abs(np.fft.rfft(band * np.hanning(n))) + 1e-12
        # Wiener entropy (inverse of flatness → high for tonal)
        log_mean   = float(np.exp(np.mean(np.log(X))))
        arith_mean = float(np.mean(X))
        tonality   = log_mean / (arith_mean + 1e-9)
        # Also check duration: wheeze is sustained
        env    = _envelope(band, sr, smooth_hz=20.0)
        active = float((env > env.max() * 0.2).mean())
        score  = float(np.clip(tonality * 3.0 * active * 2.0, 0, 1))
        return score

    def _detect_crackles(self, audio: np.ndarray, sr: int) -> float:
        """
        Crackles: brief, discontinuous, high-frequency (200–2000 Hz) sounds.
        Detected as high-energy transient spikes.
        """
        band  = _bandpass(audio, sr, 200, 2000, order=4)
        env   = _envelope(band, sr, smooth_hz=50.0)
        if len(env) < 10:
            return 0.0
        # Find transient spikes
        threshold = float(np.percentile(env, 85)) * 1.5
        min_dist  = int(0.01 * sr)    # 10ms minimum between crackles
        peaks, _  = sps.find_peaks(env, height=threshold, distance=min_dist)
        # Score by density and amplitude ratio
        density   = len(peaks) / (len(env) / sr)   # crackles per second
        score     = float(np.clip(density / 20.0, 0, 1))
        return score


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bandpass(x: np.ndarray, sr: int, low: float, high: float,
              order: int = 4) -> np.ndarray:
    nyq  = sr / 2.0
    low  = max(low, 1.0) / nyq
    high = min(high, nyq * 0.98) / nyq
    if low >= high:
        return x.copy()
    try:
        sos = sps.butter(order, [low, high], btype="bandpass", output="sos")
        return sps.sosfiltfilt(sos, x.astype(np.float64)).astype(np.float32)
    except Exception:
        return x.copy()


def _envelope(x: np.ndarray, sr: int, smooth_hz: float = 30.0) -> np.ndarray:
    from scipy.signal import hilbert
    env = np.abs(hilbert(x.astype(np.float64))).astype(np.float32)
    nyq = sr / 2.0
    freq = min(smooth_hz, nyq * 0.95)
    try:
        sos = sps.butter(4, freq / nyq, btype="low", output="sos")
        env = sps.sosfiltfilt(sos, env.astype(np.float64)).astype(np.float32)
    except Exception:
        pass
    return env