"""
services/classifier.py
=======================
Stage 2 — Biomedical audio classification.

Classifies cleaned audio as:
  - heart      : dominant energy in 20-180 Hz, cardiac periodicity present
  - lungs      : broadband 100-2000 Hz, respiratory rhythm
  - mixed      : both heart and lung components
  - invalid    : very low energy, non-biomedical, artefact

Algorithm: Multi-band energy + autocorrelation periodicity scoring.
  Heart score  = (20-180 Hz energy ratio) × cardiac_periodicity
  Lung score   = (100-2000 Hz broadband ratio) × respiratory_breadth
  Mixed        = both scores > 0.3
  Invalid      = total RMS < -60 dBFS or non-physiological
"""
from __future__ import annotations
import logging
from typing import Dict

import numpy as np
import scipy.signal as sps

from utils.audio_io import resample, TARGET_SR_ANALYSIS
from utils.dsp import bandpass, hilbert_envelope, rms

log = logging.getLogger("steth.classifier")

SR = TARGET_SR_ANALYSIS

# Energy thresholds
MIN_RMS_DBFS    = -72.0   # below this → invalid
HEART_SCORE_THR = 0.22
LUNG_SCORE_THR  = 0.18

# Cardiac autocorrelation: 30–200 BPM
CARDIAC_LAG_MIN  = 60.0 / 200.0   # s
CARDIAC_LAG_MAX  = 60.0 / 30.0    # s

# Respiratory: 8–30 breaths/min
RESP_LAG_MIN  = 60.0 / 30.0   # 2 s
RESP_LAG_MAX  = 60.0 / 8.0    # 7.5 s


class ClassifierService:

    def classify(self, audio: np.ndarray, src_sr: int) -> Dict:
        """
        Returns {
          "heart": bool,
          "lungs": bool,
          "label": "heart"|"lungs"|"mixed"|"invalid",
          "heart_score": float,
          "lung_score":  float,
          "confidence":  float,
        }
        """
        # 1. RMS gate
        db = 20.0 * np.log10(rms(audio) + 1e-12)
        if db < MIN_RMS_DBFS:
            return self._label(False, False, 0.0, 0.0, reason="silent")

        audio4k = resample(audio, src_sr, SR)

        # 2. Band energy ratios
        heart_filt = bandpass(audio4k, SR, 20.0,  180.0)
        lung_filt  = bandpass(audio4k, SR, 100.0, 2000.0)
        total_rms  = rms(audio4k) + 1e-12
        heart_e    = rms(heart_filt) / total_rms
        lung_e     = rms(lung_filt)  / total_rms

        # 3. Periodicity via envelope autocorrelation
        heart_env  = hilbert_envelope(heart_filt, smooth_hz=10.0, sr=SR)
        lung_env   = hilbert_envelope(lung_filt,  smooth_hz=3.0,  sr=SR)

        card_period = self._autocorr_period(heart_env, SR, CARDIAC_LAG_MIN, CARDIAC_LAG_MAX)
        resp_period = self._autocorr_period(lung_env,  SR, RESP_LAG_MIN,    RESP_LAG_MAX)

        # 4. Composite scores
        heart_score = float(np.clip(0.60 * heart_e + 0.40 * card_period, 0, 1))
        lung_score  = float(np.clip(0.50 * lung_e  + 0.50 * resp_period, 0, 1))

        log.debug("classify  heart_score=%.3f  lung_score=%.3f", heart_score, lung_score)

        has_heart = heart_score >= HEART_SCORE_THR
        has_lung  = lung_score  >= LUNG_SCORE_THR

        return self._label(has_heart, has_lung, heart_score, lung_score)

    @staticmethod
    def _autocorr_period(env: np.ndarray, sr: int,
                          lag_min: float, lag_max: float) -> float:
        """Autocorrelation peak strength in a physiological lag range."""
        N = len(env)
        lag_min_n = int(lag_min * sr)
        lag_max_n = int(min(lag_max * sr, N - 1))
        if lag_min_n >= lag_max_n or N < lag_max_n * 2:
            return 0.0
        x = env - env.mean()
        corr = np.real(
            np.fft.irfft(np.abs(np.fft.rfft(x, n=2 * N)) ** 2)
        )[:N]
        corr /= corr[0] + 1e-12
        window = corr[lag_min_n:lag_max_n]
        return float(np.clip(window.max(), 0.0, 1.0))

    @staticmethod
    def _label(has_heart: bool, has_lung: bool,
               heart_score: float, lung_score: float,
               reason: str = "") -> Dict:
        if not has_heart and not has_lung:
            label = "invalid"
        elif has_heart and has_lung:
            label = "mixed"
        elif has_heart:
            label = "heart"
        else:
            label = "lungs"

        conf = max(heart_score, lung_score)
        d = {"heart": has_heart, "lungs": has_lung,
             "label": label,
             "heart_score": round(heart_score, 4),
             "lung_score":  round(lung_score,  4),
             "confidence":  round(conf, 4)}
        if reason:
            d["reason"] = reason
        return d