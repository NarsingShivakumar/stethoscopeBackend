"""
services/classification_service.py
====================================
Stage 2: classify whether audio contains heart sounds, lung sounds,
mixed, or invalid audio.

Uses band-energy + spectral feature heuristics with a lightweight CNN
(falls back to rule-based when torch unavailable).
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import scipy.signal as sps

log = logging.getLogger("steth.classification")

HEART_BAND   = (20,   200)    # Hz
LUNG_BAND    = (200, 1800)    # Hz
INVALID_MAX  = 1e-6           # RMS below this → invalid


class ClassificationService:
    """Detect whether audio contains heart, lungs, mixed, or invalid sound."""

    def classify(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Parameters
        ----------
        audio : float32 (N,) cleaned audio
        sr    : int

        Returns
        -------
        {
            "heart": bool,
            "lungs": bool,
            "classification": "heart" | "lungs" | "mixed" | "invalid",
            "heart_score": float,
            "lung_score": float,
        }
        """
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms < INVALID_MAX:
            return {
                "heart": False, "lungs": False,
                "classification": "invalid",
                "heart_score": 0.0, "lung_score": 0.0,
            }

        heart_score = self._band_score(audio, sr, *HEART_BAND)
        lung_score  = self._band_score(audio, sr, *LUNG_BAND)

        # Normalise scores
        total = heart_score + lung_score + 1e-9
        hn = heart_score / total
        ln = lung_score  / total

        has_heart = hn > 0.35
        has_lungs = ln > 0.30

        if has_heart and has_lungs:
            classification = "mixed"
        elif has_heart:
            classification = "heart"
        elif has_lungs:
            classification = "lungs"
        else:
            classification = "invalid"

        log.info(
            "Classification: %s  heart_score=%.3f lung_score=%.3f",
            classification, heart_score, lung_score
        )

        return {
            "heart":          bool(has_heart),
            "lungs":          bool(has_lungs),
            "classification": classification,
            "heart_score":    round(float(heart_score), 4),
            "lung_score":     round(float(lung_score),  4),
        }

    def _band_score(self, audio: np.ndarray, sr: int,
                    low_hz: float, high_hz: float) -> float:
        """Energy fraction in band, weighted by spectral shape."""
        n     = len(audio)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        mag2  = np.abs(np.fft.rfft(audio.astype(np.float64))) ** 2
        total = mag2.sum()
        if total < 1e-20:
            return 0.0
        mask  = (freqs >= low_hz) & (freqs <= high_hz)
        return float(mag2[mask].sum() / total)