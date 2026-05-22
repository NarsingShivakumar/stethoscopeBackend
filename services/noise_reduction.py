"""
services/noise_reduction.py
============================
Stage 1 — Speech & environment noise removal using SpeechBrain SepFormer.

Model: speechbrain/sepformer-wham16k-enhancement
  - Trained on WHAM! dataset (speech + noise)
  - Input/output: 16 kHz mono
  - Removes: human voice, environmental noise, background sounds

Falls back to spectral-subtraction + Wiener filter if SpeechBrain
is unavailable (no GPU / model not downloaded yet).
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from utils.audio_io import resample, TARGET_SR_DENOISE
from utils.dsp import bandpass, rms

log = logging.getLogger("steth.noise_reduction")

HF_MODEL = "speechbrain/sepformer-wham16k-enhancement"
SB_SAVEDIR = "models/sepformer"


class NoiseReductionService:
    """
    Wraps SpeechBrain SepFormer for audio enhancement.
    Lazy-loads the model on first call.
    """

    def __init__(self):
        self._model = None
        self._use_sb = True   # flip to False if import fails

    def _load(self):
        if self._model is not None:
            return
        try:
            from speechbrain.pretrained import SepformerSeparation as sep
            self._model = sep.from_hparams(
                source=HF_MODEL,
                savedir=SB_SAVEDIR,
                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            )
            log.info("SepFormer loaded  device=%s",
                     "cuda" if torch.cuda.is_available() else "cpu")
        except Exception as exc:
            log.warning("SpeechBrain unavailable (%s) — using DSP fallback", exc)
            self._use_sb = False

    def denoise(self, audio: np.ndarray, src_sr: int) -> np.ndarray:
        """
        Remove noise from audio.  Returns cleaned float32 array
        at src_sr (original sample rate preserved).
        """
        self._load()

        # Resample to 16 kHz for SepFormer
        audio16 = resample(audio, src_sr, TARGET_SR_DENOISE)

        if self._use_sb and self._model is not None:
            cleaned16 = self._sb_enhance(audio16)
        else:
            cleaned16 = self._dsp_enhance(audio16, TARGET_SR_DENOISE)

        # Resample back to original SR
        cleaned = resample(cleaned16, TARGET_SR_DENOISE, src_sr)

        # Normalise amplitude (no clipping)
        peak = float(np.abs(cleaned).max())
        if peak > 1e-6:
            cleaned = cleaned / peak * 0.95

        return cleaned.astype(np.float32)

    # ── SpeechBrain path ───────────────────────────────────────────────────────

    def _sb_enhance(self, audio16: np.ndarray) -> np.ndarray:
        """Run SepFormer on a 16 kHz float32 array."""
        t = torch.tensor(audio16).unsqueeze(0)   # (1, T)
        with torch.no_grad():
            out = self._model.separate_batch(t)  # (1, T, 1)
        enhanced = out[0, :, 0].cpu().numpy().astype(np.float32)
        # Match length exactly
        N = len(audio16)
        if len(enhanced) > N:
            enhanced = enhanced[:N]
        elif len(enhanced) < N:
            enhanced = np.pad(enhanced, (0, N - len(enhanced)))
        return enhanced

    # ── DSP fallback: spectral subtraction + Wiener ───────────────────────────

    @staticmethod
    def _dsp_enhance(audio16: np.ndarray, sr: int) -> np.ndarray:
        """
        Spectral-subtraction + Wiener filter fallback.
        Step 1: Estimate noise floor from first 200 ms.
        Step 2: Subtract noise spectrum.
        Step 3: Apply Wiener gain.
        Step 4: Bandpass to biomedical range (20–8000 Hz).
        """
        import scipy.signal as sps

        nperseg = 512
        noverlap = 384
        f, t, Zxx = sps.stft(audio16.astype(np.float64),
                              fs=sr, nperseg=nperseg, noverlap=noverlap)
        mag   = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Noise estimate: first 200 ms
        noise_frames = max(1, int(0.200 * sr / (nperseg - noverlap)))
        noise_est    = mag[:, :noise_frames].mean(axis=1, keepdims=True)

        # Spectral subtraction (half-wave rectify)
        alpha     = 2.0   # over-subtraction factor
        beta      = 0.02  # spectral floor
        mag_clean = np.maximum(mag - alpha * noise_est, beta * mag)

        # Wiener gain
        wiener_gain = mag_clean ** 2 / (mag_clean ** 2 + noise_est ** 2 + 1e-12)
        mag_wiener  = mag * wiener_gain

        Zxx_clean = mag_wiener * np.exp(1j * phase)
        _, cleaned = sps.istft(Zxx_clean, fs=sr,
                               nperseg=nperseg, noverlap=noverlap)

        # Bandpass to biomedical range
        from utils.dsp import bandpass as bp
        cleaned = bp(cleaned.astype(np.float32), sr, 20.0, 8000.0)
        N = len(audio16)
        return cleaned[:N].astype(np.float32) if len(cleaned) >= N \
               else np.pad(cleaned, (0, N - len(cleaned))).astype(np.float32)