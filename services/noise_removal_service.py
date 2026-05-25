"""
services/noise_removal_service.py
==================================
Stage 1 of the clinical pipeline: remove voice, environmental noise,
and background sounds using SpeechBrain SepFormer.

SepFormer (Subakan et al., ICASSP 2021) is a Transformer-based speech
enhancement model trained on WSJ0-2mix and DNS datasets.
We use the speechbrain/sepformer-whamr-enhancement checkpoint which
handles reverb + noise removal for single-channel audio.

Output: clean biomedical audio (full length preserved, millisecond timestamps).
"""
from __future__ import annotations

import logging
import os
import tempfile
import warnings
from typing import Tuple

import numpy as np
import scipy.signal as sps

log = logging.getLogger("steth.noise_removal")

# SpeechBrain model is loaded lazily (only once, on first call)
_sepformer_model = None
_model_lock = None


def _get_lock():
    global _model_lock
    if _model_lock is None:
        import threading
        _model_lock = threading.Lock()
    return _model_lock


def _load_sepformer():
    """Load SpeechBrain SepFormer enhancement model (thread-safe singleton)."""
    global _sepformer_model
    if _sepformer_model is not None:
        return _sepformer_model

    with _get_lock():
        if _sepformer_model is not None:
            return _sepformer_model

        try:
            from speechbrain.pretrained import SepformerSeparation as separator
            log.info("Loading SpeechBrain SepFormer enhancement model…")
            # speechbrain/sepformer-whamr-enhancement handles single-mic denoising
            _sepformer_model = separator.from_hparams(
                source="speechbrain/sepformer-whamr-enhancement",
                savedir=os.path.expanduser("~/.cache/steth/sepformer"),
                run_opts={"device": "cpu"},
            )
            log.info("SepFormer model loaded successfully.")
        except Exception as e:
            log.warning("SepFormer load failed (%s) — using fallback bandpass denoiser.", e)
            _sepformer_model = "fallback"

    return _sepformer_model


class NoiseRemovalService:
    """
    Remove human voice, environmental noise, and background sounds
    from stethoscope audio. Preserves full length and timestamps.
    """

    def __init__(self, config):
        self.cfg = config
        self._model = None  # lazy load

    def remove_noise(self, audio: np.ndarray, sr: int) -> dict:
        """
        Run SepFormer denoising on raw stethoscope audio.

        Parameters
        ----------
        audio : float32 array shape (N,)
        sr    : original sample rate

        Returns
        -------
        {
            "clean_audio": np.ndarray float32 (N,),   # same length as input
            "noise_segments": [                        # residual noise regions
                {"start_ms": int, "end_ms": int, "rms": float}
            ],
            "snr_estimate_db": float,
        }
        """
        N = len(audio)
        original_sr = sr

        # Step 1 — Resample to 16 kHz (SepFormer requirement)
        audio_16k = _resample(audio, sr, self.cfg.DL_SR)

        # Step 2 — Run SepFormer (or fallback)
        clean_16k = self._enhance(audio_16k, self.cfg.DL_SR)

        # Step 3 — Resample back to original SR
        clean = _resample(clean_16k, self.cfg.DL_SR, original_sr)

        # Step 4 — Match original length exactly (no trimming of content)
        clean = _match(clean, N)
        clean = clean.astype(np.float32)

        # Step 5 — Normalise
        peak = float(np.abs(clean).max())
        if peak > 1e-9:
            clean = clean * (0.95 / peak)

        # Step 6 — Detect residual noise segments
        noise_segments = _detect_noise_segments(audio, clean, original_sr)

        # Step 7 — Estimate SNR improvement
        snr_db = _estimate_snr(audio, clean)

        log.info(
            "NoiseRemoval: input_len=%d clean_len=%d snr=%.1f dB noise_segs=%d",
            N, len(clean), snr_db, len(noise_segments)
        )

        return {
            "clean_audio":    clean,
            "noise_segments": noise_segments,
            "snr_estimate_db": round(snr_db, 2),
        }

    def _enhance(self, audio_16k: np.ndarray, sr: int) -> np.ndarray:
        """Run SepFormer enhancement. Falls back to bandpass on model failure."""
        model = _load_sepformer()

        if model == "fallback":
            return _bandpass_denoise(audio_16k, sr)

        try:
            import torch
            import torchaudio

            # SepFormer expects (1, N) tensor
            wav = torch.from_numpy(audio_16k).unsqueeze(0)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                torchaudio.save(tmp.name, wav, sr)
                tmp_path = tmp.name

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    enhanced = model.separate_file(path=tmp_path)
                # enhanced shape: (1, N, 1) or (1, N)
                if enhanced.dim() == 3:
                    enhanced = enhanced[:, :, 0]
                result = enhanced.squeeze().numpy().astype(np.float32)
                return result
            finally:
                os.unlink(tmp_path)

        except Exception as e:
            log.warning("SepFormer inference failed (%s) — using bandpass fallback.", e)
            return _bandpass_denoise(audio_16k, sr)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _bandpass_denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Fallback: bandpass filter 20–1800 Hz to remove out-of-band noise.
    This is a simple fallback; the real SepFormer handles voice and broadband noise.
    """
    nyq = sr / 2.0
    low = 20.0 / nyq
    high = min(1800.0, nyq * 0.98) / nyq
    if low >= high:
        return audio.copy()
    try:
        sos = sps.butter(8, [low, high], btype="bandpass", output="sos")
        return sps.sosfiltfilt(sos, audio.astype(np.float64)).astype(np.float32)
    except Exception:
        return audio.copy()


def _detect_noise_segments(
    original: np.ndarray,
    clean: np.ndarray,
    sr: int,
    window_ms: int = 50,
    threshold_ratio: float = 0.25,
) -> list:
    """
    Detect residual noise: frames where noise energy > threshold_ratio * clean energy.
    Returns list of {start_ms, end_ms, rms}.
    """
    win   = ms_to_samples(window_ms, sr)
    hop   = win // 2
    segs  = []
    n     = min(len(original), len(clean))

    for i in range(0, n - win, hop):
        orig_frame  = original[i:i + win].astype(np.float64)
        clean_frame = clean[i:i + win].astype(np.float64)
        noise_frame = orig_frame - clean_frame

        clean_rms = float(np.sqrt(np.mean(clean_frame ** 2)))
        noise_rms = float(np.sqrt(np.mean(noise_frame ** 2)))

        if clean_rms < 1e-9:
            continue

        ratio = noise_rms / (clean_rms + noise_rms)
        if ratio > threshold_ratio:
            start_ms = int(round(i * 1000 / sr))
            end_ms   = int(round((i + win) * 1000 / sr))
            segs.append({
                "start_ms": start_ms,
                "end_ms":   end_ms,
                "rms":      round(noise_rms, 6),
            })

    # Merge overlapping segments
    merged = []
    for seg in segs:
        if merged and seg["start_ms"] <= merged[-1]["end_ms"]:
            merged[-1]["end_ms"] = max(merged[-1]["end_ms"], seg["end_ms"])
        else:
            merged.append(dict(seg))

    return merged


def _estimate_snr(noisy: np.ndarray, clean: np.ndarray) -> float:
    """Estimate output SNR in dB."""
    noise = noisy.astype(np.float64) - clean.astype(np.float64)
    clean_power = float(np.mean(clean.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise ** 2))
    if noise_power < 1e-20:
        return 60.0
    return float(10.0 * np.log10(clean_power / noise_power + 1e-9))


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x.astype(np.float32)
    n_out = int(round(len(x) * dst / src))
    return sps.resample_poly(x.astype(np.float64), dst, src)[:n_out].astype(np.float32)


def _match(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - len(x), dtype=np.float32)])


def ms_to_samples(ms: int, sr: int) -> int:
    return int(round(ms * sr / 1000))