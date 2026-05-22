"""utils/audio_io.py — unified audio loading, resampling, saving."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import soundfile as sf
import resampy

log = logging.getLogger("steth.audio_io")

TARGET_SR_DENOISE  = 16000   # SpeechBrain SepFormer requirement
TARGET_SR_ANALYSIS = 4000    # NeoSSNet / cardiac analysis


def load_audio(path: str | Path) -> Tuple[np.ndarray, int]:
    """Load any audio file → (float32 mono ndarray, sample_rate)."""
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    mono = data.mean(axis=1)   # stereo → mono, preserves full length
    log.debug("load_audio  file=%s  sr=%d  samples=%d", path, sr, len(mono))
    return mono, sr


def resample(audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    """High-quality resampling using resampy (sinc interpolation)."""
    if src_sr == tgt_sr:
        return audio.copy()
    return resampy.resample(
        audio.astype(np.float64), src_sr, tgt_sr
    ).astype(np.float32)


def save_wav(audio: np.ndarray, sr: int, path: str | Path) -> None:
    """Save float32 ndarray as 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    log.debug("save_wav  path=%s  sr=%d  samples=%d", path, sr, len(audio))