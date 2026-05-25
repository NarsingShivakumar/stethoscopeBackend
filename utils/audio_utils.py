"""
utils/audio_utils.py
====================
Base64 ↔ PCM-16 / WAV helpers + file-upload helpers.

New in v5:
  - decode_uploaded_file(file_storage) → (np.ndarray, int)
  - samples_to_ms(n_samples, sr) → int
  - ms_to_samples(ms, sr) → int
"""
from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
import wave
from typing import Tuple

import numpy as np

log = logging.getLogger("steth.audio")
PCM_MAX = 32768.0


class AudioDecodeError(ValueError):
    """Raised when the inbound audio payload cannot be decoded."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def samples_to_ms(n: int, sr: int) -> int:
    """Convert sample count to milliseconds."""
    return int(round(n * 1000 / sr))


def ms_to_samples(ms: int, sr: int) -> int:
    """Convert milliseconds to sample count."""
    return int(round(ms * sr / 1000))


# ── Decode base64 ─────────────────────────────────────────────────────────────

def decode_audio_payload(b64: str, claimed_sr: int) -> Tuple[np.ndarray, int]:
    """Decode base64 audio (WAV or raw PCM-16 LE mono)."""
    try:
        raw = base64.b64decode(b64, validate=False)
    except Exception as e:
        raise AudioDecodeError(f"Base64 decode failed: {e}") from e

    if len(raw) == 0:
        raise AudioDecodeError("Decoded audio is empty.")

    if raw[:4] == b"RIFF":
        return _from_wav(raw)
    return _from_raw_pcm16(raw, claimed_sr)


def _from_wav(data: bytes) -> Tuple[np.ndarray, int]:
    try:
        buf = io.BytesIO(data)
        with wave.open(buf, "rb") as wf:
            ch  = wf.getnchannels()
            sw  = wf.getsampwidth()
            sr  = wf.getframerate()
            pcm = wf.readframes(wf.getnframes())
    except Exception as e:
        raise AudioDecodeError(f"WAV parse failed: {e}") from e

    if sw != 2:
        raise AudioDecodeError(f"Only 16-bit PCM WAV supported; got {sw*8}-bit.")

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / PCM_MAX
    if ch == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    return samples, sr


def _from_raw_pcm16(data: bytes, sr: int) -> Tuple[np.ndarray, int]:
    if len(data) % 2:
        data = data[:-1]
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / PCM_MAX
    return samples, sr


# ── Decode uploaded file (WAV or MP3 via torchaudio) ──────────────────────────

def decode_uploaded_file(file_storage) -> Tuple[np.ndarray, int]:
    """
    Accept a Werkzeug FileStorage (wav/mp3) → (float32 array [-1,1], sr).
    Full-length — no trimming.
    """
    import torchaudio
    import torch

    suffix = os.path.splitext(file_storage.filename or "audio.wav")[1].lower()
    if suffix not in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"):
        suffix = ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file_storage.save(tmp.name)
        tmp_path = tmp.name

    try:
        waveform, sr = torchaudio.load(tmp_path)  # (C, N) float32
        # Mix down to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        audio = waveform.squeeze().numpy()
        # Normalise to [-1, 1]
        peak = float(np.abs(audio).max())
        if peak > 1e-9:
            audio = audio / peak * 0.95
        return audio.astype(np.float32), int(sr)
    finally:
        os.unlink(tmp_path)


# ── Encode ─────────────────────────────────────────────────────────────────────

def encode_audio_response(audio: np.ndarray, sr: int) -> str:
    """Encode float32 audio as base64 WAV."""
    clipped   = np.clip(audio, -1.0, 1.0)
    pcm_bytes = (clipped * PCM_MAX).astype(np.int16).tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_bytes)

    return base64.b64encode(buf.getvalue()).decode("ascii")


def save_wav(audio: np.ndarray, sr: int, path: str) -> None:
    """Save float32 audio to a WAV file on disk."""
    clipped   = np.clip(audio, -1.0, 1.0)
    pcm_bytes = (clipped * PCM_MAX).astype(np.int16).tobytes()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_bytes)