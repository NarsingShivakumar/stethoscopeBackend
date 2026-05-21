"""
utils/audio_utils.py
====================
Base64 ↔ PCM-16 / WAV helpers for the Flask API.

BluetoothAudioRecorder.java writes PCM raw data then calls addWavHeader()
so every file on disk is a proper 44-byte WAV. The native module
base64-encodes those WAV bytes before POSTing. We handle both WAV-wrapped
and raw PCM-16 transparently.
"""
from __future__ import annotations

import base64
import io
import logging
import wave
from typing import Tuple

import numpy as np

log = logging.getLogger("steth.audio")
PCM_MAX = 32768.0  # 2^15


class AudioDecodeError(ValueError):
    """Raised when the inbound audio payload cannot be decoded."""


# ── Decode ────────────────────────────────────────────────────────────────────

def decode_audio_payload(b64: str, claimed_sr: int) -> Tuple[np.ndarray, int]:
    """
    Decode base64 audio (WAV or raw PCM-16 LE mono).

    Returns (float32 array in [-1,1], sample_rate).
    """
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
            ch   = wf.getnchannels()
            sw   = wf.getsampwidth()
            sr   = wf.getframerate()
            pcm  = wf.readframes(wf.getnframes())
    except Exception as e:
        raise AudioDecodeError(f"WAV parse failed: {e}") from e

    if sw != 2:
        raise AudioDecodeError(f"Only 16-bit PCM WAV supported; got {sw*8}-bit.")

    samples = np.frombuffer(pcm, dtype="<i2")
    if ch == 2:
        samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)
    elif ch != 1:
        raise AudioDecodeError(f"Expected 1-2 channels; got {ch}.")

    return samples.astype(np.float32) / PCM_MAX, sr


def _from_raw_pcm16(data: bytes, sr: int) -> Tuple[np.ndarray, int]:
    if len(data) % 2:
        data = data[:-1]
    samples = np.frombuffer(data, dtype="<i2")
    return samples.astype(np.float32) / PCM_MAX, sr


# ── Encode ────────────────────────────────────────────────────────────────────

def encode_audio_response(audio: np.ndarray, sr: int) -> str:
    """
    Encode float32 audio as base64 WAV at the given sample rate.
    The Android SeparationAudioPlayer decodes this for AudioTrack playback.
    """
    clipped   = np.clip(audio, -1.0, 1.0)
    pcm_bytes = (clipped * PCM_MAX).astype(np.int16).tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_bytes)

    return base64.b64encode(buf.getvalue()).decode("ascii")
