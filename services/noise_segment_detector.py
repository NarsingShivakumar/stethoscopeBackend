"""
services/noise_segment_detector.py
====================================
Stage 7 — Post-denoising residual noise segment detection.
Flags any high-frequency (1500-6000 Hz) burst exceeding 3× median energy.
"""
from __future__ import annotations
import logging
from typing import Dict, List
import numpy as np
from utils.audio_io import resample, TARGET_SR_DENOISE
from utils.dsp import bandpass, hilbert_envelope

log = logging.getLogger("steth.noise_seg")
SR = TARGET_SR_DENOISE
NOISE_BAND = (1500.0, 6000.0)
NOISE_THR  = 3.0
MIN_MS     = 50.0
MERGE_MS   = 30.0


class NoiseSegmentDetector:

    def detect(self, audio: np.ndarray, src_sr: int) -> List[Dict]:
        a16  = resample(audio, src_sr, SR)
        filt = bandpass(a16, SR, NOISE_BAND[0], min(NOISE_BAND[1], SR/2*0.95))
        env  = hilbert_envelope(filt, 50.0, SR)
        med  = float(np.median(env)) + 1e-9
        act  = env > med * NOISE_THR
        return _runs(act, SR, MIN_MS, MERGE_MS)


def _runs(active, sr, min_ms, merge_ms):
    min_n = int(min_ms * sr / 1000); mg_n = int(merge_ms * sr / 1000)
    segs = []
    in_r = False; s = 0
    for i in range(len(active)):
        if active[i] and not in_r:   in_r=True; s=i
        elif not active[i] and in_r:
            in_r=False
            if i-s >= min_n: segs.append((s, i))
    if not segs: return []
    merged = [segs[0]]
    for a, b in segs[1:]:
        if a - merged[-1][1] <= mg_n: merged[-1] = (merged[-1][0], b)
        else: merged.append((a, b))
    return [{"start_ms": round(s/sr*1000,1), "end_ms": round(e/sr*1000,1),
             "duration_ms": round((e-s)/sr*1000,1), "type": "residual_noise"}
            for s, e in merged]