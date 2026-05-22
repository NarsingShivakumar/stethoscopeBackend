"""
services/heart_analysis.py
==========================
Stage 4 — Cardiac cycle segmentation + S3/S4 detection.

S1/S2 detection:
  1. Bandpass 20-180 Hz → Hilbert envelope → smoothed at 8 Hz
  2. Peak detection constrained to 30-220 BPM
  3. S2 found as secondary peak within 25-65% of RR interval
  4. Enforce diastole > systole (clinical rule)

S3: 100-250 Hz burst at 60-220 ms after S2 (early diastole)
S4: 20-100 Hz burst at 20-90 ms before S1 (presystole)

All timestamps in milliseconds.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Tuple

import numpy as np

from utils.audio_io import resample, TARGET_SR_ANALYSIS
from utils.dsp import bandpass, hilbert_envelope, find_peaks_physiological, rms

log = logging.getLogger("steth.heart")
SR = TARGET_SR_ANALYSIS

S1_MIN_BPM = 30.0;  S1_MAX_BPM = 220.0
S1_THR     = 1.8;   S1_MIN_W   = 25.0;  S1_MAX_W = 180.0
S2_MIN_F   = 0.25;  S2_MAX_F   = 0.65
S3_BAND    = (100.0, 250.0);  S4_BAND = (20.0, 100.0)
S3_WIN_MS  = (60.0, 220.0);   S4_WIN_MS = (20.0, 90.0)
EXTRA_THR  = 1.4


class HeartAnalysisService:

    def analyse(self, heart_audio: np.ndarray, src_sr: int) -> Dict:
        if rms(heart_audio) < 1e-5:
            return {"cardiac_cycles": [], "extra_sounds": [],
                    "bpm": None, "total_beats": 0}

        audio    = resample(heart_audio, src_sr, SR)
        filtered = bandpass(audio, SR, 20.0, 180.0)
        env      = hilbert_envelope(filtered, smooth_hz=8.0, sr=SR)

        s1s = find_peaks_physiological(
            env, SR, S1_MIN_BPM, S1_MAX_BPM, S1_THR, S1_MIN_W, S1_MAX_W)

        if len(s1s) < 2:
            return {"cardiac_cycles": [], "extra_sounds": [],
                    "bpm": None, "total_beats": len(s1s)}

        cycles, s2s = self._build_cycles(env, s1s, SR)
        extra       = self._extra_sounds(audio, s1s, s2s, SR)

        rr  = np.diff(s1s) / SR
        bpm = round(60.0 / float(np.median(rr)), 1) if len(rr) > 0 else None

        return {"cardiac_cycles": cycles, "extra_sounds": extra,
                "bpm": bpm, "total_beats": len(s1s)}

    # ── Cycle builder ────────────────────────────────────────────────────────

    @staticmethod
    def _build_cycles(env, s1s, sr):
        cycles, s2s = [], []
        for i in range(len(s1s) - 1):
            s1 = int(s1s[i]); nx = int(s1s[i + 1]); rr = nx - s1
            ws = s1 + int(rr * S2_MIN_F)
            we = min(s1 + int(rr * S2_MAX_F), len(env) - 1)
            if we <= ws: continue
            s2 = ws + int(np.argmax(env[ws:we]))
            s2s.append(s2)

            s1_ms  = round(s1 / sr * 1000, 1)
            s2_ms  = round(s2 / sr * 1000, 1)
            nx_ms  = round(nx / sr * 1000, 1)
            sys_d  = round(s2_ms - s1_ms, 1)
            dia_d  = round(nx_ms - s2_ms, 1)

            cycles.append({
                "cycle_id": i + 1,
                "s1_ms": s1_ms, "s2_ms": s2_ms,
                "systole":  {"start_ms": s1_ms, "end_ms": s2_ms,
                             "duration_ms": sys_d},
                "diastole": {"start_ms": s2_ms, "end_ms": nx_ms,
                             "duration_ms": dia_d},
            })
        return cycles, s2s

    # ── S3 / S4 ──────────────────────────────────────────────────────────────

    def _extra_sounds(self, audio, s1s, s2s, sr):
        results = []
        s3_filt = bandpass(audio, sr, *S3_BAND)
        s4_filt = bandpass(audio, sr, *S4_BAND)
        s3_env  = hilbert_envelope(s3_filt, 30.0, sr)
        s4_env  = hilbert_envelope(s4_filt, 20.0, sr)
        global_med = float(np.median(s3_env)) + 1e-9

        for s2 in s2s:
            ws = s2 + int(S3_WIN_MS[0] * sr / 1000)
            we = min(s2 + int(S3_WIN_MS[1] * sr / 1000), len(s3_env) - 1)
            if ws < we:
                seg = s3_env[ws:we]
                peak_v = float(seg.max())
                if peak_v > global_med * EXTRA_THR:
                    off = int(np.argmax(seg))
                    t_ms = round((ws + off) / sr * 1000, 1)
                    results.append({
                        "type": "S3", "time_ms": t_ms,
                        "after_s2_ms": round((ws + off - s2) / sr * 1000, 1),
                        "confidence":  round(min(peak_v / global_med / EXTRA_THR, 1.0), 3),
                        "description": "S3 gallop — early diastolic (LV dysfunction / volume overload)",
                    })

        for s1 in s1s:
            we = s1 - int(S4_WIN_MS[0] * sr / 1000)
            ws = max(0, s1 - int(S4_WIN_MS[1] * sr / 1000))
            if ws < we < len(s4_env):
                seg = s4_env[ws:we]
                med4 = float(np.median(s4_env)) + 1e-9
                peak_v = float(seg.max())
                if peak_v > med4 * EXTRA_THR:
                    off = int(np.argmax(seg))
                    t_ms = round((ws + off) / sr * 1000, 1)
                    results.append({
                        "type": "S4", "time_ms": t_ms,
                        "before_s1_ms": round((s1 - (ws + off)) / sr * 1000, 1),
                        "confidence":   round(min(peak_v / med4 / EXTRA_THR, 1.0), 3),
                        "description":  "S4 gallop — presystolic (reduced LV compliance / HTN)",
                    })
        return results