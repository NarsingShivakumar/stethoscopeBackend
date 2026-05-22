"""
services/lung_analysis.py
=========================
Stage 6 — Lung classification: normal | wheeze | crackles | mixed

Crackles: short (<20 ms) transient energy spikes
Wheezes:  sustained high-pitched sounds (200-800 Hz, >100 ms)
Normal:   smooth low-frequency (100-500 Hz) dominant
"""
from __future__ import annotations
import logging
from typing import Dict, List

import numpy as np
import scipy.signal as sps

from utils.audio_io import resample, TARGET_SR_DENOISE
from utils.dsp import bandpass, hilbert_envelope, rms

log = logging.getLogger("steth.lung")
SR = TARGET_SR_DENOISE   # 16 kHz for lung detail

LUNG_BAND   = (100.0, 2000.0)
WHEEZE_BAND = (200.0, 800.0)
CRACK_MS    = 20.0
WHEEZE_SMIN = 0.10


class LungAnalysisService:

    def analyse(self, lung_audio: np.ndarray, src_sr: int) -> Dict:
        if rms(lung_audio) < 1e-5:
            return self._empty()

        audio = resample(lung_audio, src_sr, SR)
        filt  = bandpass(audio, SR, *LUNG_BAND)
        env   = hilbert_envelope(filt, 50.0, SR)

        cs, ce = self._crackles(filt, env, SR)
        ws, we = self._wheezes(filt, SR)
        ns     = self._normal_score(filt, SR)

        has_w  = ws > 0.35; has_c = cs > 0.35
        label  = ("mixed" if has_w and has_c
                  else "wheeze" if has_w
                  else "crackles" if has_c
                  else "normal")

        total = cs + ws + ns + 1e-9
        conf  = max(cs, ws, ns) / total
        return {"label": label,
                "wheeze_score": round(ws,4), "crackle_score": round(cs,4),
                "normal_score": round(ns,4), "confidence": round(conf,4),
                "crackle_events": ce, "wheeze_events": we}

    @staticmethod
    def _crackles(filt, env, sr):
        max_w  = int(CRACK_MS * sr / 1000)
        median = float(np.median(env))
        if median < 1e-9: return 0.0, []
        try:
            peaks, props = sps.find_peaks(
                env.astype(np.float64), height=median*2.5,
                distance=max(1, int(0.010*sr)),
                width=(1, max_w), prominence=median*1.5)
        except Exception:
            return 0.0, []
        widths = props.get("widths", np.zeros(len(peaks)))
        events = [{"time_ms": round(p/sr*1000,1),
                   "duration_ms": round(w/sr*1000,1)}
                  for p, w in zip(peaks, widths)]
        rate  = len(events) / max(len(filt)/sr, 0.1)
        score = float(np.clip(rate/15.0, 0, 1))
        return score, events

    @staticmethod
    def _wheezes(filt, sr):
        wf    = bandpass(filt, sr, *WHEEZE_BAND)
        env_w = hilbert_envelope(wf, 10.0, sr)
        med   = float(np.median(env_w)) + 1e-9
        act   = env_w > med * 1.8
        mlen  = int(WHEEZE_SMIN * sr)
        events: List[Dict] = []
        in_r = False; start = 0
        for i in range(len(act)):
            if act[i] and not in_r:   in_r=True; start=i
            elif not act[i] and in_r:
                in_r = False
                if i - start >= mlen:
                    seg  = wf[start:i]
                    sp   = np.abs(np.fft.rfft(seg))
                    freq = np.fft.rfftfreq(len(seg), 1.0/sr)
                    events.append({"start_ms": round(start/sr*1000,1),
                                   "end_ms": round(i/sr*1000,1),
                                   "freq_hz": round(float(freq[np.argmax(sp)]),1)})
        tot_s = sum((e["end_ms"]-e["start_ms"]) for e in events)/1000.0
        score = float(np.clip(tot_s/max(len(filt)/sr*0.3,0.1), 0, 1))
        return score, events

    @staticmethod
    def _normal_score(filt, sr):
        el = rms(bandpass(filt, sr, 100.0, 500.0))
        eh = rms(bandpass(filt, sr, 500.0, 2000.0))
        r  = el / (el + eh + 1e-9)
        return float(np.clip((r - 0.30) / 0.40, 0, 1))

    @staticmethod
    def _empty():
        return {"label":"normal","wheeze_score":0.0,"crackle_score":0.0,
                "normal_score":0.0,"confidence":0.0,
                "crackle_events":[],"wheeze_events":[]}