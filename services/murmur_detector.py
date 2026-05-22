"""
services/murmur_detector.py
============================
Stage 5 — Full murmur detection covering ALL clinical subtypes.

SYSTOLIC: holosystolic | midsystolic | early_systolic | late_systolic
DIASTOLIC: early_diastolic | mid_diastolic | presystolic

Per murmur: start_ms, end_ms, phase, type, pattern,
            confidence, murmur_energy_ratio, possible_condition
"""
from __future__ import annotations
import logging
from typing import Dict, List, Tuple

import numpy as np
import scipy.signal as sps

from utils.audio_io import resample, TARGET_SR_ANALYSIS
from utils.dsp import bandpass, rms

log = logging.getLogger("steth.murmur")

SR               = TARGET_SR_ANALYSIS
MURMUR_LOW       = 80.0
MURMUR_HIGH      = 600.0
MURMUR_RATIO_THR = 0.32
HOLO_THR         = 0.70
MIN_DUR_MS       = 50.0
FRAME_MS         = 10.0
HOP_MS           = 5.0


def _ste(seg: np.ndarray, sr: int) -> np.ndarray:
    """Short-time energy curve."""
    frame_n = max(1, int(FRAME_MS * sr / 1000))
    hop_n   = max(1, int(HOP_MS   * sr / 1000))
    n       = max(1, 1 + (len(seg) - frame_n) // hop_n)
    e       = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = i * hop_n
        f = seg[s: s + frame_n].astype(np.float64)
        e[i] = float(np.mean(f ** 2))
    return e


def _pattern(ste: np.ndarray) -> Tuple[str, float]:
    """Classify temporal shape of STE curve."""
    if len(ste) < 3:
        return "plateau", 0.5
    n    = len(ste)
    sn   = ste / (ste.max() + 1e-9)
    x    = np.linspace(0, 1, n)
    p1   = np.polyfit(x, sn, 1)
    trend= p1[0]
    resid= sn - np.polyval(p1, x)
    flat = 1.0 - float(np.std(resid))
    pk   = int(np.argmax(sn)) / max(n - 1, 1)
    is_cd= 0.15 < pk < 0.85 and sn[int(np.argmax(sn))] > 0.70
    if is_cd and abs(trend) < 0.30:
        return "crescendo_decrescendo", float(np.clip(sn.max(), 0, 1))
    if trend > 0.20:  return "crescendo",   float(np.clip(trend, 0, 1))
    if trend < -0.20: return "decrescendo", float(np.clip(-trend, 0, 1))
    return "plateau", float(np.clip(flat, 0, 1))


class MurmurDetector:

    def detect(self, heart_audio: np.ndarray,
               src_sr: int, cardiac_cycles: List[Dict]) -> List[Dict]:
        if not cardiac_cycles or rms(heart_audio) < 1e-5:
            return []

        audio = resample(heart_audio, src_sr, SR)
        filt  = bandpass(audio, SR, MURMUR_LOW, MURMUR_HIGH)
        out   = []

        for cyc in cardiac_cycles:
            out.extend(self._systole(filt, cyc))
            out.extend(self._diastole(filt, cyc))

        return _merge(out)

    # ── Systolic ──────────────────────────────────────────────────────────────

    def _systole(self, filt, cyc):
        s1  = cyc["systole"]["start_ms"]
        s2  = cyc["systole"]["end_ms"]
        dur = s2 - s1
        if dur < MIN_DUR_MS: return []

        s1n = int(s1 * SR / 1000); s2n = min(int(s2 * SR / 1000), len(filt))
        seg = filt[s1n:s2n]
        if len(seg) < 4: return []

        bw  = max(1, int(0.06 * SR))
        be  = float(np.mean(filt[max(0,s1n-bw):s1n+bw].astype(np.float64)**2)) + 1e-12
        e   = _ste(seg, SR)
        ratio = float(np.mean(e.astype(np.float64))) / be
        if ratio < MURMUR_RATIO_THR: return []

        pat, pc = _pattern(e)
        en      = e / (e.max() + 1e-9)
        flat    = 1.0 - float(np.std(en))
        pk_rel  = int(np.argmax(e)) / max(len(e) - 1, 1)

        if flat >= HOLO_THR:
            mtype = "holosystolic"
            cond  = "Mitral regurgitation / Tricuspid regurgitation / VSD"
        elif pat == "crescendo_decrescendo":
            mtype = "midsystolic"
            cond  = "Aortic stenosis / Pulmonic stenosis / HOCM"
        elif pk_rel < 0.25:
            mtype = "early_systolic"
            cond  = "Small VSD / Acute MR / Tricuspid regurgitation"
        else:
            mtype = "late_systolic"
            cond  = "Mitral valve prolapse / Mitral regurgitation"

        conf = float(np.clip(0.60 * min(ratio / 0.80, 1) + 0.40 * pc, 0, 1))
        return [{"start_ms": round(s1,1), "end_ms": round(s2,1),
                 "phase": "systolic", "type": mtype, "pattern": pat,
                 "confidence": round(conf, 4),
                 "murmur_energy_ratio": round(ratio, 4),
                 "possible_condition": cond}]

    # ── Diastolic ─────────────────────────────────────────────────────────────

    def _diastole(self, filt, cyc):
        s2  = cyc["diastole"]["start_ms"]
        nx  = cyc["diastole"]["end_ms"]
        dur = nx - s2
        if dur < MIN_DUR_MS: return []

        s2n = int(s2 * SR / 1000); nxn = min(int(nx * SR / 1000), len(filt))
        seg = filt[s2n:nxn]
        if len(seg) < 4: return []

        bw  = max(1, int(0.06 * SR))
        be  = float(np.mean(filt[max(0,s2n-bw):s2n+bw].astype(np.float64)**2)) + 1e-12
        e   = _ste(seg, SR)
        ratio = float(np.mean(e.astype(np.float64))) / be
        if ratio < MURMUR_RATIO_THR: return []

        pat, pc  = _pattern(e)
        pk_rel   = int(np.argmax(e)) / max(len(e) - 1, 1)

        if pk_rel < 0.30 and pat in ("decrescendo", "crescendo_decrescendo"):
            mtype = "early_diastolic"
            cond  = "Aortic regurgitation / Pulmonic regurgitation"
        elif 0.25 <= pk_rel <= 0.75:
            mtype = "mid_diastolic"
            cond  = "Mitral stenosis / Tricuspid stenosis / Austin Flint"
        else:
            mtype = "presystolic"
            cond  = "Mitral stenosis (sinus rhythm) / Tricuspid stenosis"

        conf = float(np.clip(0.60 * min(ratio / 0.80, 1) + 0.40 * pc, 0, 1))
        return [{"start_ms": round(s2,1), "end_ms": round(nx,1),
                 "phase": "diastolic", "type": mtype, "pattern": pat,
                 "confidence": round(conf, 4),
                 "murmur_energy_ratio": round(ratio, 4),
                 "possible_condition": cond}]


def _merge(murmurs):
    if not murmurs: return []
    murmurs = sorted(murmurs, key=lambda m: m["start_ms"])
    merged = [murmurs[0]]
    for m in murmurs[1:]:
        p = merged[-1]
        if m["start_ms"] <= p["end_ms"] and m["type"] == p["type"]:
            p["end_ms"]     = max(p["end_ms"], m["end_ms"])
            p["confidence"] = max(p["confidence"], m["confidence"])
        else:
            merged.append(m)
    return merged