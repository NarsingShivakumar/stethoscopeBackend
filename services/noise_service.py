"""
services/noise_service.py  v3 — Complete rewrite
================================================
BUGS FIXED:
  1. SR MISMATCH: NMF runs at TARGET_SR=4000 Hz. Separated heart WAV is at
     4000 Hz. If frontend sends sample_rate=44100, heart band 20-150 Hz appears
     at 220-1650 Hz -> spectral feature is completely wrong.
     FIX: Frontend now passes sample_rate=4000 (NMF_TARGET_SR). Endpoint also
     auto-reads SR from WAV header if sample_rate=0 is passed.

  2. FRONTEND SENT ORIGINAL AUDIO: payloadLen was same as original file.
     FIX: Frontend now passes `heart` from Redux (NMF output), not original.

  3. VOICE FOOLS TRANSIENT CHECK: Voiced speech has AM bursts like S1/S2.
     FIX: Added duty cycle check (voice > 55% active, heart < 30%) and
     inter-burst asymmetry check (S1-S2 short, S2-S1 long, ratio >= 1.4).

  4. SPECTRAL CENTROID ALONE NOT ENOUGH:
     FIX: Added spectral flatness (tonal heart < 0.15, noisy voice > 0.25).

ALGORITHM v3 — 5 features, weighted combination:
  Feature 1  SPECTRAL     (30%): centroid < 200 Hz, HF < 25%, flatness < 0.30
  Feature 2  TRANSIENT    (25%): >= 2 short bursts (S1, S2) in valid range
  Feature 3  DUTY CYCLE   (20%): active < 30% of recording (heart is punctuated)
  Feature 4  ASYMMETRY    (15%): S1-S2 gap shorter than S2-S1 gap (ratio >= 1.4)
  Feature 5  ZCR          (10%): low zero-crossing rate in heart band

  3 veto rules apply on top (see code).
  confidence threshold raised to 0.50.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.signal as sps

log = logging.getLogger("steth.noise")

NOISE_VOICE  = "voice"
NOISE_WHITE  = "white"
NOISE_PINK   = "pink"
NOISE_BROWN  = "brown"
VALID_NOISE_TYPES = {NOISE_VOICE, NOISE_WHITE, NOISE_PINK, NOISE_BROWN}

# Spectral (all Hz values calibrated for 4000 Hz SR)
HEART_BAND_LOW   = 20.0
HEART_BAND_HIGH  = 150.0
HF_BOUNDARY      = 500.0
CENTROID_IDEAL   = 80.0
CENTROID_MAX     = 200.0
HF_IDEAL         = 0.05
HF_MAX           = 0.25
FLATNESS_IDEAL   = 0.12
FLATNESS_MAX     = 0.30

# Transient
TRANSIENT_RATIO  = 2.5
MIN_GAP_S        = 0.10
MAX_GAP_S        = 1.60
BURST_MAX_S      = 0.20
MIN_BURSTS       = 2

# Duty cycle
DUTY_MAX_HEART   = 0.30
DUTY_MIN_VOICE   = 0.55

# Inter-burst asymmetry
ASYMMETRY_MIN    = 1.4
ASYMMETRY_MAX    = 8.0

# ZCR
ZCR_IDEAL        = 50.0
ZCR_MAX          = 180.0

# Weights
W_SPECTRAL   = 0.30
W_TRANSIENT  = 0.25
W_DUTY       = 0.20
W_ASYMMETRY  = 0.15
W_ZCR        = 0.10

HEART_DETECT_THRESHOLD = 0.50


class NoiseService:

    def mix_noise(self, audio, sr, noise_type=NOISE_WHITE, snr_db=10.0):
        noise_type = (noise_type or NOISE_WHITE).lower()
        if noise_type not in VALID_NOISE_TYPES:
            noise_type = NOISE_WHITE
        N     = len(audio)
        noise = self._generate_noise(noise_type, N, sr)
        sig_rms   = _rms(audio)
        noise_rms = _rms(noise)
        if sig_rms < 1e-9 or noise_rms < 1e-9:
            return audio.copy()
        desired = sig_rms / (10.0 ** (snr_db / 20.0))
        noisy   = audio + (noise * (desired / noise_rms)).astype(np.float32)
        peak = float(np.abs(noisy).max())
        if peak > 1.0:
            noisy = noisy / peak * 0.95
        return noisy.astype(np.float32)

    def detect_heart(self, audio: np.ndarray, sr: int) -> Dict:
        audio = audio.astype(np.float64)
        peak = float(np.abs(audio).max())
        if peak > 1e-9:
            audio = audio / peak

        filtered = _bandpass(audio, sr, HEART_BAND_LOW, HEART_BAND_HIGH)
        env      = _envelope(filtered, sr, smooth_hz=min(30.0, sr/2.0*0.95))

        spectral_score, centroid_hz, hf_ratio, flatness = _spectral_score(audio, sr)
        transient_score, n_bursts, all_gaps_s           = _transient_score(env, sr)
        duty_score, duty_cycle                          = _duty_score(env)
        asymmetry_score, asym_ratio                     = _asymmetry_score(all_gaps_s)
        zcr_score, zcr_per_sec                          = _zcr_score(filtered, sr)

        confidence = float(np.clip(
            W_SPECTRAL  * spectral_score  +
            W_TRANSIENT * transient_score +
            W_DUTY      * duty_score      +
            W_ASYMMETRY * asymmetry_score +
            W_ZCR       * zcr_score,
            0.0, 1.0,
        ))

        # Veto rules
        if hf_ratio    > 0.30: confidence = min(confidence, 0.35)
        if duty_cycle  > 0.65: confidence = min(confidence, 0.35)
        if flatness    > 0.35: confidence = min(confidence, 0.40)

        heart_detected = confidence >= HEART_DETECT_THRESHOLD

        log.info(
            "detect_heart v3  spec=%.3f trans=%.3f duty=%.3f asym=%.3f zcr=%.3f "
            "conf=%.3f det=%s  centHz=%.0f hf=%.3f flat=%.3f dutyCyc=%.2f bursts=%d",
            spectral_score, transient_score, duty_score, asymmetry_score, zcr_score,
            confidence, heart_detected, centroid_hz, hf_ratio, flatness,
            duty_cycle, n_bursts,
        )

        return {
            "heart_detected":    bool(heart_detected),
            "confidence":        round(confidence, 4),
            "spectral_score":    round(spectral_score, 4),
            "transient_score":   round(transient_score, 4),
            "duty_score":        round(duty_score, 4),
            "asymmetry_score":   round(asymmetry_score, 4),
            "zcr_score":         round(zcr_score, 4),
            "dominant_bpm":      _estimate_bpm(all_gaps_s),
            "n_transients":      int(n_bursts),
            "hf_ratio":          round(hf_ratio, 4),
            "centroid_hz":       round(centroid_hz, 1),
            "spectral_flatness": round(flatness, 4),
            "duty_cycle":        round(duty_cycle, 3),
        }

    def _generate_noise(self, noise_type, N, sr):
        rng = np.random.default_rng()
        if noise_type == NOISE_WHITE:  return rng.standard_normal(N).astype(np.float32)
        if noise_type == NOISE_PINK:   return _pink_noise(N, rng)
        if noise_type == NOISE_BROWN:  return _brown_noise(N, rng)
        if noise_type == NOISE_VOICE:  return _voice_like_noise(N, sr, rng)
        return rng.standard_normal(N).astype(np.float32)


def _spectral_score(audio, sr):
    N = len(audio)
    win   = np.hanning(N)
    X     = np.abs(np.fft.rfft(audio * win)) + 1e-12
    freqs = np.fft.rfftfreq(N, d=1.0/sr)
    power = X ** 2
    total = power.sum()
    if total < 1e-12:
        return 0.0, 0.0, 0.0, 1.0

    centroid_hz = float(np.sum(freqs * power) / total)
    hf_ratio    = float(power[freqs >= HF_BOUNDARY].sum() / total)
    log_mean    = float(np.exp(np.mean(np.log(X))))
    arith_mean  = float(np.mean(X))
    flatness    = float(np.clip(log_mean / (arith_mean + 1e-12), 0.0, 1.0))

    c_s = float(np.clip(1.0 - (centroid_hz - CENTROID_IDEAL) / (CENTROID_MAX - CENTROID_IDEAL), 0, 1))
    h_s = float(np.clip(1.0 - (hf_ratio    - HF_IDEAL      ) / (HF_MAX       - HF_IDEAL),       0, 1))
    f_s = float(np.clip(1.0 - (flatness    - FLATNESS_IDEAL ) / (FLATNESS_MAX - FLATNESS_IDEAL),  0, 1))

    score = float((c_s * h_s * f_s) ** (1.0 / 3.0))
    return float(np.clip(score, 0, 1)), centroid_hz, hf_ratio, flatness


def _transient_score(env, sr):
    if len(env) < int(sr * 0.3):
        return 0.0, 0, []
    median_env = float(np.median(env))
    thresh     = max(median_env * TRANSIENT_RATIO, float(np.max(env)) * 0.15, 1e-9)
    peak_sep   = int(MIN_GAP_S * sr)
    peaks, _   = sps.find_peaks(env, height=thresh, distance=peak_sep)
    if len(peaks) < MIN_BURSTS:
        return 0.0, len(peaks), []

    burst_max = int(BURST_MAX_S * sr)
    valid = []
    for p in peaks:
        half = env[p] * 0.5
        lo, hi = p, p
        while lo > 0 and env[lo] > half: lo -= 1
        while hi < len(env) - 1 and env[hi] > half: hi += 1
        if (hi - lo) <= burst_max:
            valid.append(p)

    n = len(valid)
    if n < MIN_BURSTS:
        return float(np.clip(n/6.0, 0, 0.25)), n, []

    gaps_s   = list(np.diff(np.array(valid)) / sr)
    in_range = [MIN_GAP_S <= g <= MAX_GAP_S for g in gaps_s]
    good_frac = sum(in_range) / len(in_range) if in_range else 0.0
    n_score  = float(np.clip(n / 10.0, 0, 1))
    return float(np.clip(n_score * good_frac, 0, 1)), n, gaps_s


def _duty_score(env):
    if len(env) == 0: return 0.0, 0.0
    p90 = float(np.percentile(env, 90))
    if p90 < 1e-9: return 0.0, 0.0
    duty_cycle = float((env >= p90 * 0.20).mean())
    score = float(np.clip(
        1.0 - (duty_cycle - DUTY_MAX_HEART) / (DUTY_MIN_VOICE - DUTY_MAX_HEART), 0, 1))
    return score, duty_cycle


def _asymmetry_score(gaps_s):
    if len(gaps_s) < 2: return 0.0, None
    gaps  = np.array(gaps_s)
    med   = float(np.median(gaps))
    short = gaps[gaps <= med]
    long_ = gaps[gaps >  med]
    if len(short) == 0 or len(long_) == 0: return 0.0, None
    ratio = float(np.median(long_) / (np.median(short) + 1e-9))
    if ratio < ASYMMETRY_MIN: return 0.0, ratio
    if ratio > ASYMMETRY_MAX: return 0.3, ratio
    score = float(np.clip(1.0 - abs(ratio - 3.0) / 3.0, 0, 1))
    return score, ratio


def _zcr_score(bandpassed, sr):
    if len(bandpassed) < 2: return 0.0, 0.0
    crossings = int(np.sum(np.abs(np.diff(np.sign(bandpassed))) > 0))
    zcr_s     = crossings / (len(bandpassed) / sr)
    score     = float(np.clip(1.0 - (zcr_s - ZCR_IDEAL) / (ZCR_MAX - ZCR_IDEAL), 0, 1))
    return score, zcr_s


def _estimate_bpm(gaps_s):
    if len(gaps_s) < 2: return None
    pairs = [(gaps_s[i] + gaps_s[i+1]) for i in range(0, len(gaps_s)-1, 2)]
    if not pairs: return None
    period = float(np.median(pairs))
    return round(60.0 / period, 1) if period > 0.1 else None


def _rms(x): return float(np.sqrt(np.mean(x.astype(np.float64)**2)))

def _bandpass(x, sr, low, high, order=4):
    nyq = sr/2.0; low = max(low,1.0); high = min(high, nyq*0.99)
    if low >= high: return x.copy()
    try:
        sos = sps.butter(order, [low/nyq, high/nyq], btype="bandpass", output="sos")
        return sps.sosfiltfilt(sos, x.astype(np.float64)).astype(np.float32)
    except: return x.copy()

def _envelope(x, sr, smooth_hz=30.0):
    env  = np.abs(sps.hilbert(x.astype(np.float64))).astype(np.float32)
    nyq  = sr/2.0; freq = min(smooth_hz, nyq*0.95)
    try:
        sos = sps.butter(4, freq/nyq, btype="low", output="sos")
        env = sps.sosfiltfilt(sos, env.astype(np.float64)).astype(np.float32)
    except: pass
    return env

def _pink_noise(N, rng):
    X = np.fft.rfft(rng.standard_normal(N))
    f = np.fft.rfftfreq(N); f[0] = 1e-9
    X *= 1.0 / np.sqrt(f)
    x = np.fft.irfft(X, n=N)
    return (x/(np.abs(x).max()+1e-9)).astype(np.float32)

def _brown_noise(N, rng):
    x = np.cumsum(rng.standard_normal(N)); x -= x.mean()
    return (x/(np.abs(x).max()+1e-9)).astype(np.float32)

def _voice_like_noise(N, sr, rng):
    t   = np.arange(N)/sr
    f0  = 150.0 + 100.0*np.sin(2*np.pi*3.0*t)
    ph  = np.cumsum(2*np.pi*f0/sr)
    voiced = sum(np.sin(k*ph)/k for k in range(1,6))
    voiced *= 0.5 + 0.5*np.sin(2*np.pi*4.0*t)
    wn = rng.standard_normal(N).astype(np.float32)
    un = _bandpass(wn.astype(np.float64), sr, min(2000.0,sr/2*0.4), min(8000.0,sr/2*0.9))
    mix = 0.70*voiced + 0.30*un; mix -= mix.mean()
    peak = float(np.abs(mix).max())
    return (mix/peak if peak>1e-9 else rng.standard_normal(N)).astype(np.float32)
