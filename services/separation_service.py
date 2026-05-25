"""
services/separation_service.py
NMF-based heart/lung separator – v4 (full-length output guarantee)

Key changes vs v3:
  - `separate()` now stores N = len(audio) BEFORE any processing and
    uses matchresample() on the final heart/lung arrays so the returned
    numpy arrays are ALWAYS exactly N samples (same as the input).
  - `matchresample()` helper replaces the old two-step
    resample() + match() so the output length is always N, never N±1.
"""
import logging
from typing import Dict, List, Tuple

import numpy as np
import scipy.signal as sps

log = logging.getLogger("steth.separation")


class SeparationService:
    """Thread-safe stateless NMF separation service. Instantiate once."""

    def __init__(self, config):
        self.cfg = config
        self.K = config.K_HEART + config.K_LUNG + config.K_NOISE + 50
        log.info(
            "SeparationService v4 ready  target=%dr  fft=%d  hop=%d  "
            "K=%d(%d,%d,%d)  iter=%d  beta=%d  sparsity=%.2f",
            config.TARGET_SR, config.FFTSIZE, config.HOPSIZE,
            config.K_HEART, config.K_LUNG, config.K_NOISE, self.K,
            config.MAXITER, config.BETA_LOSS, config.SPARSITY,
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def separate(self, audio: np.ndarray, sr: int) -> Dict:
        """Separate mono stethoscope audio into heart, lung, noise.

        Parameters
        ----------
        audio : np.ndarray  float32  shape (N,)  values in [-1, 1]
        sr    : int  original sample rate

        Returns
        -------
        dict with keys:
            heart         np.ndarray float32 (N,)   ← ALWAYS same length as input
            lung          np.ndarray float32 (N,)   ← ALWAYS same length as input
            noise_level   float
            signal_quality float
        """
        # ── 0. Remember original length (used for final length guarantee) ──
        N = len(audio)

        audio = _removedc(audio)
        audio = _normalise(audio)

        # ── 1. Resample to 4 kHz ──
        x4k = _resample(audio, sr, self.cfg.TARGET_SR)
        N4k = len(x4k)

        # ── 2. Fast-path: already heart-dominant ──
        heart_energy_frac = _band_energy_frac(x4k, self.cfg.TARGET_SR, 20.0, 180.0)
        if heart_energy_frac > 0.60:
            log.info(
                "Input is already heart-dominant (%.1f%% in 20-180 Hz) "
                "– skipping NMF, applying bandpass only.",
                heart_energy_frac * 100,
            )
            heart4k = _bandpass(x4k, self.cfg.TARGET_SR, 20.0, 180.0, order=6)
            lung4k  = _bandpass(x4k, self.cfg.TARGET_SR, 60.0, 1800.0, order=4)
            lung4k  = lung4k * (1.0 - heart_energy_frac)
        else:
            # ── 3. STFT ──
            S = self._stft(x4k)
            V = np.maximum(np.abs(S), self.cfg.NMF_EPS)
            F, T = V.shape

            # ── 4. NMF ──
            W, H = self._nmf(V, F, T)

            # ── 5. Component assignment ──
            hidx, lidx, nidx = self._assign(W)

            # ── 6. Wiener-mask reconstruction ──
            heart4k = self._reconstruct(S, W, H, hidx, N4k)
            lung4k  = self._reconstruct(S, W, H, lidx, N4k)
            noise4k = self._reconstruct(S, W, H, nidx, N4k)

            # ── 7. Post-filter ──
            heart4k = _bandpass(heart4k, self.cfg.TARGET_SR, 20.0, 180.0, order=6)
            lung4k  = _bandpass(lung4k,  self.cfg.TARGET_SR, 60.0, 1800.0, order=4)

        # ── 8. Resample back to original SR and GUARANTEE original length ──
        heart = _matchresample(heart4k, self.cfg.TARGET_SR, sr, N)
        lung  = _matchresample(lung4k,  self.cfg.TARGET_SR, sr, N)

        # ── 9. Normalise ──
        heart = _normalise(heart)
        lung  = _normalise(lung)

        # ── 10. Quality metrics ──
        noise_ref = _matchresample(
            x4k - heart4k - lung4k if heart_energy_frac <= 0.60
            else x4k - heart4k - lung4k,
            self.cfg.TARGET_SR, sr, N
        ).astype(np.float32)
        nl, sq = _quality(heart, lung, noise_ref)

        return {
            "heart":          heart.astype(np.float32),
            "lung":           lung.astype(np.float32),
            "noise_level":    nl,
            "signal_quality": sq,
        }

    # ------------------------------------------------------------------ #
    #  STFT / iSTFT
    # ------------------------------------------------------------------ #
    def _stft(self, x: np.ndarray) -> np.ndarray:
        win     = sps.get_window("hann", self.cfg.FFTSIZE, fftbins=True)
        noverlap = self.cfg.FFTSIZE - self.cfg.HOPSIZE
        _, _, S = sps.stft(
            x, fs=self.cfg.TARGET_SR, window=win,
            nperseg=self.cfg.FFTSIZE, noverlap=noverlap,
            nfft=self.cfg.FFTSIZE, boundary="zeros", padded=True,
        )
        return S[: self.cfg.FFTSIZE // 2 + 1, :]

    def _istft(self, S: np.ndarray, length: int) -> np.ndarray:
        win     = sps.get_window("hann", self.cfg.FFTSIZE, fftbins=True)
        noverlap = self.cfg.FFTSIZE - self.cfg.HOPSIZE
        _, x = sps.istft(
            S, fs=self.cfg.TARGET_SR, window=win,
            nperseg=self.cfg.FFTSIZE, noverlap=noverlap,
            nfft=self.cfg.FFTSIZE, boundary=True,
        )
        return x[:length].astype(np.float32)

    # ------------------------------------------------------------------ #
    #  NMF
    # ------------------------------------------------------------------ #
    def _nmf(self, V: np.ndarray, F: int, T: int) -> Tuple[np.ndarray, np.ndarray]:
        K   = self.K
        eps = self.cfg.NMF_EPS
        lam = self.cfg.SPARSITY
        rng = np.random.RandomState(42)
        W   = np.abs(rng.randn(F, K)).astype(np.float32) + eps
        H   = np.abs(rng.randn(K, T)).astype(np.float32) + eps
        col_norms = W.sum(axis=0, keepdims=True) + eps
        W /= col_norms; H *= col_norms.T
        onesF = np.ones((F, 1), dtype=np.float32)
        onesT = np.ones((1, T), dtype=np.float32)
        for i in range(self.cfg.MAXITER):
            WH    = np.maximum(W @ H, eps)
            num_H = W.T @ (V / WH)
            den_H = np.maximum(W.T @ onesF + lam, eps)
            H     = np.maximum(H * num_H / den_H, eps)
            WH    = np.maximum(W @ H, eps)
            num_W = (V / WH) @ H.T
            den_W = np.maximum(onesT @ H.T, eps)
            W     = np.maximum(W * num_W / den_W, eps)
            if (i + 1) % 10 == 0:
                col_norms = W.sum(axis=0, keepdims=True) + eps
                W /= col_norms; H *= col_norms.T
        return W, H

    # ------------------------------------------------------------------ #
    #  Component assignment
    # ------------------------------------------------------------------ #
    def _assign(self, W: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
        cfg   = self.cfg
        F, K  = W.shape
        nyq   = cfg.TARGET_SR / 2.0
        freqs = np.linspace(0, nyq, F)
        heart_mask = (freqs >= 20.0)  & (freqs <= 180.0)
        lung_mask  = (freqs >= 60.0)  & (freqs <= min(1800.0, nyq))
        heart_frac = np.array([W[heart_mask, k].sum() / (W[:, k].sum() + cfg.NMF_EPS) for k in range(K)])
        lung_frac  = np.array([W[lung_mask,  k].sum() / (W[:, k].sum() + cfg.NMF_EPS) for k in range(K)])
        h_order = np.argsort(-heart_frac)
        hidx    = h_order[: cfg.K_HEART].tolist()
        hset    = set(hidx)
        remaining = sorted([k for k in range(K) if k not in hset], key=lambda k: -lung_frac[k])
        lidx  = remaining[: cfg.K_LUNG]
        lset  = set(lidx)
        nidx  = [k for k in range(K) if k not in hset and k not in lset]
        return hidx, lidx, nidx

    def _reconstruct(self, S: np.ndarray, W: np.ndarray, H: np.ndarray,
                     indices: List[int], length: int) -> np.ndarray:
        eps     = self.cfg.NMF_EPS
        WH_tot  = np.maximum(W @ H, eps)
        WH_comp = np.zeros_like(WH_tot)
        for k in indices:
            WH_comp += np.outer(W[:, k], H[k, :])
        mask  = np.maximum(WH_comp, eps) / WH_tot
        S_comp = mask * S
        return self._istft(S_comp, length)


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _removedc(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()).astype(np.float32)

def _normalise(x: np.ndarray, headroom: float = 0.95) -> np.ndarray:
    p = float(np.abs(x).max())
    return (x * headroom / p).astype(np.float32) if p > 1e-9 else x.astype(np.float32)

def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x.astype(np.float32)
    n_out = int(round(len(x) * dst / src))
    return sps.resample_poly(x.astype(np.float64), dst, src)[:n_out].astype(np.float32)

def _matchresample(x: np.ndarray, src: int, dst: int, n: int) -> np.ndarray:
    """Resample from src → dst, then hard-clip / zero-pad to exactly n samples."""
    resampled = _resample(x, src, dst)
    if len(resampled) >= n:
        return resampled[:n].astype(np.float32)
    return np.concatenate([resampled, np.zeros(n - len(resampled), dtype=np.float32)])

def _bandpass(x: np.ndarray, sr: int, lo_hz: float, hi_hz: float, order: int = 6) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq  = sr / 2.0
    low  = max(lo_hz, 1.0) / nyq
    high = min(hi_hz, nyq * 0.98) / nyq
    if low >= high:
        return x.copy()
    try:
        sos = sps.butter(order, [low, high], btype="bandpass", output="sos")
        return sps.sosfiltfilt(sos, x.astype(np.float64)).astype(np.float32)
    except Exception as e:
        log.warning("bandpass failed %s – returning unfiltered", e)
        return x.copy()

def _band_energy_frac(x: np.ndarray, sr: int, lo_hz: float, hi_hz: float) -> float:
    n = len(x)
    if n < 16:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mag2  = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    inband = (freqs >= lo_hz) & (freqs <= hi_hz)
    total  = mag2.sum()
    return float(mag2[inband].sum() / total) if total > 1e-20 else 0.0

def _quality(heart: np.ndarray, lung: np.ndarray, noise: np.ndarray) -> Tuple[float, float]:
    eps = 1e-9
    rh  = float(np.sqrt(np.mean(heart ** 2))) + eps
    rl  = float(np.sqrt(np.mean(lung  ** 2))) + eps
    rn  = float(np.sqrt(np.mean(noise ** 2))) + eps
    nl  = rn / (rh + rl + rn)
    sq  = 1.0 - nl
    return round(nl, 4), round(sq, 4)