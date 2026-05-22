"""
services/separation_service.py
================================
Python port of the egrooby Monash NMF heart/lung separation algorithm.

Reference: Grooby et al., "Noisy Neonatal Chest Sound Separation for
           High-Quality Heart and Lung Sounds", IEEE TBME 2022.
           https://github.com/egrooby-monash/Heart-and-Lung-Sound-Separation

MATLAB source (example_code.m) key parameters:
  options_tf.FFTSIZE   = 1024
  options_tf.HOPSIZE   = 256
  options_tf.WINDOWSIZE = 512
  options_nmf.beta_loss = 1   (KL divergence)
  options_nmf.sparsity  = 0.1
  MAXITER = 100
  K = [20 10 20 20 20 20]     → we use K_heart=20, K_lung=10, K_noise=20

Pipeline (matches nmcf_overall2 / nmf_cluster1 logic):
  1. Resample mixed signal to fs=4000 Hz
  2. STFT → magnitude spectrogram V (F×T)
  3. NMF: V ≈ W·H   (W: F×K, H: K×T, beta=1 KL divergence)
  4. Assign component groups to heart/lung/noise via spectral heuristics
  5. Wiener-mask reconstruction per group → masked complex STFT
  6. ISTFT → time domain
  7. Resample back to original SR, normalise
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import scipy.signal as sps

log = logging.getLogger("steth.separation")


class SeparationService:
    """Thread-safe stateless NMF separation service. Instantiate once."""

    def __init__(self, config):
        self.cfg = config
        self.K = config.K_HEART + config.K_LUNG + config.K_NOISE  # = 50
        log.info(
            "SeparationService ready  target_sr=%d  fft=%d  hop=%d  win=%d  "
            "K=[%d,%d,%d]=%d  iter=%d  beta=%d  sparsity=%.2f",
            config.TARGET_SR, config.FFTSIZE, config.HOPSIZE, config.WINDOWSIZE,
            config.K_HEART, config.K_LUNG, config.K_NOISE, self.K,
            config.MAXITER, config.BETA_LOSS, config.SPARSITY,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def separate(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Separate mono stethoscope audio into heart, lung, noise.

        Parameters
        ----------
        audio : np.ndarray  float32 shape (N,)  values in [-1, 1]
        sr    : int         original sample rate (e.g. 44100)

        Returns
        -------
        dict:
          heart          : np.ndarray float32 (N,)
          lung           : np.ndarray float32 (N,)
          noise_level    : float  [0, 1]
          signal_quality : float  [0, 1]
        """
        N = len(audio)

        # 1. Pre-process
        audio = _remove_dc(audio)
        audio = _normalise(audio)

        # 2. Resample to 4000 Hz (egrooby mandatory)
        x4k = _resample(audio, sr, self.cfg.TARGET_SR)
        N4k = len(x4k)
        log.debug("Resampled %d→%d samples (%d→%d Hz)", N, N4k, sr, self.cfg.TARGET_SR)

        # 3. STFT (egrooby: FFTSIZE=1024, HOPSIZE=256, WINDOWSIZE=512)
        S = self._stft(x4k)          # complex (F, T)
        V = np.abs(S)                 # magnitude  (F, T)
        V = np.maximum(V, self.cfg.NMF_EPS)
        F, T = V.shape

        # 4. NMF → W (F×K), H (K×T)
        W, H = self._nmf(V, F, T)

        # 5. Assign component indices to heart / lung / noise
        h_idx, l_idx, n_idx = self._assign(W)
        log.debug("Assigned  heart=%s  lung=%s  noise=%s", h_idx[:3], l_idx[:3], n_idx[:3])

        # 6. Wiener-mask reconstruction per group
        heart4k = self._reconstruct(S, W, H, h_idx, N4k)
        lung4k  = self._reconstruct(S, W, H, l_idx,  N4k)
        noise4k = self._reconstruct(S, W, H, n_idx,  N4k)

        # 7. Resample back to original SR
        heart = _match(_resample(heart4k, self.cfg.TARGET_SR, sr), N)
        lung  = _match(_resample(lung4k,  self.cfg.TARGET_SR, sr), N)
        noise = _match(_resample(noise4k, self.cfg.TARGET_SR, sr), N)

        # 8. Normalise (egrooby: heart / max(abs(heart)))
        heart = _normalise(heart)
        lung  = _normalise(lung)

        # 9. Quality metrics
        nl, sq = _quality(heart, lung, noise)

        return {
            "heart":          heart.astype(np.float32),
            "lung":           lung.astype(np.float32),
            "noise_level":    nl,
            "signal_quality": sq,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  STFT / ISTFT
    # ─────────────────────────────────────────────────────────────────────────

    def _stft(self, x: np.ndarray) -> np.ndarray:
        """
        STFT matching egrooby MATLAB stft() call.
        Returns complex array shape (F, T) where F = FFTSIZE//2 + 1.
        """
        win = sps.get_window("hann", self.cfg.WINDOWSIZE, fftbins=True)
        noverlap = self.cfg.WINDOWSIZE - self.cfg.HOPSIZE
        _, _, S = sps.stft(
            x,
            fs=self.cfg.TARGET_SR,
            window=win,
            nperseg=self.cfg.WINDOWSIZE,
            noverlap=noverlap,
            nfft=self.cfg.FFTSIZE,
            boundary="zeros",
            padded=True,
        )
        return S  # (F, T)

    def _istft(self, S: np.ndarray, length: int) -> np.ndarray:
        win = sps.get_window("hann", self.cfg.WINDOWSIZE, fftbins=True)
        noverlap = self.cfg.WINDOWSIZE - self.cfg.HOPSIZE
        _, x = sps.istft(
            S,
            fs=self.cfg.TARGET_SR,
            window=win,
            nperseg=self.cfg.WINDOWSIZE,
            noverlap=noverlap,
            nfft=self.cfg.FFTSIZE,
            boundary=True,
        )
        return x[:length].astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    #  NMF — beta=1 (KL divergence / IS) multiplicative updates
    # ─────────────────────────────────────────────────────────────────────────

    def _nmf(self, V: np.ndarray, F: int, T: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        NMF with Kullback-Leibler divergence (beta=1) and L1 sparsity on H.

        Multiplicative update rules (Lee & Seung 2001):
          H ← H * (W^T (V/WH)) / (W^T 1_F + λ)
          W ← W * ((V/WH) H^T)  / (1_T H^T)

        All shapes explicitly:
          V  : (F, T)
          W  : (F, K)
          H  : (K, T)
          W^T (V/WH)  →  (K,F)·(F,T)  = (K,T)   ✓
          W^T 1_F     →  (K,F)·(F,1)  = (K,1)   ✓
          (V/WH) H^T  →  (F,T)·(T,K)  = (F,K)   ✓
          1_T H^T     →  (1,T)·(T,K)  = (1,K)   broadcast to (F,K) ✓
        """
        K   = self.K
        eps = self.cfg.NMF_EPS
        lam = self.cfg.SPARSITY

        rng = np.random.RandomState(42)
        W = np.abs(rng.randn(F, K).astype(np.float32)) + eps
        H = np.abs(rng.randn(K, T).astype(np.float32)) + eps

        # Column-normalise W; absorb scale into H
        col_norms = W.sum(axis=0, keepdims=True) + eps   # (1, K)
        W /= col_norms
        H *= col_norms.T                                  # (K, 1) → broadcasts

        # Pre-compute constant vectors for KL update denominators
        ones_F = np.ones((F, 1), dtype=np.float32)   # (F, 1)  for H update denom
        ones_T = np.ones((1, T), dtype=np.float32)   # (1, T)  for W update denom

        for i in range(self.cfg.MAXITER):
            # ── H update ──────────────────────────────────────────────────────
            WH    = np.maximum(W @ H, eps)             # (F, T)
            # W^T · (V / WH) → (K, F) · (F, T) = (K, T)
            num_H = W.T @ (V / WH)                     # (K, T)
            # W^T · ones_F → (K, F) · (F, 1) = (K, 1)
            den_H = np.maximum(W.T @ ones_F + lam, eps)  # (K, 1)
            H     = np.maximum(H * (num_H / den_H), eps)

            # ── W update ──────────────────────────────────────────────────────
            WH    = np.maximum(W @ H, eps)             # (F, T)
            # (V / WH) · H^T → (F, T) · (T, K) = (F, K)
            num_W = (V / WH) @ H.T                     # (F, K)
            # ones_T · H^T → (1, T) · (T, K) = (1, K)  broadcasts to (F, K)
            den_W = np.maximum(ones_T @ H.T, eps)      # (1, K) → (F, K) via broadcast
            W     = np.maximum(W * (num_W / den_W), eps)

            # Re-normalise columns every 10 iters to prevent scale drift
            if (i + 1) % 10 == 0:
                col_norms = W.sum(axis=0, keepdims=True) + eps
                W /= col_norms
                H *= col_norms.T

        log.debug("NMF done after %d iters  W=%s  H=%s", self.cfg.MAXITER, W.shape, H.shape)
        return W, H

    # ─────────────────────────────────────────────────────────────────────────
    #  Component assignment heuristics
    # ─────────────────────────────────────────────────────────────────────────

    def _assign(self, W: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
        """
        Assign K NMF columns to heart / lung / noise groups.

        egrooby uses K=[20,10,20,...] pre-assigned groups.
        We replicate that structure by sorting all K columns by spectral
        centroid, then assigning the lowest-centroid K_HEART to heart,
        the next highest-broadband K_LUNG to lung, the rest to noise.

        Spectral centroid: heart sounds peak at 50–100 Hz (low centroid)
        Broadband energy:  lung sounds are wideband (high energy > 80 Hz)
        """
        cfg  = self.cfg
        F, K = W.shape
        freqs = np.linspace(0, cfg.TARGET_SR / 2.0, F)  # Hz per bin

        # Compute centroid per column
        cents = np.zeros(K, dtype=np.float64)
        bb    = np.zeros(K, dtype=np.float64)
        bb_mask = freqs >= cfg.LUNG_MIN_BROADBAND_HZ

        for k in range(K):
            col   = W[:, k]
            s     = col.sum() + cfg.NMF_EPS
            cents[k] = (freqs * col).sum() / s
            bb[k]    = col[bb_mask].sum()

        # Sort ascending by centroid → lowest = heart-like
        order = np.argsort(cents)
        h_idx = order[:cfg.K_HEART].tolist()

        # From remaining columns, sort by broadband energy descending → lung
        rest  = order[cfg.K_HEART:].tolist()
        bb_rest = bb[rest]
        bb_order = np.argsort(bb_rest)[::-1]
        l_idx = [rest[i] for i in bb_order[:cfg.K_LUNG]]
        n_idx = [rest[i] for i in bb_order[cfg.K_LUNG:]]

        return h_idx, l_idx, n_idx

    # ─────────────────────────────────────────────────────────────────────────
    #  Wiener soft-mask reconstruction (egrooby: reconstruction="Filtering")
    # ─────────────────────────────────────────────────────────────────────────

    def _reconstruct(
        self,
        S:       np.ndarray,   # (F, T) complex STFT
        W:       np.ndarray,   # (F, K)
        H:       np.ndarray,   # (K, T)
        indices: List[int],
        length:  int,
    ) -> np.ndarray:
        """
        Wiener mask for a subset of NMF components.

          mask = Σ_{k∈indices} W[:,k]⊗H[k,:] / Σ_k W[:,k]⊗H[k,:]
          S_comp = mask * S
          x = ISTFT(S_comp)
        """
        eps = self.cfg.NMF_EPS
        WH_total = np.maximum(W @ H, eps)                  # (F, T)

        WH_comp = np.zeros_like(WH_total)
        for k in indices:
            WH_comp += np.outer(W[:, k], H[k, :])

        mask   = np.maximum(WH_comp, eps) / WH_total       # (F, T) Wiener ratio
        S_comp = mask * S                                   # (F, T) complex
        return self._istft(S_comp, length)


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _remove_dc(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()).astype(np.float32)


def _normalise(x: np.ndarray, headroom: float = 0.95) -> np.ndarray:
    p = float(np.abs(x).max())
    if p < 1e-9:
        return x.astype(np.float32)
    return (x * (headroom / p)).astype(np.float32)


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x.astype(np.float32)
    n_out = int(round(len(x) * dst / src))
    y = sps.resample_poly(x.astype(np.float64), dst, src)
    return y[:n_out].astype(np.float32)


def _match(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - len(x), dtype=np.float32)])


def _quality(heart: np.ndarray, lung: np.ndarray,
             noise: np.ndarray) -> Tuple[float, float]:
    eps = 1e-9
    rh = float(np.sqrt(np.mean(heart ** 2)) + eps)
    rl = float(np.sqrt(np.mean(lung  ** 2)) + eps)
    rn = float(np.sqrt(np.mean(noise ** 2)) + eps)
    nl = rn / (rh + rl + rn)
    sq = 1.0 - nl
    return round(nl, 4), round(sq, 4)
