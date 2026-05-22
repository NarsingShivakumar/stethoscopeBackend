

# Reference: Grooby et al., "Noisy Neonatal Chest Sound Separation for



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
            "SeparationService v3 ready  target_sr=%d  fft=%d  hop=%d  "
            "K=[%d,%d,%d]=%d  iter=%d  beta=%d  sparsity=%.2f",
            config.TARGET_SR, config.FFTSIZE, config.HOPSIZE,
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
        """
        N = len(audio)

        # 1. Pre-process
        audio = _remove_dc(audio)
        audio = _normalise(audio)

        # 2. Resample to 4000 Hz
        x4k = _resample(audio, sr, self.cfg.TARGET_SR)
        N4k = len(x4k)
        log.debug("Resampled %d→%d samples (%d→%d Hz)", N, N4k, sr, self.cfg.TARGET_SR)

        # ── BUG 4 FIX: Pass-through for already-clean heart recordings ─────
        # If the input already has most energy in the heart band, skip NMF.
        # This handles "normal.mp3" style clean heart recordings correctly.
        heart_energy_frac = _band_energy_frac(x4k, self.cfg.TARGET_SR,
                                               20.0, 180.0)
        log.debug("Heart band energy fraction: %.3f", heart_energy_frac)

        if heart_energy_frac >= 0.60:
            log.info(
                "Input is already heart-dominant (%.1f%% in 20-180Hz) — "
                "skipping NMF, applying bandpass only.",
                heart_energy_frac * 100,
            )
            heart4k = _bandpass(x4k, self.cfg.TARGET_SR, 20.0, 180.0, order=6)
            # Lung output will be near-zero (correct — input has no lung content)
            lung4k = _bandpass(x4k, self.cfg.TARGET_SR, 60.0, 1800.0, order=4)
            lung4k = lung4k * (1.0 - heart_energy_frac)  # attenuate proportionally
        else:
            # ── Normal NMF separation path ─────────────────────────────────
            # 3. STFT — BUG 1 FIX: nperseg = nfft = FFTSIZE = 1024
            S = self._stft(x4k)
            V = np.maximum(np.abs(S), self.cfg.NMF_EPS)
            F, T = V.shape

            # 4. NMF
            W, H = self._nmf(V, F, T)

            # 5. Component assignment — BUG 2 FIX: by band-energy fraction
            h_idx, l_idx, n_idx = self._assign(W)
            log.debug("Assigned  heart=%s  lung=%s  noise=%s",
                      h_idx[:5], l_idx[:5], n_idx[:3])

            # 6. Wiener-mask reconstruction
            heart4k = self._reconstruct(S, W, H, h_idx, N4k)
            lung4k  = self._reconstruct(S, W, H, l_idx, N4k)
            noise4k = self._reconstruct(S, W, H, n_idx, N4k)

            # 7. Post-filter each channel — BUG 3 FIX
            heart4k = _bandpass(heart4k, self.cfg.TARGET_SR, 20.0, 180.0, order=6)
            lung4k  = _bandpass(lung4k,  self.cfg.TARGET_SR, 60.0, 1800.0, order=4)

        # 8. Resample back to original SR
        heart = _match(_resample(heart4k, self.cfg.TARGET_SR, sr), N)
        lung  = _match(_resample(lung4k,  self.cfg.TARGET_SR, sr), N)

        # 9. Normalise
        heart = _normalise(heart)
        lung  = _normalise(lung)

        # 10. Quality metrics
        noise_check = _resample(x4k - heart4k - lung4k
                                 if 'noise4k' not in dir()
                                 else noise4k,
                                 self.cfg.TARGET_SR, sr)
        noise_check = _match(noise_check.astype(np.float32), N)
        nl, sq = _quality(heart, lung, noise_check)

        return {
            "heart":          heart.astype(np.float32),
            "lung":           lung.astype(np.float32),
            "noise_level":    nl,
            "signal_quality": sq,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  STFT / ISTFT — BUG 1 FIXED: nperseg = nfft = FFTSIZE = 1024
    # ─────────────────────────────────────────────────────────────────────────

    def _stft(self, x: np.ndarray) -> np.ndarray:
        """
        STFT exactly matching egrooby MATLAB:
          spectrogram(x, hann(FFTSIZE), FFTSIZE-HOPSIZE, FFTSIZE, fs)
          → window length = FFTSIZE = 1024
          → nfft          = FFTSIZE = 1024
          → hop           = HOPSIZE = 256
          → F = 513 bins, freq resolution = 4000/1024 = 3.9 Hz/bin
        """
        win      = sps.get_window("hann", self.cfg.FFTSIZE, fftbins=True)
        noverlap = self.cfg.FFTSIZE - self.cfg.HOPSIZE
        _, _, S  = sps.stft(
            x,
            fs      = self.cfg.TARGET_SR,
            window  = win,
            nperseg = self.cfg.FFTSIZE,    # ← KEY FIX: was WINDOWSIZE=512
            noverlap= noverlap,
            nfft    = self.cfg.FFTSIZE,
            boundary= "zeros",
            padded  = True,
        )
        return S  # (513, T)

    def _istft(self, S: np.ndarray, length: int) -> np.ndarray:
        win      = sps.get_window("hann", self.cfg.FFTSIZE, fftbins=True)
        noverlap = self.cfg.FFTSIZE - self.cfg.HOPSIZE
        _, x = sps.istft(
            S,
            fs      = self.cfg.TARGET_SR,
            window  = win,
            nperseg = self.cfg.FFTSIZE,
            noverlap= noverlap,
            nfft    = self.cfg.FFTSIZE,
            boundary= True,
        )
        return x[:length].astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    #  NMF — beta=1 KL divergence (unchanged logic, same updates)
    # ─────────────────────────────────────────────────────────────────────────

    def _nmf(self, V: np.ndarray, F: int, T: int) -> Tuple[np.ndarray, np.ndarray]:
        K   = self.K
        eps = self.cfg.NMF_EPS
        lam = self.cfg.SPARSITY

        rng = np.random.RandomState(42)
        W = np.abs(rng.randn(F, K).astype(np.float32)) + eps
        H = np.abs(rng.randn(K, T).astype(np.float32)) + eps

        col_norms = W.sum(axis=0, keepdims=True) + eps
        W /= col_norms
        H *= col_norms.T

        ones_F = np.ones((F, 1), dtype=np.float32)
        ones_T = np.ones((1, T), dtype=np.float32)

        for i in range(self.cfg.MAXITER):
            WH    = np.maximum(W @ H, eps)
            num_H = W.T @ (V / WH)
            den_H = np.maximum(W.T @ ones_F + lam, eps)
            H     = np.maximum(H * (num_H / den_H), eps)

            WH    = np.maximum(W @ H, eps)
            num_W = (V / WH) @ H.T
            den_W = np.maximum(ones_T @ H.T, eps)
            W     = np.maximum(W * (num_W / den_W), eps)

            if (i + 1) % 10 == 0:
                col_norms = W.sum(axis=0, keepdims=True) + eps
                W /= col_norms
                H *= col_norms.T

        log.debug("NMF done  W=%s  H=%s", W.shape, H.shape)
        return W, H

    # ─────────────────────────────────────────────────────────────────────────
    #  BUG 2 FIX — Component assignment by band-energy fraction
    # ─────────────────────────────────────────────────────────────────────────

    def _assign(self, W: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
        """
        Assign K NMF basis vectors to heart / lung / noise groups.

        Method (egrooby Section II-C spectral heuristics, Python translation):

          heart_frac[k] = energy(W[:,k] in 20-180 Hz) / total_energy(W[:,k])
          lung_frac[k]  = energy(W[:,k] in 60-1800 Hz) / total_energy(W[:,k])

          Step 1: Rank by heart_frac ↓ → top K_HEART = heart group
          Step 2: From remaining, rank by lung_frac ↓ → top K_LUNG = lung group
          Step 3: Rest = noise

        This prevents low-frequency noise components (DC drift, motion artefacts)
        from contaminating the heart group, which was the primary cause of
        noise-sounding heart output.
        """
        cfg  = self.cfg
        F, K = W.shape
        nyq  = cfg.TARGET_SR / 2.0
        freqs = np.linspace(0, nyq, F)

        heart_mask = (freqs >= 20.0) & (freqs <= 180.0)
        lung_mask  = (freqs >= 60.0) & (freqs <= min(1800.0, nyq))

        heart_frac = np.array([
            W[:, k][heart_mask].sum() / (W[:, k].sum() + cfg.NMF_EPS)
            for k in range(K)
        ])
        lung_frac = np.array([
            W[:, k][lung_mask].sum() / (W[:, k].sum() + cfg.NMF_EPS)
            for k in range(K)
        ])

        # ── Heart: top K_HEART by heart_frac ─────────────────────────────────
        h_order = np.argsort(heart_frac)[::-1]
        h_idx   = h_order[:cfg.K_HEART].tolist()
        h_set   = set(h_idx)

        # ── Lung: top K_LUNG by lung_frac from remaining ─────────────────────
        remaining = [k for k in range(K) if k not in h_set]
        remaining.sort(key=lambda k: lung_frac[k], reverse=True)
        l_idx = remaining[:cfg.K_LUNG]
        l_set = set(l_idx)

        # ── Noise: everything else ────────────────────────────────────────────
        n_idx = [k for k in range(K) if k not in h_set and k not in l_set]

        log.debug(
            "_assign  top heart_frac=%s  top lung_frac=%s",
            [round(heart_frac[k], 3) for k in h_idx[:5]],
            [round(lung_frac[k],  3) for k in l_idx[:5]],
        )
        return h_idx, l_idx, n_idx

    # ─────────────────────────────────────────────────────────────────────────
    #  Wiener reconstruction (unchanged — was correct)
    # ─────────────────────────────────────────────────────────────────────────

    def _reconstruct(
        self,
        S:       np.ndarray,
        W:       np.ndarray,
        H:       np.ndarray,
        indices: List[int],
        length:  int,
    ) -> np.ndarray:
        eps      = self.cfg.NMF_EPS
        WH_total = np.maximum(W @ H, eps)
        WH_comp  = np.zeros_like(WH_total)
        for k in indices:
            WH_comp += np.outer(W[:, k], H[k, :])
        mask   = np.maximum(WH_comp, eps) / WH_total
        S_comp = mask * S
        return self._istft(S_comp, length)


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _remove_dc(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()).astype(np.float32)


def _normalise(x: np.ndarray, headroom: float = 0.95) -> np.ndarray:
    p = float(np.abs(x).max())
    return (x * (headroom / p)).astype(np.float32) if p >= 1e-9 else x.astype(np.float32)


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x.astype(np.float32)
    n_out = int(round(len(x) * dst / src))
    return sps.resample_poly(x.astype(np.float64), dst, src)[:n_out].astype(np.float32)


def _match(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - len(x), dtype=np.float32)])


def _bandpass(x: np.ndarray, sr: int, low_hz: float, high_hz: float,
              order: int = 6) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq  = sr / 2.0
    low  = max(low_hz,  1.0) / nyq
    high = min(high_hz, nyq * 0.98) / nyq
    if low >= high:
        return x.copy()
    try:
        sos = sps.butter(order, [low, high], btype="bandpass", output="sos")
        return sps.sosfiltfilt(sos, x.astype(np.float64)).astype(np.float32)
    except Exception as e:
        log.warning("_bandpass failed (%s) — returning unfiltered", e)
        return x.copy()


def _band_energy_frac(x: np.ndarray, sr: int,
                       low_hz: float, high_hz: float) -> float:
    """
    Fraction of signal power that lies inside [low_hz, high_hz].
    Used to detect already-clean heart recordings (pass-through logic).
    """
    n    = len(x)
    if n < 16:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0/sr)
    mag2  = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    in_band = (freqs >= low_hz) & (freqs <= high_hz)
    total   = mag2.sum()
    if total < 1e-20:
        return 0.0
    return float(mag2[in_band].sum() / total)


def _quality(heart: np.ndarray, lung: np.ndarray,
             noise: np.ndarray) -> Tuple[float, float]:
    eps = 1e-9
    rh  = float(np.sqrt(np.mean(heart ** 2)) + eps)
    rl  = float(np.sqrt(np.mean(lung  ** 2)) + eps)
    rn  = float(np.sqrt(np.mean(noise ** 2)) + eps)
    nl  = rn / (rh + rl + rn)
    sq  = 1.0 - nl
    return round(nl, 4), round(sq, 4)

