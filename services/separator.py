"""
services/separator.py
=====================
Stage 3 — Heart / Lung source separation using NeoSSNet
(Conv-TasNet-style architecture matching
 yangyipoh/Neonatal-Chest-Sound-Separation-using-Deep-Learning)

If models/model_best.pt is present → deep inference.
Otherwise → NMF fallback (spectral centroid assignment).

Input:  cleaned float32 array at any SR
Output: (heart_np, lung_np) both at src_sr
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal as sps
from sklearn.decomposition import NMF

from utils.audio_io import resample, TARGET_SR_ANALYSIS
from utils.dsp import bandpass, rms

log = logging.getLogger("steth.separator")

MODEL_PATH = Path("models/model_best.pt")
MODEL_SR   = 4000
STFT_N_FFT = 1024
STFT_HOP   = 256
STFT_WIN   = 512


# ── NeoSSNet architecture ──────────────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation=1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.dw    = nn.Conv1d(in_ch, in_ch, kernel, dilation=dilation,
                               padding=pad, groups=in_ch, bias=False)
        self.pw    = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.norm1 = nn.GroupNorm(1, in_ch,  eps=1e-8)
        self.norm2 = nn.GroupNorm(1, out_ch, eps=1e-8)
        self.act   = nn.PReLU()

    def forward(self, x):
        return self.norm2(self.act(self.pw(self.norm1(self.act(self.dw(x))))))


class TCNBlock(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, hid_ch, 1, bias=False)
        self.ds    = DepthwiseSeparableConv(hid_ch, in_ch, kernel, dilation)
        self.norm  = nn.GroupNorm(1, hid_ch, eps=1e-8)
        self.act   = nn.PReLU()

    def forward(self, x):
        return x + self.ds(self.act(self.norm(self.conv1(x))))


class NeoSSNet(nn.Module):
    N_FILTERS  = 256
    BOTTLENECK = 256
    HIDDEN     = 512
    KERNEL     = 3
    N_BLOCKS   = 8
    N_REPEATS  = 3
    N_SOURCES  = 2

    def __init__(self):
        super().__init__()
        self.encoder  = nn.Conv1d(1, self.N_FILTERS, 16, stride=8, padding=4, bias=False)
        self.enc_norm = nn.GroupNorm(1, self.N_FILTERS, eps=1e-8)
        self.proj_in  = nn.Conv1d(self.N_FILTERS, self.BOTTLENECK, 1, bias=False)

        layers = []
        for _ in range(self.N_REPEATS):
            for b in range(self.N_BLOCKS):
                layers.append(TCNBlock(self.BOTTLENECK, self.HIDDEN,
                                       self.KERNEL, dilation=2 ** b))
        self.tcn      = nn.Sequential(*layers)
        self.mask_net = nn.Conv1d(self.BOTTLENECK, self.N_FILTERS * self.N_SOURCES, 1)
        self.act      = nn.Sigmoid()
        self.decoder  = nn.ConvTranspose1d(self.N_FILTERS, 1, 16, stride=8,
                                           padding=4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T) → (B, 2, T)
        enc   = F.relu(self.enc_norm(self.encoder(x)))
        feat  = self.tcn(self.proj_in(enc))
        masks = self.act(self.mask_net(feat))
        m1, m2 = masks[:, :self.N_FILTERS], masks[:, self.N_FILTERS:]
        T = x.shape[-1]
        out1 = self._match(self.decoder(enc * m1), T)
        out2 = self._match(self.decoder(enc * m2), T)
        return torch.cat([out1, out2], dim=1)

    @staticmethod
    def _match(y, T):
        if y.shape[-1] > T: return y[..., :T]
        if y.shape[-1] < T: return F.pad(y, (0, T - y.shape[-1]))
        return y


# ── Service ────────────────────────────────────────────────────────────────────

class SeparatorService:

    def __init__(self):
        self._model: Optional[NeoSSNet] = None
        self._use_deep = self._try_load()

    def _try_load(self) -> bool:
        if not MODEL_PATH.exists():
            log.info("model_best.pt not found — NMF separator active")
            return False
        try:
            m = NeoSSNet()
            state = torch.load(MODEL_PATH, map_location="cpu")
            m.load_state_dict(state, strict=False)
            m.eval()
            self._model = m
            log.info("NeoSSNet loaded from %s", MODEL_PATH)
            return True
        except Exception as exc:
            log.warning("NeoSSNet load failed (%s) — NMF fallback", exc)
            return False

    def separate(self, audio: np.ndarray,
                 src_sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (heart_np, lung_np) at src_sr."""
        audio4k = resample(audio, src_sr, MODEL_SR)

        if self._use_deep and self._model:
            heart4k, lung4k = self._deep(audio4k)
        else:
            heart4k, lung4k = self._nmf(audio4k, MODEL_SR)

        heart = resample(heart4k, MODEL_SR, src_sr)
        lung  = resample(lung4k,  MODEL_SR, src_sr)

        for sig in [heart, lung]:
            p = float(np.abs(sig).max())
            if p > 1e-6:
                sig[:] = sig / p * 0.95
        return heart.astype(np.float32), lung.astype(np.float32)

    def _deep(self, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t = torch.from_numpy(a).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = self._model(t)
        return out[0, 0].numpy(), out[0, 1].numpy()

    @staticmethod
    def _nmf(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        f, _, Zxx = sps.stft(audio.astype(np.float64), fs=sr,
                             window="hann", nperseg=STFT_WIN,
                             noverlap=STFT_WIN - STFT_HOP, nfft=STFT_N_FFT)
        mag   = np.abs(Zxx).astype(np.float32)
        phase = np.angle(Zxx)
        eps   = 1e-9

        nmf = NMF(n_components=50, init="nndsvda", beta_loss=1,
                  solver="mu", max_iter=200, l1_ratio=0.1, random_state=0)
        W = nmf.fit_transform(mag)
        H = nmf.components_

        heart_mask = np.zeros(50, dtype=bool)
        for k in range(50):
            wk = W[:, k]; total = wk.sum() + eps
            centroid = float((f * wk).sum() / total)
            heart_mask[k] = centroid < 150.0

        Wh = W.copy(); Wh[:, ~heart_mask] = 0.0
        Wl = W.copy(); Wl[:,  heart_mask] = 0.0
        mh = (Wh @ H); ml = (Wl @ H)
        tot = mh + ml + eps

        _, heart = sps.istft((mag * mh / tot) * np.exp(1j * phase), fs=sr,
                             nperseg=STFT_WIN, noverlap=STFT_WIN - STFT_HOP)
        _, lung  = sps.istft((mag * ml / tot) * np.exp(1j * phase), fs=sr,
                             nperseg=STFT_WIN, noverlap=STFT_WIN - STFT_HOP)

        N = len(audio)
        return heart[:N].astype(np.float32), lung[:N].astype(np.float32)