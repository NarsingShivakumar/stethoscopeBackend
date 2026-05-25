"""
services/neoSSNet.py
NeoSSNet Conv-TasNet-style two-source heart/lung separator – v2
Full-length output guarantee: returned arrays are ALWAYS len(clean_audio).
"""
from __future__ import annotations

import logging
import math
import os
from typing import Tuple

import numpy as np

log = logging.getLogger("steth.neoSSNet")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    log.warning("PyTorch not available – NeoSSNet will fall back to NMF.")

if TORCH_AVAILABLE:
    class GlobalLayerNorm(nn.Module):
        def __init__(self, channel_size: int):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(1, channel_size, 1))
            self.beta  = nn.Parameter(torch.zeros(1, channel_size, 1))
            self.eps   = 1e-8

        def forward(self, x):
            mean = x.mean(dim=(1, 2), keepdim=True)
            var  = (x - mean).pow(2).mean(dim=(1, 2), keepdim=True)
            return self.gamma * (x - mean) / (var + self.eps).sqrt() + self.beta

    class DepthwiseSeparable(nn.Module):
        def __init__(self, in_ch: int, skip_ch: int, kernel: int, dilation: int):
            super().__init__()
            pad = (kernel - 1) * dilation // 2
            self.conv1x1  = nn.Conv1d(in_ch, in_ch, 1)
            self.relu1    = nn.PReLU()
            self.norm1    = GlobalLayerNorm(in_ch)
            self.dwconv   = nn.Conv1d(in_ch, in_ch, kernel, groups=in_ch,
                                      dilation=dilation, padding=pad)
            self.relu2    = nn.PReLU()
            self.norm2    = GlobalLayerNorm(in_ch)
            self.res_out  = nn.Conv1d(in_ch, in_ch, 1)
            self.skip_out = nn.Conv1d(in_ch, skip_ch, 1)

        def forward(self, x):
            h = self.norm1(self.relu1(self.conv1x1(x)))
            h = self.norm2(self.relu2(self.dwconv(h)))
            return x + self.res_out(h), self.skip_out(h)

    class NeoSSNetModel(nn.Module):
        """Two-source (heart/lung) Conv-TasNet separator."""

        def __init__(self, cfg):
            super().__init__()
            N, L, H = cfg.N_FILTERS, cfg.FILTER_LEN, cfg.TCN_CHANNELS
            Sc, P    = cfg.TCN_SKIP, cfg.TCN_KERNEL
            R, B_    = cfg.N_REPEATS, cfg.N_BLOCKS
            self.encoder     = nn.Conv1d(1, N, L, stride=L // 2, padding=0, bias=False)
            self.encoder_act = nn.ReLU()
            self.bottleneck  = nn.Sequential(nn.LayerNorm(N), nn.Conv1d(N, H, 1))
            self.tcn_blocks  = nn.ModuleList([
                DepthwiseSeparable(H, Sc, P, 2 ** b)
                for r in range(R) for b in range(B_)
            ])
            self.masknet = nn.Sequential(nn.PReLU(), nn.Conv1d(Sc, N * 2, 1))
            self.decoder = nn.ConvTranspose1d(N, 1, L, stride=L // 2, padding=0, bias=False)

        def forward(self, mixture: torch.Tensor) -> torch.Tensor:
            B, _, T = mixture.shape
            enc  = self.encoder_act(self.encoder(mixture))          # (B, N, F)
            h    = self.bottleneck(enc)
            skip_sum = 0.0
            for block in self.tcn_blocks:
                h, skip = block(h)
                skip_sum = skip_sum + skip
            masks = self.masknet(skip_sum).reshape(B, 2, enc.shape[1], -1)
            masks = torch.sigmoid(masks)
            enc_e = enc.unsqueeze(1)
            masked = masks * enc_e
            sources = []
            for s in range(2):
                dec = self.decoder(masked[:, s])
                sources.append(dec)
            out = torch.cat(sources, dim=1)
            if out.shape[-1] > T:
                out = out[..., :T]
            elif out.shape[-1] < T:
                out = F.pad(out, (0, T - out.shape[-1]))
            return out


_neo_instance = None
_neo_lock     = None

def _get_neo_lock():
    global _neo_lock
    if _neo_lock is None:
        import threading
        _neo_lock = threading.Lock()
    return _neo_lock


class NeoSSNetService:
    CKPT_PATH = os.path.expanduser("~/.cache/steth/neoSSNet.pt")

    def __init__(self, config):
        self.cfg     = config
        self.model   = None
        self.device  = "cpu"
        self.use_nmf = False

    def load(self):
        global _neo_instance
        if self.model is not None:
            return
        with _get_neo_lock():
            if self.model is not None:
                return
            if not TORCH_AVAILABLE:
                log.warning("PyTorch unavailable – using NMF fallback.")
                self.use_nmf = True
                return
            try:
                model = NeoSSNetModel(self.cfg)
                if os.path.exists(self.CKPT_PATH):
                    state = torch.load(self.CKPT_PATH, map_location="cpu")
                    model.load_state_dict(state, strict=False)
                    log.info("NeoSSNet loaded weights from %s", self.CKPT_PATH)
                else:
                    log.warning(
                        "NeoSSNet – no pretrained weights at %s; "
                        "falling back to NMF for reliable output.",
                        self.CKPT_PATH,
                    )
                    self.use_nmf = True
                    return
                model.eval()
                self.model = model
                log.info("NeoSSNet model ready.")
            except Exception as e:
                log.warning("NeoSSNet load failed: %s – using NMF fallback.", e)
                self.use_nmf = True

    def separate(self, clean_audio: np.ndarray, sr: int, nmf_service=None) -> dict:
        """
        Returns heart/lung arrays of EXACTLY len(clean_audio) samples.
        """
        self.load()
        if self.use_nmf or self.model is None:
            if nmf_service is None:
                raise RuntimeError("NMF fallback requested but no nmf_service provided.")
            log.info("Separation using NMF fallback.")
            result = nmf_service.separate(clean_audio, sr)
            result["method"] = "nmf"
            return result
        return self._infer(clean_audio, sr)

    def _infer(self, audio: np.ndarray, sr: int) -> dict:
        import torch
        import scipy.signal as sps
        N = len(audio)

        # Resample to 16 kHz
        audio16k = _resample_np(audio, sr, self.cfg.DL_SR)
        peak = float(np.abs(audio16k).max())
        if peak > 1e-9:
            audio16k = audio16k / peak

        with torch.no_grad():
            inp = torch.from_numpy(audio16k).unsqueeze(0).unsqueeze(0)  # (1,1,T)
            out = self.model(inp)                                        # (1,2,T)

        heart16k = out[0, 0].numpy().astype(np.float32)
        lung16k  = out[0, 1].numpy().astype(np.float32)

        # Resample back → guarantee original length N
        heart = _matchresample_np(heart16k, self.cfg.DL_SR, sr, N)
        lung  = _matchresample_np(lung16k,  self.cfg.DL_SR, sr, N)

        heart = _normalise_np(heart)
        lung  = _normalise_np(lung)

        noise = audio.astype(np.float32) - heart - lung
        nl, sq = _quality_np(heart, lung, noise)
        log.info("NeoSSNet separation done  noise_level=%.3f  quality=%.3f", nl, sq)
        return {
            "heart":          heart,
            "lung":           lung,
            "noise_level":    nl,
            "signal_quality": sq,
            "method":         "neoSSNet",
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers (module-level)
# ─────────────────────────────────────────────────────────────────────────────

def _resample_np(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x.astype(np.float32)
    import scipy.signal as sps
    n_out = int(round(len(x) * dst / src))
    return sps.resample_poly(x.astype(np.float64), dst, src)[:n_out].astype(np.float32)

def _matchresample_np(x: np.ndarray, src: int, dst: int, n: int) -> np.ndarray:
    resampled = _resample_np(x, src, dst)
    if len(resampled) >= n:
        return resampled[:n].astype(np.float32)
    return np.concatenate([resampled, np.zeros(n - len(resampled), dtype=np.float32)])

def _normalise_np(x: np.ndarray, headroom: float = 0.95) -> np.ndarray:
    p = float(np.abs(x).max())
    return (x * headroom / p).astype(np.float32) if p > 1e-9 else x.astype(np.float32)

def _quality_np(heart, lung, noise):
    eps = 1e-9
    rh  = float(np.sqrt(np.mean(heart ** 2))) + eps
    rl  = float(np.sqrt(np.mean(lung  ** 2))) + eps
    rn  = float(np.sqrt(np.mean(noise ** 2))) + eps
    nl  = rn / (rh + rl + rn)
    return round(nl, 4), round(1.0 - nl, 4)