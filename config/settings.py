"""
config/settings.py
==================
Unified configuration for the clinical AI stethoscope pipeline v5.
"""
import os

class Config:
    # ── egrooby STFT (NMF fallback) ───────────────────────────────────────────
    TARGET_SR    = 4000
    FFTSIZE      = 1024
    HOPSIZE      = 256
    WINDOWSIZE   = 512
    BETA_LOSS    = 1
    SPARSITY     = 0.1
    MAXITER      = 100
    K_HEART      = 20
    K_LUNG       = 10
    K_NOISE      = 20
    NMF_EPS      = 1e-9

    # ── Deep learning target SR ───────────────────────────────────────────────
    # SepFormer & NeoSSNet both work at 16 kHz
    DL_SR        = 16000

    # ── NeoSSNet (Conv-TasNet-style) architecture ─────────────────────────────
    N_FILTERS    = 512      # encoder filters
    FILTER_LEN   = 16       # encoder kernel length (samples @16kHz → 1ms)
    N_REPEATS    = 4        # TCN repeat blocks
    N_BLOCKS     = 8        # TCN blocks per repeat
    TCN_CHANNELS = 256      # bottleneck channels
    TCN_SKIP     = 128      # skip channels
    TCN_KERNEL   = 3        # depthwise conv kernel

    # ── Cardiac segmentation ──────────────────────────────────────────────────
    HEART_SR     = 4000     # analysis SR for S1/S2 (same as NMF)
    S1_BAND      = (20, 150)   # Hz
    S2_BAND      = (50, 200)   # Hz
    MIN_RR_MS    = 400      # min RR interval (150 BPM max)
    MAX_RR_MS    = 1500     # max RR interval (40 BPM min)
    SYSTOLE_RATIO_MIN = 0.25   # systole/RR min
    SYSTOLE_RATIO_MAX = 0.50   # systole/RR max

    # ── Murmur detection ──────────────────────────────────────────────────────
    MURMUR_WINDOW_MS   = 50    # sliding window for spectrogram analysis
    MURMUR_OVERLAP_MS  = 25
    MURMUR_THRESHOLD   = 0.35  # minimum confidence to report

    # ── Audio constraints ─────────────────────────────────────────────────────
    MIN_DURATION_SEC = 0.5
    MAX_DURATION_SEC = 60.0

    # ── Frequency heuristics (NMF assign) ────────────────────────────────────
    HEART_MAX_CENTROID_HZ = 180.0
    LUNG_MIN_BROADBAND_HZ = 80.0

    # ── Paths ─────────────────────────────────────────────────────────────────
    OUTPUT_DIR   = os.environ.get("OUTPUT_DIR", "/tmp/steth_outputs")

    # ── Flask ─────────────────────────────────────────────────────────────────
    PORT         = int(os.environ.get("PORT", 5000))
    FLASK_DEBUG  = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    JSON_SORT_KEYS = False