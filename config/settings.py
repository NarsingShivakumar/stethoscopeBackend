"""
config/settings.py
==================
All NMF/STFT parameters directly from egrooby example_code.m:

  options_tf.FFTSIZE   = 1024
  options_tf.HOPSIZE   = 256
  options_tf.WINDOWSIZE = 512
  options_nmf.beta_loss = 1      (KL divergence)
  options_nmf.sparsity  = 0.1
  MAXITER = 100
  K = [20 10 20 20 20 20]        (heart, lung, noise, stmv, bubble, cpap)

We use the first 3 groups: K_HEART=20, K_LUNG=10, K_NOISE=20
"""
import os


class Config:
    # ── egrooby STFT parameters (from example_code.m) ────────────────────────
    TARGET_SR   = 4000    # Hz — egrooby paper hardcodes 4000 Hz
    FFTSIZE     = 1024    # options_tf.FFTSIZE  = 1024
    HOPSIZE     = 256     # options_tf.HOPSIZE  = 256
    WINDOWSIZE  = 512     # options_tf.WINDOWSIZE = 512

    # ── egrooby NMF parameters ────────────────────────────────────────────────
    BETA_LOSS   = 1       # options_nmf.beta_loss = 1  (KL divergence / IS)
    SPARSITY    = 0.1     # options_nmf.sparsity  = 0.1
    MAXITER     = 100     # MAXITER = 100

    # ── K per component group (K=[20 10 20 20 20 20]) ─────────────────────────
    K_HEART     = 20      # K[0]
    K_LUNG      = 10      # K[1]
    K_NOISE     = 20      # K[2]
    # Total = 50 components

    NMF_EPS     = 1e-9    # numerical floor

    # ── Frequency heuristics ──────────────────────────────────────────────────
    # Heart sounds: 20–150 Hz (dominant 50–100 Hz)
    # Lung sounds:  80–1000 Hz (broadband)
    HEART_MAX_CENTROID_HZ  = 180.0
    LUNG_MIN_BROADBAND_HZ  = 80.0

    # ── Audio constraints ─────────────────────────────────────────────────────
    MIN_DURATION_SEC = 0.5
    MAX_DURATION_SEC = 45.0

    # ── Noise injection defaults ──────────────────────────────────────────────
    DEFAULT_NOISE_TYPE = "white"   # white | pink | brown | voice
    DEFAULT_SNR_DB     = 10.0      # dB  (positive = signal louder than noise)
    MIN_SNR_DB         = -20.0
    MAX_SNR_DB         = 60.0

    # ── Heart detection thresholds (matching noise_service.py) ────────────────
    HEART_DETECT_ENERGY_THRESHOLD  = 0.08
    HEART_DETECT_CONFIDENCE_MIN    = 0.25
    HEART_DETECT_AUTOCORR_MIN_PEAK = 0.10
    
    # ── Flask ─────────────────────────────────────────────────────────────────
    PORT        = int(os.environ.get("PORT", 5000))
    FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    JSON_SORT_KEYS = False
