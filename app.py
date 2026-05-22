"""
app.py  —  Stethoscope Heart/Lung Separation API  v3.0.0
=========================================================
New in v3:
  POST /add_noise          Inject voice/white/pink/brown noise at a given SNR
  POST /detect_heart       Detect heart sound in audio (returns bool + confidence)
  POST /process_audio      (unchanged) NMF separation

Run locally:
    python app.py

Production:
    gunicorn -w 2 -b 0.0.0.0:5000 --timeout 90 app:app
"""

import logging
import os
import time
from collections import defaultdict
from threading import Lock

from flask import Flask, g, jsonify, request
from flask_cors import CORS

from config.settings import Config
from services.separation_service import SeparationService
from services.noise_service import NoiseService, VALID_NOISE_TYPES
from utils.audio_utils import (
    AudioDecodeError, decode_audio_payload, encode_audio_response,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("steth.api")

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(Config)
CORS(app, origins="*")

_svc_sep   = SeparationService(Config)
_svc_noise = NoiseService()

_m: dict      = defaultdict(float)
_m_lock: Lock = Lock()


def _record(key: str, val: float = 1.0):
    with _m_lock:
        _m[key] += val


@app.before_request
def _t0():
    g.t0 = time.perf_counter()


@app.after_request
def _log(resp):
    ms = (time.perf_counter() - g.t0) * 1000
    log.info("%s %s → %d  (%.1f ms)", request.method, request.path,
             resp.status_code, ms)
    return resp


# ── Helper: decode + validate audio ──────────────────────────────────────────

def _parse_audio(body):
    """
    Shared decode logic. Returns (audio_np, sr) or raises.
    Raises ValueError with a user-facing message on failure.
    """
    if not body:
        raise ValueError("Request body must be valid JSON.")

    audio_b64   = body.get("audio")
    sample_rate = body.get("sample_rate", 44100)

    if not audio_b64:
        raise ValueError("Missing required field: 'audio'.")

    try:
        sample_rate = int(sample_rate)
    except (TypeError, ValueError):
        raise ValueError("'sample_rate' must be an integer (e.g. 44100).")

    if not (4000 <= sample_rate <= 192000):
        raise ValueError(f"sample_rate {sample_rate} is out of range [4000, 192000].")

    try:
        audio_np, sr = decode_audio_payload(audio_b64, sample_rate)
    except AudioDecodeError as exc:
        raise ValueError(str(exc)) from exc

    min_s = int(sr * Config.MIN_DURATION_SEC)
    max_s = int(sr * Config.MAX_DURATION_SEC)
    if len(audio_np) < min_s:
        raise ValueError(f"Audio too short (< {Config.MIN_DURATION_SEC}s).")
    if len(audio_np) > max_s:
        raise ValueError(f"Audio too long (> {Config.MAX_DURATION_SEC}s).")

    return audio_np, sr


# ════════════════════════════════════════════════════════════════════════════════
#  Routes
# ════════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "steth-separation-api",
        "version": "3.0.0",
    }), 200


@app.route("/metrics", methods=["GET"])
def metrics():
    with _m_lock:
        snap = dict(_m)
    return jsonify(snap), 200


# ── /process_audio (unchanged) ────────────────────────────────────────────────

@app.route("/process_audio", methods=["POST"])
def process_audio():
    """
    POST /process_audio
    ───────────────────
    Request  { "audio": "<base64 WAV/PCM-16>", "sample_rate": 44100 }
    Response {
      "heart":          "<base64 WAV>",
      "lung":           "<base64 WAV>",
      "noise_level":    0.12,
      "signal_quality": 0.88,
      "processing_ms":  342.0,
      "status":         "success"
    }
    """
    t0 = time.perf_counter()
    _record("requests_total")

    body = request.get_json(force=True, silent=True)
    try:
        audio_np, sr = _parse_audio(body)
    except ValueError as exc:
        _record("errors_parse")
        return _err(422, str(exc))

    try:
        result = _svc_sep.separate(audio_np, sr)
    except Exception:
        log.exception("Separation error")
        _record("errors_separation")
        return _err(500, "Sound separation failed — see server logs.")

    try:
        heart_b64 = encode_audio_response(result["heart"], sr)
        lung_b64  = encode_audio_response(result["lung"],  sr)
    except Exception:
        log.exception("Encoding error")
        return _err(500, "Output encoding failed.")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    _record("requests_ok")
    _record("total_processing_ms", elapsed_ms)

    return jsonify({
        "heart":          heart_b64,
        "lung":           lung_b64,
        "noise_level":    result["noise_level"],
        "signal_quality": result["signal_quality"],
        "processing_ms":  round(elapsed_ms, 1),
        "status":         "success",
    }), 200


# ── /add_noise  (NEW) ─────────────────────────────────────────────────────────

@app.route("/add_noise", methods=["POST"])
def add_noise():
    """
    POST /add_noise
    ───────────────
    Injects noise into audio and returns the mixed signal.
    Useful for testing separation robustness.

    Request JSON
    {
      "audio":       "<base64 WAV/PCM-16 mono>",
      "sample_rate": 44100,          // optional, default 44100
      "noise_type":  "voice",        // one of: voice | white | pink | brown
      "snr_db":      10              // signal-to-noise ratio in dB (default 10)
    }

    Response 200
    {
      "audio":         "<base64 WAV of noisy signal>",
      "noise_type":    "voice",
      "snr_db":        10,
      "original_rms":  0.182,
      "status":        "success"
    }
    """
    _record("add_noise_total")
    body = request.get_json(force=True, silent=True)

    try:
        audio_np, sr = _parse_audio(body)
    except ValueError as exc:
        return _err(422, str(exc))

    noise_type = str(body.get("noise_type", "white")).lower()
    if noise_type not in VALID_NOISE_TYPES:
        return _err(400,
            f"'noise_type' must be one of: {', '.join(sorted(VALID_NOISE_TYPES))}.")

    try:
        snr_db = float(body.get("snr_db", 10.0))
    except (TypeError, ValueError):
        return _err(400, "'snr_db' must be a number (e.g. 10).")

    if not (-20.0 <= snr_db <= 60.0):
        return _err(400, "'snr_db' must be between -20 and 60.")

    import numpy as np
    original_rms = float(np.sqrt(np.mean(audio_np.astype(np.float64) ** 2)))

    try:
        noisy_np = _svc_noise.mix_noise(audio_np, sr, noise_type, snr_db)
    except Exception:
        log.exception("Noise injection error")
        return _err(500, "Noise injection failed — see server logs.")

    try:
        noisy_b64 = encode_audio_response(noisy_np, sr)
    except Exception:
        log.exception("Encoding error")
        return _err(500, "Output encoding failed.")

    _record("add_noise_ok")
    log.info("add_noise  type=%s  snr=%.1f  sr=%d  N=%d",
             noise_type, snr_db, sr, len(audio_np))

    return jsonify({
        "audio":        noisy_b64,
        "noise_type":   noise_type,
        "snr_db":       snr_db,
        "original_rms": round(original_rms, 4),
        "status":       "success",
    }), 200


# ── /detect_heart  (NEW) ──────────────────────────────────────────────────────

@app.route("/detect_heart", methods=["POST"])
def detect_heart():
    """
    POST /detect_heart
    ──────────────────
    Detect whether heart sound is present in the audio.
    Works on the raw mixed recording AND on the separated heart channel.

    Request JSON
    {
      "audio":       "<base64 WAV/PCM-16 mono>",
      "sample_rate": 44100     // optional
    }

    Response 200
    {
      "heart_detected": true,
      "confidence":     0.74,    // 0-1  combined score
      "energy_ratio":   0.62,    // fraction of energy in 20-150 Hz band
      "periodicity":    0.91,    // autocorrelation peak strength (cardiac rhythm)
      "dominant_bpm":   72.4,    // estimated heart rate (null if not periodic)
      "status":         "success"
    }
    """
    _record("detect_heart_total")
    body = request.get_json(force=True, silent=True)

    try:
        audio_np, sr = _parse_audio(body)
    except ValueError as exc:
        return _err(422, str(exc))

    try:
        result = _svc_noise.detect_heart(audio_np, sr)
    except Exception:
        log.exception("Heart detection error")
        return _err(500, "Heart detection failed — see server logs.")

    _record("detect_heart_ok")
    log.info(
        "detect_heart  detected=%s  confidence=%.3f  bpm=%s  sr=%d  N=%d",
        result["heart_detected"], result["confidence"],
        result["dominant_bpm"], sr, len(audio_np),
    )

    return jsonify({**result, "status": "success"}), 200


# ── Error helpers ─────────────────────────────────────────────────────────────

def _err(code: int, msg: str):
    _record(f"http_{code}")
    return jsonify({"error": msg, "status": "error"}), code


@app.errorhandler(404)
def _404(_): return _err(404, "Endpoint not found.")

@app.errorhandler(405)
def _405(_): return _err(405, "Method not allowed.")

@app.errorhandler(500)
def _500(_): return _err(500, "Internal server error.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    log.info("Steth API v3 starting on :%d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)