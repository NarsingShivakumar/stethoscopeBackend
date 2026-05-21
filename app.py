"""
app.py  —  Stethoscope Heart/Lung Separation API
=================================================
Replaces ALL AiSteth API dependencies.

Endpoints:
  GET  /health          liveness probe
  GET  /metrics         basic counters
  POST /process_audio   NMF separation (main endpoint)

Run locally:
    python app.py

Production (Docker / gunicorn):
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
from utils.audio_utils import AudioDecodeError, decode_audio_payload, encode_audio_response

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

# Singleton — NMF is stateless so safe to share across threads
_svc = SeparationService(Config)

# In-memory metrics (light-weight; not persisted across restarts)
_m: dict     = defaultdict(float)
_m_lock: Lock = Lock()


def _record(key: str, val: float = 1.0):
    with _m_lock:
        _m[key] += val


# ── Request timing ─────────────────────────────────────────────────────────────
@app.before_request
def _t0():
    g.t0 = time.perf_counter()


@app.after_request
def _log(resp):
    ms = (time.perf_counter() - g.t0) * 1000
    log.info("%s %s → %d  (%.1f ms)", request.method, request.path,
             resp.status_code, ms)
    return resp


# ════════════════════════════════════════════════════════════════════════════════
#  Routes
# ════════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    """Liveness probe — Android app calls this on startup."""
    return jsonify({"status": "ok", "service": "steth-separation-api", "version": "2.0.0"}), 200


@app.route("/metrics", methods=["GET"])
def metrics():
    with _m_lock:
        snap = dict(_m)
    return jsonify(snap), 200


@app.route("/process_audio", methods=["POST"])
def process_audio():
    """
    POST /process_audio
    ───────────────────
    Request JSON
      { "audio": "<base64 PCM-16 mono or WAV>", "sample_rate": 44100 }

    Response 200
      {
        "heart":          "<base64 WAV @ original SR>",
        "lung":           "<base64 WAV @ original SR>",
        "noise_level":    0.12,
        "signal_quality": 0.88,
        "processing_ms":  342.0,
        "status":         "success"
      }
    """
    t0 = time.perf_counter()
    _record("requests_total")

    # 1. Parse body
    body = request.get_json(force=True, silent=True)
    if not body:
        _record("errors_bad_json")
        return _err(400, "Request body must be valid JSON.")

    audio_b64   = body.get("audio")
    sample_rate = body.get("sample_rate", 44100)

    if not audio_b64:
        _record("errors_missing_audio")
        return _err(400, "Missing required field: 'audio'.")

    try:
        sample_rate = int(sample_rate)
    except (TypeError, ValueError):
        return _err(400, "'sample_rate' must be an integer (e.g. 44100).")

    if not (4000 <= sample_rate <= 192000):
        return _err(400, f"sample_rate {sample_rate} is out of range [4000, 192000].")

    # 2. Decode base64 → float32 numpy array
    try:
        audio_np, sr = decode_audio_payload(audio_b64, sample_rate)
    except AudioDecodeError as exc:
        _record("errors_decode")
        return _err(422, str(exc))
    except Exception:
        log.exception("Unexpected decode error")
        _record("errors_decode")
        return _err(422, "Audio decode failed — ensure input is base64 PCM-16 or WAV.")

    min_samples = int(sr * Config.MIN_DURATION_SEC)
    max_samples = int(sr * Config.MAX_DURATION_SEC)
    if len(audio_np) < min_samples:
        return _err(422, f"Audio too short (< {Config.MIN_DURATION_SEC}s).")
    if len(audio_np) > max_samples:
        return _err(422, f"Audio too long (> {Config.MAX_DURATION_SEC}s).")

    # 3. NMF separation
    try:
        result = _svc.separate(audio_np, sr)
    except Exception:
        log.exception("Separation error")
        _record("errors_separation")
        return _err(500, "Sound separation failed — see server logs.")

    # 4. Encode outputs
    try:
        heart_b64 = encode_audio_response(result["heart"], sr)
        lung_b64  = encode_audio_response(result["lung"],  sr)
    except Exception:
        log.exception("Encoding error")
        _record("errors_encode")
        return _err(500, "Output encoding failed.")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    _record("requests_ok")
    _record("total_processing_ms", elapsed_ms)

    log.info("OK  sr=%d  N=%d  noise=%.3f  quality=%.3f  ms=%.0f",
             sr, len(audio_np),
             result["noise_level"], result["signal_quality"], elapsed_ms)

    return jsonify({
        "heart":          heart_b64,
        "lung":           lung_b64,
        "noise_level":    result["noise_level"],
        "signal_quality": result["signal_quality"],
        "processing_ms":  round(elapsed_ms, 1),
        "status":         "success",
    }), 200


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
    log.info("Steth API starting on :%d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
