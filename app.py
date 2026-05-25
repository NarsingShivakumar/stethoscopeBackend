"""
app.py — Clinical AI Stethoscope API v5.0.0
=============================================
Endpoints:
  GET  /health
  GET  /metrics
  POST /process_audio      Legacy base64 separation
  POST /add_noise          Noise injection
  POST /detect_heart       Heart presence detection
  POST /analyze-audio      NEW clinical pipeline (multipart upload or JSON)
  POST /analyze-audio-b64  NEW clinical pipeline via base64 JSON
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections import defaultdict
from threading import Lock

from flask import Flask, g, jsonify, request
from flask_cors import CORS

from config.settings import Config
from services.separation_service import SeparationService
from services.noise_service import NoiseService, VALID_NOISE_TYPES
from services.noise_removal_service import NoiseRemovalService
from services.classification_service import ClassificationService
from services.neoSSNet import NeoSSNetService
from services.cardiac_service import CardiacService
from services.lung_service import LungService
from utils.audio_utils import (
    AudioDecodeError,
    decode_audio_payload,
    decode_uploaded_file,
    encode_audio_response,
    save_wav,
    samples_to_ms,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("steth.api")

app = Flask(__name__)
app.config.from_object(Config)
CORS(app, origins="*")

_svc_sep = SeparationService(Config)
_svc_noise = NoiseService()
_svc_nr = NoiseRemovalService(Config)
_svc_classify = ClassificationService()
_svc_neo = NeoSSNetService(Config)
_svc_cardiac = CardiacService(Config)
_svc_lung = LungService()

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

_m: dict = defaultdict(float)
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
    log.info("%s %s → %d (%.1f ms)", request.method, request.path, resp.status_code, ms)
    return resp


def _err(code: int, msg: str):
    return jsonify({"error": msg, "status": "error"}), code


def _parse_audio(body):
    if not body:
        raise ValueError("Request body must be valid JSON.")
    audio_b64 = body.get("audio")
    sample_rate = body.get("sample_rate", 44100)
    if not audio_b64:
        raise ValueError("Missing required field: 'audio'.")
    try:
        sample_rate = int(sample_rate)
    except (TypeError, ValueError):
        raise ValueError("'sample_rate' must be an integer.")
    if not (4000 <= sample_rate <= 192000):
        raise ValueError(f"sample_rate {sample_rate} out of range [4000, 192000].")
    try:
        audio_np, sr = decode_audio_payload(audio_b64, sample_rate)
    except AudioDecodeError as exc:
        raise ValueError(str(exc)) from exc
    if len(audio_np) < int(sr * Config.MIN_DURATION_SEC):
        raise ValueError(f"Audio too short (< {Config.MIN_DURATION_SEC}s).")
    if len(audio_np) > int(sr * Config.MAX_DURATION_SEC):
        raise ValueError(f"Audio too long (> {Config.MAX_DURATION_SEC}s).")
    return audio_np, sr


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "steth-clinical-api",
        "version": "5.0.0",
    }), 200


@app.route("/metrics", methods=["GET"])
def metrics():
    with _m_lock:
        snap = dict(_m)
    return jsonify(snap), 200


@app.route("/process_audio", methods=["POST"])
def process_audio():
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
        return _err(500, "Sound separation failed.")
    try:
        heart_b64 = encode_audio_response(result["heart"], sr)
        lung_b64 = encode_audio_response(result["lung"], sr)
    except Exception:
        log.exception("Encoding error")
        return _err(500, "Output encoding failed.")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _record("requests_ok")
    _record("total_processing_ms", elapsed_ms)
    return jsonify({
        "heart": heart_b64,
        "lung": lung_b64,
        "noise_level": result["noise_level"],
        "signal_quality": result["signal_quality"],
        "processing_ms": round(elapsed_ms, 1),
        "status": "success",
    }), 200


@app.route("/add_noise", methods=["POST"])
def add_noise():
    _record("add_noise_total")
    body = request.get_json(force=True, silent=True)
    try:
        audio_np, sr = _parse_audio(body)
    except ValueError as exc:
        return _err(422, str(exc))
    noise_type = str(body.get("noise_type", "white")).lower()
    if noise_type not in VALID_NOISE_TYPES:
        return _err(422, f"noise_type must be one of {sorted(VALID_NOISE_TYPES)}.")
    try:
        snr_db = float(body.get("snr_db", 10.0))
    except (TypeError, ValueError):
        return _err(422, "'snr_db' must be a number.")
    noisy = _svc_noise.mix_noise(audio_np, sr, noise_type=noise_type, snr_db=snr_db)
    import numpy as np
    original_rms = float(np.sqrt(np.mean(audio_np.astype(float) ** 2)))
    return jsonify({
        "audio": encode_audio_response(noisy, sr),
        "noise_type": noise_type,
        "snr_db": snr_db,
        "original_rms": round(original_rms, 6),
        "status": "success",
    }), 200


@app.route("/detect_heart", methods=["POST"])
def detect_heart():
    body = request.get_json(force=True, silent=True)
    try:
        audio_np, sr = _parse_audio(body)
    except ValueError as exc:
        return _err(422, str(exc))
    result = _svc_noise.detect_heart(audio_np, sr)
    return jsonify({**result, "status": "success"}), 200


@app.route("/analyze-audio", methods=["POST"])
def analyze_audio():
    t0 = time.perf_counter()
    _record("analyze_audio_total")

    try:
        if request.content_type and "multipart" in request.content_type:
            if "audio" not in request.files:
                return _err(422, "Missing 'audio' file field in form-data.")
            audio_np, sr = decode_uploaded_file(request.files["audio"])
        else:
            body = request.get_json(force=True, silent=True)
            audio_np, sr = _parse_audio(body)
    except ValueError as exc:
        return _err(422, str(exc))
    except Exception as exc:
        log.exception("Audio decode error")
        return _err(422, f"Could not decode audio: {exc}")

    N = len(audio_np)
    input_length_ms = samples_to_ms(N, sr)
    session_id = uuid.uuid4().hex[:8]

    try:
        nr_result = _svc_nr.remove_noise(audio_np, sr)
        clean_audio = nr_result["clean_audio"]
        noise_segments = nr_result["noise_segments"]
        snr_db = nr_result["snr_estimate_db"]
    except Exception:
        log.exception("Noise removal failed — using raw audio.")
        clean_audio = audio_np
        noise_segments = []
        snr_db = 0.0

    try:
        classification = _svc_classify.classify(clean_audio, sr)
    except Exception:
        log.exception("Classification error")
        classification = {"heart": True, "lungs": True, "classification": "mixed"}

    try:
        sep_result = _svc_neo.separate(clean_audio, sr, nmf_service=_svc_sep)
        heart_audio = sep_result["heart"]
        lung_audio = sep_result["lung"]
    except Exception:
        log.exception("Separation failed — using NMF fallback.")
        sep_result = _svc_sep.separate(clean_audio, sr)
        heart_audio = sep_result["heart"]
        lung_audio = sep_result["lung"]

    heart_filename = f"heart_{session_id}.wav"
    lung_filename = f"lung_{session_id}.wav"
    heart_path = os.path.join(Config.OUTPUT_DIR, heart_filename)
    lung_path = os.path.join(Config.OUTPUT_DIR, lung_filename)

    try:
        save_wav(heart_audio, sr, heart_path)
        save_wav(lung_audio, sr, lung_path)
    except Exception as e:
        log.warning("Could not save audio files: %s", e)
        heart_path = None
        lung_path = None

    cardiac_result = {"cardiac_cycles": [], "extra_sounds": [], "murmurs": [], "timeline": []}
    if classification.get("heart"):
        try:
            cardiac_result = _svc_cardiac.analyze(heart_audio, sr)
        except Exception:
            log.exception("Cardiac analysis failed.")

    lung_result = {"lung_classification": "normal"}
    if classification.get("lungs"):
        try:
            lung_result = _svc_lung.analyze(lung_audio, sr)
        except Exception:
            log.exception("Lung analysis failed.")

    try:
        heart_b64 = encode_audio_response(heart_audio, sr)
        lung_b64 = encode_audio_response(lung_audio, sr)
    except Exception:
        log.exception("Output encoding failed.")
        heart_b64 = None
        lung_b64 = None

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return jsonify({
        "status": "success",
        "session_id": session_id,
        "duration_ms": input_length_ms,
        "input_length_ms": input_length_ms,
        "audio_outputs": {
            "original": None,
            "cleaned": None,
            "heart": heart_path,
            "lungs": lung_path,
        },
        "outputs": {
            "heart_audio": heart_path,
            "lung_audio": lung_path,
        },
        "cardiac_cycles": cardiac_result.get("cardiac_cycles", []),
        "extra_sounds": cardiac_result.get("extra_sounds", []),
        "murmurs": cardiac_result.get("murmurs", []),
        "timeline": cardiac_result.get("timeline", []),
        "noise_segments": noise_segments,
        "noise_level": sep_result.get("noise_level", 0.0),
        "signal_quality": sep_result.get("signal_quality", 0.0),
        "lung_analysis": lung_result,
        "heart_audio_base64": heart_b64,
        "lung_audio_base64": lung_b64,
        "processing_ms": elapsed_ms,
        "snr_estimate_db": snr_db,
    }), 200


@app.route("/analyze-audio-b64", methods=["POST"])
def analyze_audio_b64():
    body = request.get_json(force=True, silent=True)
    if not body:
        return _err(422, "Request body must be valid JSON.")
    if "audio" not in body:
        return _err(422, "Missing required field: 'audio'.")
    body = dict(body)
    return analyze_audio()


@app.errorhandler(404)
def _404(_):
    return _err(404, "Endpoint not found.")


@app.errorhandler(405)
def _405(_):
    return _err(405, "Method not allowed.")


@app.errorhandler(500)
def _500(_):
    return _err(500, "Internal server error.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    log.info("Steth API v5 starting on :%d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)