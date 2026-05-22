"""
main.py  —  Biomedical Stethoscope Analysis API  v1.0.0
=======================================================

POST /analyze-audio
  Accepts: multipart/form-data  field "file" (wav / mp3 / flac / ogg)
  Returns: full JSON analysis

Pipeline:
  1. Load audio (full length, no trimming)
  2. Noise removal (SpeechBrain SepFormer or DSP fallback)
  3. Classification (heart / lungs / mixed / invalid)
  4. Source separation (NeoSSNet or NMF fallback)
  5. Cardiac cycle segmentation (S1/S2) + S3/S4 detection
  6. Murmur detection (all systolic + diastolic types)
  7. Lung classification (normal / wheeze / crackles)
  8. Residual noise segment detection
  9. Save outputs, return JSON

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations
import logging
import time
import uuid
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from utils.audio_io import load_audio, save_wav, resample
from services.noise_reduction       import NoiseReductionService
from services.classifier            import ClassifierService
from services.separator             import SeparatorService
from services.heart_analysis        import HeartAnalysisService
from services.murmur_detector       import MurmurDetector
from services.lung_analysis         import LungAnalysisService
from services.noise_segment_detector import NoiseSegmentDetector

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("steth.main")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Biomedical Stethoscope API",
    version="1.0.0",
    description="Clinical-grade heart & lung audio analysis",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR   = Path("outputs")
UPLOAD_DIR   = Path("uploads")
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff"}
MAX_FILE_MB = 50

# ── Lazy-loaded service singletons ────────────────────────────────────────────
_noise_svc   = NoiseReductionService()
_cls_svc     = ClassifierService()
_sep_svc     = SeparatorService()
_heart_svc   = HeartAnalysisService()
_murmur_svc  = MurmurDetector()
_lung_svc    = LungAnalysisService()
_noise_seg   = NoiseSegmentDetector()


# ════════════════════════════════════════════════════════════════════════════════
#  Endpoints
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    t_total = time.perf_counter()
    job_id  = uuid.uuid4().hex[:10]

    # ── 1. Validate & save upload ─────────────────────────────────────────────
    ext = Path(file.filename or "audio.wav").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported format '{ext}'. "
                   f"Accepted: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_FILE_MB} MB limit.",
        )

    upload_path = UPLOAD_DIR / f"{job_id}_input{ext}"
    upload_path.write_bytes(content)
    log.info("Job %s  file=%s  size=%.1f KB", job_id, file.filename,
             len(content) / 1024)

    try:
        # ── 2. Load audio (full length, no trimming) ──────────────────────────
        t0 = time.perf_counter()
        audio_raw, src_sr = load_audio(upload_path)
        input_length_ms   = round(len(audio_raw) / src_sr * 1000.0, 1)
        log.info("Job %s  loaded  sr=%d  len_ms=%.1f  load_ms=%.1f",
                 job_id, src_sr, input_length_ms,
                 (time.perf_counter() - t0) * 1000)

        # ── 3. Noise removal ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        audio_clean = _noise_svc.denoise(audio_raw, src_sr)
        log.info("Job %s  denoise_ms=%.1f", job_id,
                 (time.perf_counter() - t0) * 1000)

        # ── 4. Classification ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        classification = _cls_svc.classify(audio_clean, src_sr)
        log.info("Job %s  label=%s  cls_ms=%.1f",
                 job_id, classification["label"],
                 (time.perf_counter() - t0) * 1000)

        # Early exit for invalid audio
        if classification["label"] == "invalid":
            return JSONResponse({
                "job_id":         job_id,
                "input_length_ms":input_length_ms,
                "classification": classification,
                "outputs":        {},
                "cardiac_cycles": [],
                "extra_sounds":   [],
                "murmurs":        [],
                "noise_segments": [],
                "lung_analysis":  {},
                "status":         "invalid_audio",
                "message":        "Audio does not contain detectable biomedical sounds.",
            })

        # ── 5. Source separation ──────────────────────────────────────────────
        t0 = time.perf_counter()
        heart_audio, lung_audio = _sep_svc.separate(audio_clean, src_sr)
        log.info("Job %s  sep_ms=%.1f", job_id,
                 (time.perf_counter() - t0) * 1000)

        # Save separated files
        heart_path = OUTPUT_DIR / f"{job_id}_heart.wav"
        lung_path  = OUTPUT_DIR / f"{job_id}_lung.wav"
        save_wav(heart_audio, src_sr, heart_path)
        save_wav(lung_audio,  src_sr, lung_path)

        # ── 6. Cardiac analysis ───────────────────────────────────────────────
        cardiac_result = {"cardiac_cycles": [], "extra_sounds": [],
                          "bpm": None, "total_beats": 0}
        murmurs        = []

        if classification["heart"]:
            t0 = time.perf_counter()
            cardiac_result = _heart_svc.analyse(heart_audio, src_sr)
            log.info("Job %s  beats=%d  bpm=%s  cardiac_ms=%.1f",
                     job_id, cardiac_result["total_beats"],
                     cardiac_result["bpm"],
                     (time.perf_counter() - t0) * 1000)

            # ── 7. Murmur detection ───────────────────────────────────────────
            t0 = time.perf_counter()
            murmurs = _murmur_svc.detect(
                heart_audio, src_sr, cardiac_result["cardiac_cycles"])
            log.info("Job %s  murmurs=%d  murmur_ms=%.1f",
                     job_id, len(murmurs),
                     (time.perf_counter() - t0) * 1000)

        # ── 8. Lung analysis ──────────────────────────────────────────────────
        lung_result = {}
        if classification["lungs"]:
            t0 = time.perf_counter()
            lung_result = _lung_svc.analyse(lung_audio, src_sr)
            log.info("Job %s  lung_label=%s  lung_ms=%.1f",
                     job_id, lung_result.get("label"),
                     (time.perf_counter() - t0) * 1000)

        # ── 9. Residual noise segments ────────────────────────────────────────
        t0 = time.perf_counter()
        noise_segs = _noise_seg.detect(audio_clean, src_sr)
        log.info("Job %s  noise_segs=%d  nseg_ms=%.1f",
                 job_id, len(noise_segs),
                 (time.perf_counter() - t0) * 1000)

        # ── 10. Assemble response ─────────────────────────────────────────────
        total_ms = round((time.perf_counter() - t_total) * 1000, 1)
        log.info("Job %s  DONE  total_ms=%.1f", job_id, total_ms)

        return JSONResponse({
            "job_id":           job_id,
            "input_length_ms":  input_length_ms,
            "classification": {
                "heart": classification["heart"],
                "lungs": classification["lungs"],
                "label": classification["label"],
                "heart_score": classification["heart_score"],
                "lung_score":  classification["lung_score"],
                "confidence":  classification["confidence"],
            },
            "outputs": {
                "heart_audio": f"/audio/{job_id}_heart.wav",
                "lung_audio":  f"/audio/{job_id}_lung.wav",
            },
            "cardiac": {
                "bpm":         cardiac_result["bpm"],
                "total_beats": cardiac_result["total_beats"],
            },
            "cardiac_cycles": cardiac_result["cardiac_cycles"],
            "extra_sounds":   cardiac_result["extra_sounds"],
            "murmurs":        murmurs,
            "noise_segments": noise_segs,
            "lung_analysis":  lung_result,
            "processing_ms":  total_ms,
            "status":         "success",
        })

    except Exception as exc:
        log.exception("Job %s FAILED: %s", job_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(exc)}",
        )
    finally:
        # Clean up uploaded file
        if upload_path.exists():
            upload_path.unlink(missing_ok=True)


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve separated audio files for playback."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, f"Audio file '{filename}' not found.")
    return FileResponse(str(path), media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)