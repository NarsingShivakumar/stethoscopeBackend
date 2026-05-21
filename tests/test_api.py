"""
tests/test_api.py  —  Full test suite
Run:  cd backend && pytest tests/ -v
"""
import base64, io, json, wave
import numpy as np
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_wav(sr=44100, duration=3.0):
    t   = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = (0.35 * np.sin(2 * np.pi * 80 * t)   # heart-like
         + 0.25 * np.sin(2 * np.pi * 300 * t)  # lung-like
         + 0.04 * np.random.default_rng(0).standard_normal(len(t)))
    pcm = (np.clip(sig, -1, 1) * 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(pcm)
    return buf.getvalue()


def b64(wav_bytes): return base64.b64encode(wav_bytes).decode()
def b64_raw(sr=44100, dur=2.0):
    t = np.linspace(0, dur, int(sr * dur))
    return base64.b64encode((np.sin(2*np.pi*100*t)*16000).astype(np.int16).tobytes()).decode()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as c: yield c

@pytest.fixture(scope="module")
def wav_b64(): return b64(make_wav())


# ── Health ────────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert json.loads(r.data)["status"] == "ok"


# ── Happy path ────────────────────────────────────────────────────────────────

def test_process_wav(client, wav_b64):
    r = client.post("/process_audio",
                    data=json.dumps({"audio": wav_b64, "sample_rate": 44100}),
                    content_type="application/json")
    assert r.status_code == 200, r.data
    d = json.loads(r.data)
    assert d["status"] == "success"
    assert "heart" in d and "lung" in d
    assert base64.b64decode(d["heart"])[:4] == b"RIFF"
    assert base64.b64decode(d["lung"])[:4]  == b"RIFF"
    assert 0.0 <= d["noise_level"]    <= 1.0
    assert 0.0 <= d["signal_quality"] <= 1.0
    assert d["processing_ms"] > 0


def test_process_raw_pcm(client):
    r = client.post("/process_audio",
                    data=json.dumps({"audio": b64_raw(44100, 2.0), "sample_rate": 44100}),
                    content_type="application/json")
    assert r.status_code == 200, r.data


def test_process_8khz(client):
    r = client.post("/process_audio",
                    data=json.dumps({"audio": b64(make_wav(8000, 3.0)), "sample_rate": 8000}),
                    content_type="application/json")
    assert r.status_code == 200, r.data


def test_output_length_matches_input(client, wav_b64):
    r = client.post("/process_audio",
                    data=json.dumps({"audio": wav_b64, "sample_rate": 44100}),
                    content_type="application/json")
    d  = json.loads(r.data)
    for key in ("heart", "lung"):
        raw = base64.b64decode(d[key])
        buf = io.BytesIO(raw)
        with wave.open(buf, "rb") as wf:
            n = wf.getnframes()
        # allow ±5 % rounding from resample
        assert abs(n - 44100 * 3) < 44100 * 3 * 0.05


# ── Error cases ───────────────────────────────────────────────────────────────

def test_missing_audio(client):
    r = client.post("/process_audio",
                    data=json.dumps({"sample_rate": 44100}),
                    content_type="application/json")
    assert r.status_code == 400


def test_bad_sample_rate_str(client, wav_b64):
    r = client.post("/process_audio",
                    data=json.dumps({"audio": wav_b64, "sample_rate": "x"}),
                    content_type="application/json")
    assert r.status_code == 400


def test_sample_rate_too_low(client, wav_b64):
    r = client.post("/process_audio",
                    data=json.dumps({"audio": wav_b64, "sample_rate": 100}),
                    content_type="application/json")
    assert r.status_code == 400


def test_too_short(client):
    short = base64.b64encode(np.zeros(100, dtype="<i2").tobytes()).decode()
    r = client.post("/process_audio",
                    data=json.dumps({"audio": short, "sample_rate": 44100}),
                    content_type="application/json")
    assert r.status_code == 422


def test_bad_base64(client):
    r = client.post("/process_audio",
                    data=json.dumps({"audio": "!!!NOT_B64!!!", "sample_rate": 44100}),
                    content_type="application/json")
    assert r.status_code == 422


def test_empty_body(client):
    r = client.post("/process_audio", data="", content_type="application/json")
    assert r.status_code == 400


def test_404(client):
    assert client.get("/nonexistent").status_code == 404


def test_405(client):
    assert client.get("/process_audio").status_code == 405


# ── Separation service unit tests ─────────────────────────────────────────────

class TestSep:
    @pytest.fixture(autouse=True)
    def setup(self):
        from config.settings import Config
        from services.separation_service import SeparationService
        self.svc = SeparationService(Config)

    def _audio(self, sr=44100, secs=2.0):
        t = np.linspace(0, secs, int(sr * secs))
        return (0.5 * np.sin(2*np.pi*80*t)).astype(np.float32)

    def test_shape(self):
        a = self._audio(); r = self.svc.separate(a, 44100)
        assert r["heart"].shape == a.shape
        assert r["lung"].shape  == a.shape

    def test_dtype(self):
        a = self._audio(); r = self.svc.separate(a, 44100)
        assert r["heart"].dtype == np.float32
        assert r["lung"].dtype  == np.float32

    def test_normalised(self):
        a = self._audio(); r = self.svc.separate(a, 44100)
        assert np.abs(r["heart"]).max() <= 1.05
        assert np.abs(r["lung"]).max()  <= 1.05

    def test_quality_range(self):
        a = self._audio(secs=3.0); r = self.svc.separate(a, 44100)
        assert 0 <= r["noise_level"]    <= 1
        assert 0 <= r["signal_quality"] <= 1

    def test_silent(self):
        a = np.zeros(44100 * 2, dtype=np.float32)
        r = self.svc.separate(a, 44100)   # must not crash
        assert r["heart"].shape == a.shape

    def test_min_length(self):
        a = np.random.randn(int(44100 * 0.6)).astype(np.float32)
        r = self.svc.separate(a, 44100)
        assert r["heart"].shape == a.shape


# ── Audio utils unit tests ────────────────────────────────────────────────────

class TestAudio:
    def test_wav_roundtrip(self):
        from utils.audio_utils import decode_audio_payload, encode_audio_response
        wav = make_wav(44100, 2.0)
        arr, sr = decode_audio_payload(b64(wav), 44100)
        assert sr == 44100; assert arr.dtype == np.float32
        enc = encode_audio_response(arr, sr)
        assert base64.b64decode(enc)[:4] == b"RIFF"

    def test_raw_pcm(self):
        from utils.audio_utils import decode_audio_payload
        arr, sr = decode_audio_payload(b64_raw(), 44100)
        assert sr == 44100; assert arr.dtype == np.float32

    def test_bad_b64_raises(self):
        from utils.audio_utils import decode_audio_payload, AudioDecodeError
        with pytest.raises(AudioDecodeError): decode_audio_payload("!!!BAD!!!", 44100)

    def test_empty_raises(self):
        from utils.audio_utils import decode_audio_payload, AudioDecodeError
        with pytest.raises(AudioDecodeError):
            decode_audio_payload(base64.b64encode(b"").decode(), 44100)
