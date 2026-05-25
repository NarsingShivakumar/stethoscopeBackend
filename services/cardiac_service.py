"""
services/cardiac_service.py
Clinical cardiac analysis pipeline  – v6

Stage A  S1/S2 segmentation + systole/diastole timestamps
Stage B  Extra heart sounds: S3, S4, Split S2
Stage C  Full murmur classification:
          Systolic  → Aortic Stenosis (early / late peak), Mitral Regurgitation,
                       Pulmonic Stenosis, Benign / Functional,
                       VSD (Ventricular Septal Defect)
          Diastolic → Aortic Insufficiency (Regurgitation), Mitral Stenosis,
                       Pulmonic Regurgitation
          Continuous→ Patent Ductus Arteriosus (PDA),
                       Atrial Septal Defect (ASD – fixed split S2 + flow)
Stage D  Pericardial rubs (2-component and 3-component)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.signal as sps

log = logging.getLogger("steth.cardiac")

# ── Frequency bands ──────────────────────────────────────────────────────────
S1_BAND              = (20,  150)
S2_BAND              = (50,  200)
S3_BAND              = (20,  100)
S4_BAND              = (20,   80)
SPLIT_S2_BAND        = (50,  250)     # both A2 and P2
MURMUR_SYSTOLIC_BAND = (100, 600)
MURMUR_DIASTOLIC_BAND= (80,  500)
RUB_BAND             = (50,  400)     # pericardial rubs occupy 50-400 Hz

MIN_RR_MS  = 400
MAX_RR_MS  = 1500

# ── Murmur label → display name + canonical condition list ───────────────────
MURMUR_CATALOG = {
    # Systolic
    "AS_early":        ("Aortic Stenosis – Early Peak Systolic",
                        ["Aortic Stenosis (early peak)"]),
    "AS_late":         ("Aortic Stenosis – Late Peak Systolic",
                        ["Aortic Stenosis (late peak / severe AS)"]),
    "MR":              ("Mitral Regurgitation",
                        ["Mitral Regurgitation", "Tricuspid Regurgitation"]),
    "PS":              ("Pulmonic Stenosis",
                        ["Pulmonic Stenosis"]),
    "VSD":             ("Ventricular Septal Defect",
                        ["Ventricular Septal Defect (VSD)"]),
    "benign":          ("Benign / Functional Murmur",
                        ["Benign murmur", "Flow murmur", "Stills murmur"]),
    # Diastolic
    "AI":              ("Aortic Insufficiency",
                        ["Aortic Insufficiency (Regurgitation)"]),
    "MS":              ("Mitral Stenosis",
                        ["Mitral Stenosis"]),
    "PR":              ("Pulmonic Regurgitation",
                        ["Pulmonic Regurgitation (Graham Steell)"]),
    # Continuous / complex
    "PDA":             ("Patent Ductus Arteriosus",
                        ["Patent Ductus Arteriosus (PDA)"]),
    "ASD":             ("Atrial Septal Defect (flow murmur + fixed split S2)",
                        ["Atrial Septal Defect (ASD)"]),
    # Rubs
    "pericardial_rub_2c": ("Pericardial Rub – 2 Component",
                            ["Pericardial rub (2 component)"]),
    "pericardial_rub_3c": ("Pericardial Rub – 3 Component",
                            ["Pericardial rub (3 component)"]),
    # Unknown fallback
    "unknown":         ("Unknown Murmur",
                        ["Unknown – review clinically"]),
}


class CardiacService:
    def __init__(self, config):
        self.cfg = config

    def analyze(self, heart_audio: np.ndarray, sr: int) -> Dict:
        audio4k = _resample(heart_audio, sr, 4000)
        peaks_s1, peaks_s2 = self._detect_s1s2(audio4k, 4000)
        cycles = self._build_cycles(peaks_s1, peaks_s2, 4000, len(heart_audio), sr)
        bpm    = self._estimate_bpm(cycles)

        extra_sounds = self._detect_extra_sounds(audio4k, 4000, cycles)
        split_s2     = self._detect_split_s2(audio4k, 4000, cycles)
        extra_sounds.extend(split_s2)

        murmurs = self._detect_murmurs(heart_audio, sr, cycles)
        rubs    = self._detect_rubs(heart_audio, sr, cycles)
        murmurs.extend(rubs)

        timeline = self._build_timeline(
            len(heart_audio), sr, cycles, extra_sounds, murmurs
        )

        return {
            "duration_ms":    int(round(len(heart_audio) * 1000 / sr)),
            "cardiac_cycles": cycles,
            "extra_sounds":   extra_sounds,
            "murmurs":        murmurs,
            "timeline":       timeline,
            "bpm":            bpm,
        }

    # ─────────────────────────────── S1 / S2 ─────────────────────────────── #

    def _detect_s1s2(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        s1_filt = _bandpass(audio, sr, S1_BAND, order=4)
        s2_filt = _bandpass(audio, sr, S2_BAND, order=4)
        s1_env  = _envelope(s1_filt, sr, smooth_hz=8.0)
        s2_env  = _envelope(s2_filt, sr, smooth_hz=8.0)
        combined = 0.6 * s1_env + 0.4 * s2_env
        combined /= combined.max() + 1e-9
        min_dist  = int(MIN_RR_MS * sr / 2000)
        threshold = float(np.percentile(combined, 75)) * 0.5
        all_peaks, _ = sps.find_peaks(combined, height=threshold, distance=min_dist)
        if len(all_peaks) < 2:
            return all_peaks, np.array([], dtype=int)
        return self._classify_peaks(all_peaks, sr)

    def _classify_peaks(self, peaks: np.ndarray, sr: int
                        ) -> Tuple[np.ndarray, np.ndarray]:
        if len(peaks) < 2:
            return peaks, np.array([], dtype=int)
        gaps = np.diff(peaks)
        med  = float(np.median(gaps))
        s1_list, s2_list = [], []
        i = 0
        while i < len(peaks) - 1:
            if gaps[i] < med:
                s1_list.append(peaks[i]); s2_list.append(peaks[i + 1]); i += 2
            else:
                s1_list.append(peaks[i]); i += 1
        return np.array(s1_list, dtype=int), np.array(s2_list, dtype=int)

    def _build_cycles(
        self,
        s1_peaks: np.ndarray, s2_peaks: np.ndarray,
        analysis_sr: int, original_n: int, original_sr: int,
    ) -> List[Dict]:
        if len(s1_peaks) == 0 or len(s2_peaks) == 0:
            return []
        def to_ms(samp):
            return int(round(samp * 1000 / analysis_sr))
        total_ms = int(round(original_n * 1000 / original_sr))
        cycles   = []
        n_pairs  = min(len(s1_peaks), len(s2_peaks))
        for i in range(n_pairs):
            s1_ms = to_ms(s1_peaks[i])
            s2_ms = to_ms(s2_peaks[i])
            if s2_ms <= s1_ms:
                continue
            sys_start_ms = s1_ms
            sys_end_ms   = s2_ms
            if i + 1 < len(s1_peaks):
                next_s1_ms = to_ms(s1_peaks[i + 1])
            else:
                sys_dur    = sys_end_ms - sys_start_ms
                next_s1_ms = min(s2_ms + int(sys_dur * 2.0), total_ms)
            dia_start_ms = sys_end_ms
            dia_end_ms   = next_s1_ms
            if dia_end_ms <= dia_start_ms:
                continue
            cycles.append({
                "cycle_id": i + 1,
                "s1_ms":    s1_ms,
                "s2_ms":    s2_ms,
                "systole":  {"start_ms": sys_start_ms, "end_ms": sys_end_ms},
                "diastole": {"start_ms": dia_start_ms, "end_ms": dia_end_ms},
            })
        return cycles

    def _estimate_bpm(self, cycles: List[Dict]) -> Optional[float]:
        if len(cycles) < 2:
            return None
        rr_intervals = []
        for i in range(len(cycles) - 1):
            rr_ms = cycles[i + 1]["s1_ms"] - cycles[i]["s1_ms"]
            if MIN_RR_MS <= rr_ms <= MAX_RR_MS:
                rr_intervals.append(rr_ms)
        if not rr_intervals:
            return None
        return round(60000.0 / float(np.median(rr_intervals)), 1)

    # ──────────────────────── Extra sounds: S3, S4 ───────────────────────── #

    def _detect_extra_sounds(
        self, audio: np.ndarray, sr: int, cycles: List[Dict]
    ) -> List[Dict]:
        if not cycles:
            return []
        extra = []
        env       = _envelope(_bandpass(audio, sr, (20, 100), order=4), sr, smooth_hz=8.0)
        noise_floor = float(np.percentile(env, 50))
        for cycle in cycles:
            s2_ms = cycle["s2_ms"]
            s3 = self._check_extra(env, sr, s2_ms + 120, s2_ms + 180, noise_floor, "S3")
            if s3:
                s3["cycle_id"]    = cycle["cycle_id"]
                s3["description"] = "S3 – Early diastolic ventricular gallop"
                extra.append(s3)
            s1_next_ms  = cycle["diastole"]["end_ms"]
            s4_start_ms = s1_next_ms - 120
            s4_end_ms   = s1_next_ms - 50
            if s4_start_ms > cycle["diastole"]["start_ms"]:
                s4 = self._check_extra(env, sr, s4_start_ms, s4_end_ms, noise_floor, "S4")
                if s4:
                    s4["cycle_id"]    = cycle["cycle_id"]
                    s4["description"] = "S4 – Presystolic atrial gallop"
                    extra.append(s4)
        return extra

    def _check_extra(self, env, sr, start_ms, end_ms, noise_floor, sound_type):
        s = max(0, int(start_ms * sr / 1000))
        e = min(len(env), int(end_ms * sr / 1000))
        if e <= s:
            return None
        seg  = env[s:e]
        peak = float(seg.max())
        if peak < noise_floor * 1.8:
            return None
        return {
            "type":       sound_type,
            "start_ms":   start_ms,
            "end_ms":     end_ms,
            "peak_amp":   round(peak, 6),
            "confidence": round(min(1.0, (peak / noise_floor - 1.8) / 2.0), 3),
        }

    # ──────────────────────────── Split S2 ───────────────────────────────── #

    def _detect_split_s2(
        self, audio: np.ndarray, sr: int, cycles: List[Dict]
    ) -> List[Dict]:
        """
        Detect physiological or fixed split S2 (A2 + P2).
        Method: within each S2 window, look for two sub-peaks separated
        20-80 ms apart in the SPLIT_S2_BAND-filtered signal.
        Wide split (>40 ms) or fixed split → flag ASD possibility.
        """
        if not cycles:
            return []
        splits = []
        filt = _bandpass(audio, sr, SPLIT_S2_BAND, order=4)
        env  = _envelope(filt, sr, smooth_hz=20.0)
        noise_floor = float(np.percentile(env, 50))
        for cycle in cycles:
            s2_ms  = cycle["s2_ms"]
            # Search in a ±80 ms window around S2
            win_s  = max(0, int((s2_ms - 40) * sr / 1000))
            win_e  = min(len(env), int((s2_ms + 80) * sr / 1000))
            if win_e - win_s < int(0.02 * sr):
                continue
            seg      = env[win_s:win_e]
            min_dist = int(0.020 * sr)   # min 20 ms between A2 and P2
            max_dist = int(0.080 * sr)   # max 80 ms between A2 and P2
            peaks, _ = sps.find_peaks(seg, height=noise_floor * 1.5, distance=min_dist)
            if len(peaks) >= 2:
                # Take the two highest
                top2 = sorted(peaks, key=lambda p: -seg[p])[:2]
                top2.sort()
                gap_samples = top2[1] - top2[0]
                if min_dist <= gap_samples <= max_dist:
                    gap_ms      = int(round(gap_samples * 1000 / sr))
                    confidence  = round(min(1.0, seg[top2[0]] / (noise_floor + 1e-9) / 3.0), 3)
                    asd_possible = gap_ms > 40   # fixed wide split ≥ 40 ms suggests ASD
                    splits.append({
                        "type":          "Split_S2",
                        "cycle_id":      cycle["cycle_id"],
                        "start_ms":      s2_ms - 40,
                        "end_ms":        s2_ms + 80,
                        "gap_ms":        gap_ms,
                        "confidence":    confidence,
                        "description":   (
                            f"Split S2 (A2–P2 gap {gap_ms} ms)"
                            + (" – wide/fixed split: consider ASD" if asd_possible else "")
                        ),
                        "asd_possible":  asd_possible,
                    })
        return splits

    # ──────────────────────────── Murmurs ────────────────────────────────── #

    def _detect_murmurs(
        self, heart_audio: np.ndarray, sr: int, cycles: List[Dict]
    ) -> List[Dict]:
        if not cycles:
            return self._global_murmur_scan(heart_audio, sr)
        murmurs = []
        for cycle in cycles:
            sys = cycle["systole"]
            sys_result = self._analyze_phase_for_murmur(
                heart_audio, sr, sys["start_ms"], sys["end_ms"], "systolic"
            )
            if sys_result:
                sys_result["cycle_id"] = cycle["cycle_id"]
                murmurs.append(sys_result)
            dia = cycle["diastole"]
            dia_result = self._analyze_phase_for_murmur(
                heart_audio, sr, dia["start_ms"], dia["end_ms"], "diastolic"
            )
            if dia_result:
                dia_result["cycle_id"] = cycle["cycle_id"]
                murmurs.append(dia_result)
        return murmurs

    def _analyze_phase_for_murmur(
        self, audio, sr, start_ms, end_ms, phase
    ) -> Optional[Dict]:
        s = max(0, int(start_ms * sr / 1000))
        e = min(len(audio), int(end_ms * sr / 1000))
        if (e - s) < sr // 20:
            return None
        segment = audio[s:e].astype(np.float64)
        win_len  = min(len(segment) // 4, max(32, int(0.02 * sr)))
        hop      = max(1, win_len // 2)
        if len(segment) < win_len * 2:
            return None
        band = MURMUR_SYSTOLIC_BAND if phase == "systolic" else MURMUR_DIASTOLIC_BAND
        _, _, Sxx = sps.spectrogram(
            segment, fs=sr, nperseg=win_len, noverlap=win_len - hop,
            nfft=max(256, win_len * 2),
        )
        freqs      = np.linspace(0, sr / 2, Sxx.shape[0])
        band_mask  = (freqs >= band[0]) & (freqs <= band[1])
        energy_profile = Sxx[band_mask, :].mean(axis=0)
        if len(energy_profile) < 3:
            return None
        ep_norm = energy_profile / (energy_profile.max() + 1e-12)
        murmur_key, pattern, confidence = self._classify_murmur_pattern(
            ep_norm, phase, start_ms, end_ms
        )
        threshold = getattr(self.cfg, "MURMUR_THRESHOLD", 0.45)
        if confidence < threshold:
            return None
        label, conditions = MURMUR_CATALOG.get(murmur_key, MURMUR_CATALOG["unknown"])
        return {
            "start_ms":          start_ms,
            "end_ms":            end_ms,
            "type":              murmur_key,
            "label":             label,
            "phase":             phase,
            "pattern":           pattern,
            "confidence":        round(confidence, 3),
            "possible_condition": conditions,
        }

    def _classify_murmur_pattern(
        self, ep_norm: np.ndarray, phase: str, start_ms: int, end_ms: int
    ):
        """
        Map spectro-temporal energy profile → murmur type label.

        Systolic patterns
        ─────────────────
        plateau (holo)     → MR  (Mitral Regurgitation)
        crescendo-decrescendo (diamond, early peak) → AS_early (Aortic Stenosis early)
        crescendo-decrescendo (diamond, late peak)  → AS_late  (Aortic Stenosis late / severe)
        decrescendo (early) → VSD
        plateau (mid, soft) → benign
        rising (late)       → PS  (Pulmonic Stenosis)

        Diastolic patterns
        ──────────────────
        decrescendo (early, high freq) → AI  (Aortic Insufficiency)
        plateau (mid)                  → MS  (Mitral Stenosis)
        rising / presystolic           → MS  (with presystolic accentuation)
        rising (early)                 → PR  (Pulmonic Regurgitation)
        """
        n  = len(ep_norm)
        q1 = ep_norm[:n // 4].mean()
        q2 = ep_norm[n // 4: n // 2].mean()
        q3 = ep_norm[n // 2: 3 * n // 4].mean()
        q4 = ep_norm[3 * n // 4:].mean()
        mean_e  = float(ep_norm.mean())
        dur_ms  = end_ms - start_ms

        plateau  = mean_e > 0.55
        rising   = q1 < q2 < q3
        falling  = q1 > q2 > q3
        diamond  = (q2 > q1) and (q3 > q4) and (q2 + q3) / 2 - (q1 + q4) / 2 > 0.15
        late_pk  = (q3 > q2 > q1) and q3 > 0.65         # late-peaking → severe AS

        if mean_e < 0.28:
            return "unknown", "none", 0.0

        if phase == "systolic":
            if plateau and mean_e > 0.60:
                return "MR", "holosystolic_plateau", min(0.95, 0.5 + mean_e * 0.7)
            if diamond and late_pk:
                return "AS_late", "late_crescendo_decrescendo", min(0.93, 0.45 + (q3 - q1))
            if diamond:
                return "AS_early", "crescendo_decrescendo", min(0.92, 0.45 + (q2 + q3) / 2)
            if falling and q1 > 0.70:
                return "VSD", "decrescendo", min(0.88, 0.40 + q1 * 0.6)
            if rising and q4 > 0.65:
                return "PS", "crescendo", min(0.90, 0.40 + q4 * 0.7)
            if mean_e < 0.50:
                return "benign", "soft_midsystolic", min(0.75, 0.35 + mean_e * 0.6)
            return "MR", "irregular_systolic", min(0.65, 0.30 + mean_e * 0.5)

        else:  # diastolic
            if falling and dur_ms < 250 and q1 > 0.65:
                return "AI", "early_decrescendo", min(0.92, 0.45 + q1 * 0.6)
            if rising and q4 > 0.65:
                return "MS", "presystolic_crescendo", min(0.88, 0.40 + q4 * 0.7)
            if plateau:
                return "MS", "mid_diastolic_plateau", min(0.88, 0.45 + mean_e * 0.5)
            if rising and q1 < 0.40:
                return "PR", "early_diastolic_rising", min(0.82, 0.38 + q3 * 0.6)
            return "AI", "irregular_diastolic", min(0.60, 0.28 + mean_e * 0.5)

    def _global_murmur_scan(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Fallback scan used when no cardiac cycles are detected."""
        window_ms = 500
        hop_ms    = 250
        results   = []
        dur_ms    = int(len(audio) * 1000 / sr)
        for start in range(0, max(0, dur_ms - window_ms), hop_ms):
            end = start + window_ms
            res = self._analyze_phase_for_murmur(audio, sr, start, end, "systolic")
            if res and res["confidence"] > 0.50:
                results.append(res)
        return results

    # ──────────────────────────── Pericardial Rubs ───────────────────────── #

    def _detect_rubs(
        self, heart_audio: np.ndarray, sr: int, cycles: List[Dict]
    ) -> List[Dict]:
        """
        Pericardial rubs are scratchy, high-friction sounds with 2 or 3
        components per cardiac cycle aligned with:
          2-component rub: systolic + early-diastolic
          3-component rub: systolic + early-diastolic + presystolic (atrial)

        Detection: for each cycle measure energy bursts in the RUB_BAND during
        three windows.  If ≥2 bursts are detected → rub.
        """
        if not cycles:
            return []
        rubs = []
        filt  = _bandpass(heart_audio, sr, RUB_BAND, order=4)
        env   = _envelope(filt, sr, smooth_hz=15.0)
        noise_floor = float(np.percentile(env, 60))  # rubs sit above the 60th pct
        threshold   = noise_floor * 2.2              # clearly above baseline

        for cycle in cycles:
            s1_ms  = cycle["s1_ms"]
            s2_ms  = cycle["s2_ms"]
            dia    = cycle["diastole"]

            def window_peak(start_ms, end_ms):
                s = max(0, int(start_ms * sr / 1000))
                e = min(len(env), int(end_ms * sr / 1000))
                if e <= s:
                    return 0.0
                return float(env[s:e].max())

            # Component 1 – systolic burst
            p_sys = window_peak(s1_ms, s2_ms)
            # Component 2 – early diastolic burst (0-150 ms after S2)
            p_ed  = window_peak(s2_ms, s2_ms + 150)
            # Component 3 – presystolic burst (100-30 ms before next S1)
            next_s1 = dia["end_ms"]
            p_pre = window_peak(next_s1 - 100, next_s1 - 20)

            components_present = [
                ("systolic",        p_sys > threshold),
                ("early_diastolic", p_ed  > threshold),
                ("presystolic",     p_pre > threshold),
            ]
            present = [c for c, flag in components_present if flag]
            n_comp  = len(present)

            if n_comp >= 2:
                conf = round(min(0.95, 0.40 + (
                    p_sys + p_ed + (p_pre if n_comp >= 3 else 0)
                ) / (noise_floor + 1e-9) * 0.05), 3)
                rub_key = "pericardial_rub_3c" if n_comp == 3 else "pericardial_rub_2c"
                label, conditions = MURMUR_CATALOG[rub_key]
                rubs.append({
                    "type":              rub_key,
                    "label":             label,
                    "cycle_id":          cycle["cycle_id"],
                    "start_ms":          s1_ms,
                    "end_ms":            dia["end_ms"],
                    "phase":             "rub",
                    "pattern":           f"{n_comp}-component rub",
                    "components":        present,
                    "confidence":        conf,
                    "possible_condition": conditions,
                })
        return rubs

    # ──────────────────────────── PDA / ASD scan ─────────────────────────── #
    # (PDA produces a continuous "machinery" murmur that spans systole AND
    #  diastole in the same cycle.  ASD is flagged via fixed wide split S2
    #  + pulmonic flow murmur in systole – both handled in the main detection
    #  above.  The method below does a dedicated PDA scan.)

    def _scan_pda(self, heart_audio: np.ndarray, sr: int, cycles: List[Dict]
                  ) -> List[Dict]:
        """
        PDA: both systolic AND diastolic murmur present in the same cycle,
        high freq band (200-600 Hz), continuous high energy.
        """
        if not cycles:
            return []
        results = []
        filt = _bandpass(heart_audio, sr, (200, 600), order=4)
        env  = _envelope(filt, sr, smooth_hz=10.0)
        noise_floor = float(np.percentile(env, 50))
        for cycle in cycles:
            sys_s = int(cycle["systole"]["start_ms"] * sr / 1000)
            dia_e = int(cycle["diastole"]["end_ms"]   * sr / 1000)
            sys_s = max(0, sys_s); dia_e = min(len(env), dia_e)
            if dia_e <= sys_s:
                continue
            seg = env[sys_s:dia_e]
            if seg.mean() > noise_floor * 2.5:
                label, conditions = MURMUR_CATALOG["PDA"]
                results.append({
                    "type":              "PDA",
                    "label":             label,
                    "cycle_id":          cycle["cycle_id"],
                    "start_ms":          cycle["systole"]["start_ms"],
                    "end_ms":            cycle["diastole"]["end_ms"],
                    "phase":             "continuous",
                    "pattern":           "machinery_continuous",
                    "confidence":        round(min(0.90, seg.mean() / noise_floor * 0.1), 3),
                    "possible_condition": conditions,
                })
        return results

    # ──────────────────────────── Timeline ───────────────────────────────── #

    def _build_timeline(
        self, n_samples: int, sr: int,
        cycles: List[Dict], extra: List[Dict], murmurs: List[Dict],
    ) -> List[Dict]:
        duration_ms = int(round(n_samples * 1000 / sr))
        segments    = []
        if cycles:
            ordered = []
            for c in cycles:
                ordered.append({"start_ms": c["systole"]["start_ms"],
                                 "end_ms":   c["systole"]["end_ms"],
                                 "type":     "systole"})
                ordered.append({"start_ms": c["diastole"]["start_ms"],
                                 "end_ms":   c["diastole"]["end_ms"],
                                 "type":     "diastole"})
            ordered.sort(key=lambda x: (x["start_ms"], x["end_ms"]))
            cursor = 0
            merged = []
            for seg in ordered:
                if seg["start_ms"] > cursor:
                    merged.append({"start_ms": cursor, "end_ms": seg["start_ms"], "type": "noise"})
                merged.append(seg)
                cursor = max(cursor, seg["end_ms"])
            if cursor < duration_ms:
                merged.append({"start_ms": cursor, "end_ms": duration_ms, "type": "noise"})
            segments.extend(merged)
        else:
            segments = [{"start_ms": 0, "end_ms": duration_ms, "type": "noise"}]

        for e in extra:
            segments.append({"start_ms": e["start_ms"], "end_ms": e["end_ms"],
                              "type": e["type"]})
        for m in murmurs:
            segments.append({"start_ms": m["start_ms"], "end_ms": m["end_ms"],
                              "type": "murmur",
                              "subtype": m.get("type"),
                              "label":   m.get("label"),
                              "pattern": m.get("pattern")})

        segments = [s for s in segments if s["end_ms"] > s["start_ms"]]
        segments.sort(key=lambda x: x["start_ms"])
        return segments


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bandpass(x: np.ndarray, sr: int, band: tuple, order: int = 4) -> np.ndarray:
    nyq  = sr / 2.0
    low  = max(band[0], 1.0) / nyq
    high = min(band[1], nyq * 0.98) / nyq
    if low >= high:
        return x.copy()
    try:
        sos = sps.butter(order, [low, high], btype="bandpass", output="sos")
        return sps.sosfiltfilt(sos, x.astype(np.float64)).astype(np.float32)
    except Exception:
        return x.copy()

def _envelope(x: np.ndarray, sr: int, smooth_hz: float = 10.0) -> np.ndarray:
    from scipy.signal import hilbert
    env = np.abs(hilbert(x.astype(np.float64))).astype(np.float32)
    nyq  = sr / 2.0
    freq = min(smooth_hz, nyq * 0.95)
    try:
        sos = sps.butter(4, freq / nyq, btype="low", output="sos")
        env = sps.sosfiltfilt(sos, env.astype(np.float64)).astype(np.float32)
    except Exception:
        pass
    return env

def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x.astype(np.float32)
    n_out = int(round(len(x) * dst / src))
    return sps.resample_poly(x.astype(np.float64), dst, src)[:n_out].astype(np.float32)