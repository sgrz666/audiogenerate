#!/usr/bin/env python3
"""Track A/B scoring with a PRE-REGISTERED normalization scheme.

The plan requires fixing normalization before scoring, and recording 未测试 /
接口不支持 / 运行失败 separately. This script therefore:
  * computes only the dimensions it can actually measure,
  * marks every other dimension not_tested (never imputes a score),
  * reports coverage so a partial score is never mistaken for a full one.

Pre-registered normalization (project decisions, NOT model-official metrics):
  cer_score   = clip(1 - CER/0.20, 0, 1)
                0.20 CER is treated as unusable for exam listening material.
  sim_score   = clip((spk_sim - 0.50)/(0.95 - 0.50), 0, 1)
                CAM++ cosine ~0.50 ~ different-speaker baseline, 0.95 ~ near-identical.
  audio_score = 1 - (clip_penalty + silence_penalty), each in [0,1], floored at 0.
                clip_penalty    = min(1, clipped_sample_ratio / 0.01)
                silence_penalty = min(1, max(0, head_tail_silence_ratio - 0.25)/0.5)
  rate/efficiency are reported as raw values (RTF, peak VRAM) plus a fits_24gb flag,
                not folded into a quality score.

Dimensions that need P2 control sweeps or human blind listening are not_tested here.
Audio-quality proxies are computed on CPU from the WAVs (no GPU needed).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_common as C  # noqa: E402

CER_BAD = 0.20
SIM_LO, SIM_HI = 0.50, 0.95

TRACK_A_WEIGHTS = {
    "content_pronunciation": 0.30,
    "voice_consistency": 0.25,
    "naturalness": 0.20,
    "intonation": 0.10,
    "audio_quality": 0.10,
    "stability": 0.05,
}
TRACK_B_WEIGHTS = {
    "speaking_rate_control": 0.25,
    "pause_control": 0.20,
    "intonation_control": 0.15,
    "emotion_control": 0.15,
    "pronunciation_correction": 0.15,
    "vram_efficiency": 0.10,
}
# what this script can measure without P2 sweeps / human listening
MEASURABLE_A = {"content_pronunciation", "voice_consistency", "audio_quality"}
MEASURABLE_B = {"vram_efficiency"}


def clip01(x):
    return float(max(0.0, min(1.0, x)))


def audio_proxies(path):
    """CPU-only objective audio-quality proxies."""
    w, sr = C.read_wav(path)
    if w.size == 0:
        return None
    peak = float(np.max(np.abs(w)))
    clipped = float(np.mean(np.abs(w) >= 0.999))
    s0, s1 = C.speech_span(w, sr)
    dur = w.size / sr
    head_tail = max(0.0, (s0 + max(dur - s1, 0.0)) / max(dur, 1e-6))
    dc = float(abs(np.mean(w)))
    return {"peak": round(peak, 4), "clipped_ratio": round(clipped, 6),
            "head_tail_silence_ratio": round(head_tail, 4),
            "dc_offset": round(dc, 6), "duration_s": round(dur, 3)}


def audio_score(p):
    clip_pen = min(1.0, p["clipped_ratio"] / 0.01)
    sil_pen = min(1.0, max(0.0, p["head_tail_silence_ratio"] - 0.25) / 0.5)
    return clip01(1.0 - clip_pen - sil_pen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--configs", default="")
    args = ap.parse_args()

    mp = os.path.join(C.E, "results", f"metrics_{args.phase}.json")
    if not os.path.isfile(mp):
        raise SystemExit(f"missing {mp}; run mt_metrics.py first")
    metrics = json.load(open(mp, encoding="utf-8"))
    if args.configs:
        want = [x.strip() for x in args.configs.split(",")]
        metrics = [m for m in metrics if m["config_id"] in want]

    out = []
    for m in metrics:
        cid = m["config_id"]
        row = {"config_id": cid, "family": m.get("family"),
               "variant": m.get("variant"), "phase": args.phase,
               "n_ok": m.get("n_ok"), "success_rate": m.get("success_rate")}
        if not m.get("n_ok"):
            row["status"] = "no_successful_generations"
            out.append(row)
            continue

        # per-item metrics for audio proxies
        per = os.path.join(C.E, "results", args.phase, cid + ".metrics.jsonl")
        items = [json.loads(l) for l in open(per, encoding="utf-8") if l.strip()]
        res = os.path.join(C.E, "results", args.phase, cid + ".jsonl")
        wavmap = {}
        for l in open(res, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("status") == "ok":
                wavmap[(r["text_id"], r["voice_id"])] = r["out_wav"]

        a_scores, proxies = [], []
        for it in items:
            wv = wavmap.get((it["text_id"], it["voice_id"]))
            if not wv or not os.path.isfile(wv):
                continue
            p = audio_proxies(wv)
            if p is None:
                continue
            proxies.append(p)
            a_scores.append(audio_score(p))

        cer = m.get("mean_cer")
        sim = m.get("mean_spk_sim")
        sub = {}
        sub["content_pronunciation"] = (
            clip01(1 - cer / CER_BAD) if cer is not None else None)
        sub["voice_consistency"] = (
            clip01((sim - SIM_LO) / (SIM_HI - SIM_LO)) if sim is not None else None)
        sub["audio_quality"] = (float(np.mean(a_scores)) if a_scores else None)

        row["raw"] = {"mean_cer": cer, "mean_spk_sim": sim,
                      "mean_chars_per_min": m.get("mean_chars_per_min"),
                      "mean_rtf": m.get("mean_rtf"),
                      "max_peak_vram_gib": m.get("max_peak_vram_gib"),
                      "fits_24gb": m.get("fits_24gb"),
                      "cer_status": m.get("cer_status")}
        row["audio_proxies_mean"] = {
            k: round(float(np.mean([p[k] for p in proxies])), 6)
            for k in ("clipped_ratio", "head_tail_silence_ratio", "dc_offset", "peak")
        } if proxies else None
        row["trackA_subscores"] = sub
        row["trackA_not_tested"] = sorted(set(TRACK_A_WEIGHTS) - MEASURABLE_A)
        row["trackB_not_tested"] = sorted(set(TRACK_B_WEIGHTS) - MEASURABLE_B)

        # partial objective score over measurable dims ONLY, renormalized,
        # with explicit coverage so it is never read as a full Track A score.
        avail = {k: v for k, v in sub.items() if v is not None}
        wsum = sum(TRACK_A_WEIGHTS[k] for k in avail)
        row["trackA_objective_partial"] = (
            round(sum(TRACK_A_WEIGHTS[k] * v for k, v in avail.items()) / wsum, 4)
            if wsum else None)
        row["trackA_weight_coverage"] = round(wsum, 4)
        row["trackA_full_score"] = None  # requires blind listening + P3 stability
        row["trackB_full_score"] = None  # requires P2 control sweeps
        out.append(row)
        C.log(f"[{cid}] cer={cer} sim={sim} "
              f"A_sub={ {k: (round(v,3) if v is not None else None) for k,v in sub.items()} } "
              f"A_partial={row['trackA_objective_partial']} "
              f"coverage={row['trackA_weight_coverage']}")

    sp = os.path.join(C.E, "results", f"score_{args.phase}.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump({"normalization": {
            "cer_bad_threshold": CER_BAD, "sim_lo": SIM_LO, "sim_hi": SIM_HI,
            "note": "project-selected thresholds, not model-official metrics"},
            "trackA_weights": TRACK_A_WEIGHTS, "trackB_weights": TRACK_B_WEIGHTS,
            "configs": out}, f, ensure_ascii=False, indent=1)
    C.log(f"score -> {sp}")
    C.log("NOTE: trackA_full_score / trackB_full_score are null by design; "
          "naturalness/intonation need blind listening, stability needs P3, "
          "control dims need P2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())