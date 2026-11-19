#!/usr/bin/env python3
"""Track B v2 scoring, pre-registered in docs/P2_RUBRIC.md.

Reads v1 P2 results (rate/vram), P2b pause pairs, prosody, pronunciation CER,
and an optional MiniCPM/human review summary.  Missing listening is never
imputed: the dimension is scored on its objective half and the weight coverage
records that only half was measured.  trackB_full_score is null unless coverage
is exactly 1.0.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_common as C  # noqa: E402

VALID = ["voxcpm2", "indextts25", "qwen3tts_base",
         "fireredtts3_base", "fireredtts3_instruct"]
W = {"speaking_rate_control": 0.25, "pause_control": 0.20,
     "intonation_control": 0.15, "emotion_control": 0.15,
     "pronunciation_correction": 0.15, "vram_efficiency": 0.10}


def clip01(x):
    return float(max(0.0, min(1.0, x)))


def load_jsonl(p):
    if not os.path.isfile(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_json(p, default=None):
    if not os.path.isfile(p):
        return default
    return json.load(open(p, encoding="utf-8"))


def rate_score(cid):
    rows = [r for r in load_jsonl(os.path.join(C.E, "results", "P2", cid + ".jsonl"))
            if r.get("sub") == "rate" and r.get("status") == "ok"
            and r.get("control_mode") == "native"
            and r.get("rate_rel_err_pct") is not None]
    if not rows:
        return None, {"reason": "no native rate items"}
    errs = [r["rate_rel_err_pct"] for r in rows]
    mean = float(np.mean(errs))
    return clip01(1 - (mean - 5) / 20), {
        "n_native": len(rows), "mean_rel_err_pct": round(mean, 2),
        "within_5pct": round(sum(1 for e in errs if e <= 5) / len(errs), 4)}


def vram_score(cid):
    rows = [r for r in load_jsonl(os.path.join(C.E, "results", "P2", cid + ".jsonl"))
            if r.get("status") == "ok" and r.get("peak_vram_gib") is not None]
    if not rows:
        return None, {"reason": "no vram rows"}
    peak = max(r["peak_vram_gib"] for r in rows)
    rtfs = [r["rtf"] for r in rows if r.get("rtf") is not None]
    return (1.0 if peak <= 24.0 else 0.0), {
        "max_peak_vram_gib": round(peak, 3),
        "mean_rtf": round(float(np.mean(rtfs)), 4) if rtfs else None,
        "fits_24gb": peak <= 24.0,
        "note": "max_memory_allocated is a lower bound; NVML unavailable"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/score_trackB_v2.json")
    args = ap.parse_args()
    pause = load_json(os.path.join(C.E, "results", "P2b", "eval", "pause_pairs.json"), {})
    pros = load_json(os.path.join(C.E, "results", "P2b", "eval", "prosody.json"), {})
    review = load_json(os.path.join(C.E, "results", "P2b", "review", "summary.json"), {})
    calib_ok = bool(review.get("calibration_pass"))
    calib_note = review.get("calibration_note", "review summary missing")

    out = []
    for cid in VALID:
        row = {"config_id": cid, "family": C.cfg_by_id(cid)["family"],
               "variant": C.cfg_by_id(cid)["variant"]}
        sub, cov, reasons = {}, 0.0, {}
        # rate
        s, info = rate_score(cid)
        row["rate_detail"] = info
        if s is not None:
            sub["speaking_rate_control"] = round(s, 4)
            cov += W["speaking_rate_control"]
        else:
            reasons["speaking_rate_control"] = info["reason"]
        # pause
        p = (pause.get("configs", {}) or {}).get(cid, {})
        if p.get("status") == "ok":
            eff = p["pause_effect"]
            # MiniCPM's binary pause probe was biased: it said "有" for every
            # real pause clip, including the none setting (VoxCPM2/FireRed).
            # Do not turn that into a score; human spot-check is pending.
            listen = None
            sub["pause_control"] = round(eff, 4)
            cov += W["pause_control"] / 2
            reasons["pause_control"] = ("objective half only; MiniCPM pause probe "
                                        "biased to 有, human spot-check pending")
            row["pause_detail"] = {
                "n_triples": p.get("n_triples"), "delta_comma_median_s": p.get("delta_comma_median_s"),
                "effect": eff, "sign_p": p.get("sign_p"), "positive_comma": p.get("positive_comma"),
                "v1_same_text_abs_delta_median_s": p.get("v1_same_text_abs_delta_median_s"),
                "gate_v2_gt_v1_noise": p.get("gate_v2_gt_v1_noise"),
                "listen_norm": listen,
                "gap_stats_P1_MALE": p.get("gap_stats_P1_MALE")}
        else:
            reasons["pause_control"] = p.get("status", "no pause data")
        # intonation
        it = (pros.get("intonation", {}) or {}).get(cid, {})
        if it.get("status") == "ok":
            eff = it.get("effect", 0.0)
            listen = None
            if calib_ok:
                listen = ((review.get("intonation", {}) or {}).get("by_config", {}) or {}).get(cid, {}).get("acc_norm")
            if listen is not None:
                sub["intonation_control"] = round(0.5 * eff + 0.5 * listen, 4)
                cov += W["intonation_control"]
            else:
                sub["intonation_control"] = round(eff, 4)
                cov += W["intonation_control"] / 2
                reasons["intonation_control"] = "objective half only; listening not available" if not calib_ok else "listening half missing"
            row["intonation_detail"] = {
                "n": it.get("n"), "max_eta2": it.get("max_eta2"),
                "max_eta2_feature": it.get("max_eta2_feature"),
                "max_omega2": it.get("max_omega2"), "effect": eff,
                "features": it.get("features"), "listen_acc_norm": listen}
        else:
            reasons["intonation_control"] = it.get("status", "no prosody data")
        # emotion (IndexTTS only)
        em = (pros.get("emotion", {}) or {}).get(cid, {})
        if cid != "indextts25":
            reasons["emotion_control"] = "no_interface: identical emotion text, only random sampling"
        elif em.get("status") == "ok":
            obj = em.get("obj", 0.0)
            listen = None
            if calib_ok:
                listen = ((review.get("emotion", {}) or {}).get("acc_norm"))
            if listen is not None:
                sub["emotion_control"] = round(0.5 * obj + 0.5 * listen, 4)
                cov += W["emotion_control"]
            else:
                sub["emotion_control"] = round(obj, 4)
                cov += W["emotion_control"] / 2
                reasons["emotion_control"] = "objective half only; listening not available" if not calib_ok else "listening half missing"
            row["emotion_detail"] = {"n": em.get("n"), "acc": em.get("acc"),
                                     "obj": obj, "listen_acc_norm": listen,
                                     "confusion": em.get("confusion")}
        else:
            reasons["emotion_control"] = em.get("status", "no emotion data")
        # pronunciation (IndexTTS only)
        if cid != "indextts25":
            reasons["pronunciation_correction"] = "no_interface: no pinyin hint interface"
        else:
            pc = load_json(os.path.join(C.E, "results", "P2b", "eval", "pron_cer.json"), [])
            pos = [r for r in pc if not r.get("negative_control")]
            neg = [r for r in pc if r.get("negative_control")]
            mean_cer = float(np.mean([r["cer"] for r in pos])) if pos else None
            content_ok = 1.0 if (mean_cer is not None and mean_cer <= 0.05) else 0.0
            pr = (review.get("pron", {}) or {})
            after = pr.get("after_acc")
            before = pr.get("before_acc")
            chance = pr.get("chance")
            tone_ok = bool(pr.get("tone_reliable"))
            if calib_ok and tone_ok and None not in (after, before, chance):
                after_norm = clip01((after - chance) / max(1 - chance, 1e-6))
                improve = clip01((after - before) / max(1 - before, 0.01))
                score = 0.3 * content_ok + 0.4 * after_norm + 0.3 * improve
                sub["pronunciation_correction"] = round(score, 4)
                cov += W["pronunciation_correction"]
                reasons.pop("pronunciation_correction", None)
            else:
                sub["pronunciation_correction"] = round(0.3 * content_ok, 4)
                cov += W["pronunciation_correction"] * 0.3
                reasons["pronunciation_correction"] = (
                    "content CER only; tone listening not available" if not calib_ok
                    else pr.get("tone_note", "tone answers missing"))
            row["pronunciation_detail"] = {
                "n_pos": len(pos), "n_neg": len(neg), "mean_cer_pos": round(mean_cer, 4) if mean_cer is not None else None,
                "content_ok": content_ok, "after_acc": after, "before_acc": before,
                "chance": chance, "tone_reliable": tone_ok,
                "raw_after_acc": pr.get("raw_after_acc"),
                "raw_before_acc": pr.get("raw_before_acc"),
                "raw_chance": pr.get("raw_chance"),
                "neg_detects_wrong": pr.get("neg_detects_wrong"),
                "calibration_pass": calib_ok}
        # vram
        s, info = vram_score(cid)
        row["vram_detail"] = info
        if s is not None:
            sub["vram_efficiency"] = round(s, 4)
            cov += W["vram_efficiency"]
        else:
            reasons["vram_efficiency"] = info["reason"]

        row["trackB_subscores"] = sub
        row["trackB_weight_coverage"] = round(cov, 4)
        avail = {k: v for k, v in sub.items() if v is not None}
        wsum = sum(W[k] for k in avail)
        row["trackB_objective_partial"] = (
            round(sum(W[k] * v for k, v in avail.items()) / wsum, 4) if wsum else None)
        row["trackB_full_score"] = (
            round(sum(W[k] * v for k, v in avail.items()), 4)
            if abs(cov - 1.0) < 1e-9 and len(avail) == len(W) else None)
        row["rankable"] = row["trackB_full_score"] is not None
        row["not_tested_reason"] = reasons
        out.append(row)
        print(f"[{cid}] cov={row['trackB_weight_coverage']} "
              f"sub={ {k: v for k, v in sub.items()} } "
              f"full={row['trackB_full_score']} rankable={row['rankable']}")

    doc = {"rubric": "docs/P2_RUBRIC.md", "rubric_version": "P2-RUBRIC-2026-11-13",
           "weights": W, "calibration_pass": calib_ok,
           "calibration_note": calib_note,
           "note": "unsupported/no_interface is never imputed as 0; full score "
                   "only when weight coverage is 1.0",
           "configs": out}
    p = os.path.join(C.E, args.out)
    json.dump(doc, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
