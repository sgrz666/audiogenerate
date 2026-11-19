#!/usr/bin/env python3
"""Track B (control + engineering) scoring from P2/P3 results.

Pre-registered normalization (project decisions, not model-official metrics):
  rate_score  = clip(1 - (mean_rel_err_pct - 5)/(25 - 5), 0, 1)
                <=5% relative error -> 1.0 (task book target), >=25% -> 0.
                Computed ONLY over items with control_mode == native; a
                text_fallback result is never counted as native control.
  efficiency  = reported raw (mean RTF, max peak VRAM, fits_24gb), not folded
                into a quality score.

Dimensions needing human listening (语气/情绪) or a numeric pause target that
P2 does not yet define (停顿) are not_tested. 读音纠错 needs CER on the P2
pronunciation items (ASR), computed separately; marked not_tested until present.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_common as C  # noqa: E402

TRACK_B_WEIGHTS = {"speaking_rate_control": 0.25, "pause_control": 0.20,
                   "intonation_control": 0.15, "emotion_control": 0.15,
                   "pronunciation_correction": 0.15, "vram_efficiency": 0.10}
MEASURABLE = {"speaking_rate_control", "vram_efficiency"}


def clip01(x):
    return float(max(0.0, min(1.0, x)))


def load_rows(phase, cid):
    p = C.result_path(cid, phase)
    if not os.path.isfile(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="")
    args = ap.parse_args()
    cfgs = C.CONFIGS
    if args.configs:
        want = [x.strip() for x in args.configs.split(",")]
        cfgs = [c for c in cfgs if c["config_id"] in want]

    out = []
    for c in cfgs:
        cid = c["config_id"]
        p2 = load_rows("P2", cid)
        p3 = load_rows("P3", cid)
        row = {"config_id": cid, "family": c["family"], "variant": c["variant"]}
        if not p2:
            row["status"] = "not_tested"
            out.append(row)
            C.log(f"[{cid}] P2 not_tested")
            continue

        ok = [r for r in p2 if r.get("status") == "ok"]
        # rate control: native items only
        rate_native = [r for r in ok if r.get("sub") == "rate"
                       and r.get("control_mode") == "native"
                       and r.get("rate_rel_err_pct") is not None]
        rate_fallback = [r for r in ok if r.get("sub") == "rate"
                         and r.get("control_mode") != "native"]
        sub = {}
        if rate_native:
            errs = [r["rate_rel_err_pct"] for r in rate_native]
            mean_err = float(np.mean(errs))
            sub["speaking_rate_control"] = clip01(1 - (mean_err - 5) / 20)
            row["rate_mean_rel_err_pct"] = round(mean_err, 2)
            row["rate_within_5pct"] = round(
                sum(1 for e in errs if e <= 5) / len(errs), 4)
            row["rate_n_native"] = len(rate_native)
        else:
            sub["speaking_rate_control"] = None
            row["rate_n_native"] = 0
        row["rate_n_text_fallback"] = len(rate_fallback)

        # per-sub control_mode census (native vs fallback vs failed)
        census = defaultdict(lambda: defaultdict(int))
        for r in p2:
            census[r.get("sub", "?")][r.get("control_mode", r.get("status"))] += 1
        row["control_census"] = {k: dict(v) for k, v in census.items()}

        # efficiency from P2 (and P3 if present)
        rtfs = [r["rtf"] for r in ok if r.get("rtf") is not None]
        vrams = [r["peak_vram_gib"] for r in ok if r.get("peak_vram_gib") is not None]
        row["mean_rtf"] = round(float(np.mean(rtfs)), 4) if rtfs else None
        row["max_peak_vram_gib"] = round(max(vrams), 3) if vrams else None
        row["fits_24gb"] = (max(vrams) <= 24.0) if vrams else None

        # stability from P3 summaries
        p3s = [r for r in p3 if r.get("status") == "summary"]
        if p3s:
            row["stability_mean_cpm_cv_pct"] = round(
                float(np.mean([r["cpm_cv_pct"] for r in p3s])), 2)
            row["stability_n"] = len(p3s)

        row["trackB_subscores"] = sub
        row["trackB_not_tested"] = sorted(set(TRACK_B_WEIGHTS) - MEASURABLE)
        avail = {k: v for k, v in sub.items() if v is not None}
        # efficiency is reported raw, not scored; partial covers rate only
        wsum = sum(TRACK_B_WEIGHTS[k] for k in avail)
        row["trackB_objective_partial"] = (
            round(sum(TRACK_B_WEIGHTS[k] * v for k, v in avail.items()) / wsum, 4)
            if wsum else None)
        row["trackB_weight_coverage"] = round(wsum, 4)
        row["trackB_full_score"] = None
        out.append(row)
        C.log(f"[{cid}] rate_native={row['rate_n_native']} "
              f"err={row.get('rate_mean_rel_err_pct')} "
              f"B_partial={row['trackB_objective_partial']} "
              f"cov={row['trackB_weight_coverage']} vram={row['max_peak_vram_gib']}")

    sp = os.path.join(C.E, "results", "score_trackB.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump({"normalization": {"rate_err_good": 5, "rate_err_bad": 25,
                                      "note": "project thresholds; native-only"},
                   "trackB_weights": TRACK_B_WEIGHTS, "configs": out},
                  f, ensure_ascii=False, indent=1)
    C.log(f"trackB score -> {sp}")
    C.log("NOTE: 语气/情绪 need blind listening; 停顿 lacks a numeric target; "
          "读音纠错 needs P2 CER. All marked not_tested, not imputed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())