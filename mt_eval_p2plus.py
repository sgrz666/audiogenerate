#!/usr/bin/env python3
"""Evaluate P2plus across all 6 sub-experiments in two env-safe stages:
  --stage asr_spk           (run in envs/voxcpm/bin/python: SenseVoice ASR + CAM++ sim)
  --stage acoustics_report  (run in envs/indextts/bin/python: librosa F0/pauses + summary.json + REPORT.md + index.html)
"""
import argparse
import html
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_common as C

P2PLUS_DIR = os.path.join(C.E, "audio", "P2plus")
MANIFEST_PATH = os.path.join(P2PLUS_DIR, "manifest.json")
RESULTS_DIR = os.path.join(C.E, "results", "P2plus")
CONFIGS = ["voxcpm2", "voxcpm2_ctrl", "indextts25"]


def load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(p):
    if not os.path.isfile(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------- Stage 1: ASR + Speaker Similarity ----------------

def run_asr_spk(device):
    import mt_metrics as M
    from funasr import AutoModel

    asr_dir = next(
        d for d in [
            os.path.join(C.E, "asr", "SenseVoiceSmall"),
            "/data/shangguanrz/projectsss/speakgenerate/model/asr/SenseVoiceSmall",
        ] if os.path.isdir(d)
    )
    C.log(f"[eval:asr_spk] Loading SenseVoiceSmall from {asr_dir} & CAM++ on {device}")
    asr = AutoModel(model=asr_dir, disable_update=True, device=device)
    enc = M.SpeakerEncoder(device=device)

    manifest = load_manifest()
    ref_embs = {}
    gt_embs = {}
    for m in manifest:
        lvl = int(m["level"])
        ref_embs[lvl] = enc.emb(m["ref_wav"])
        gt_embs[lvl] = enc.emb(m["target_gt_wav"])

    for cid in CONFIGS:
        raw_path = os.path.join(RESULTS_DIR, f"{cid}.jsonl")
        rows = load_jsonl(raw_path)
        if not rows:
            C.log(f"[eval:asr_spk] Missing {raw_path}, skipping")
            continue
        C.log(f"[eval:asr_spk] Evaluating {cid} ({len(rows)} rows)...")
        for idx, r in enumerate(rows, 1):
            if r.get("status") != "ok" or not os.path.isfile(r.get("out_wav", "")):
                continue
            lvl = int(r["level"])
            try:
                emb = enc.emb(r["out_wav"])
                r["spk_sim_to_ref"] = round(float(M.cos(emb, ref_embs[lvl])), 4)
                r["spk_sim_to_gt"] = round(float(M.cos(emb, gt_embs[lvl])), 4)
            except Exception as ex:
                r["spk_sim_error"] = f"{type(ex).__name__}: {str(ex)[:120]}"
            try:
                res = asr.generate(input=r["out_wav"], cache={}, language="zh",
                                   use_itn=False, batch_size_s=60)
                hyp = res[0]["text"] if res else ""
                hyp = re.sub(r"<\|[^|]*\|>", "", hyp)
                ref_t = M.norm_text(r["text"], t2s=True)
                hyp_t = M.norm_text(hyp, t2s=True)
                d = M.edit_distance(ref_t, hyp_t)
                r["asr_hyp"] = hyp
                r["asr_hyp_norm"] = hyp_t
                r["cer"] = round(float(d) / max(len(ref_t), 1), 4)
            except Exception as ex:
                r["cer_error"] = f"{type(ex).__name__}: {str(ex)[:120]}"
            if idx % 30 == 0 or idx == len(rows):
                C.log(f"[eval:asr_spk] {cid} {idx}/{len(rows)}")
        out_path = os.path.join(RESULTS_DIR, f"{cid}.eval.jsonl")
        save_jsonl(out_path, rows)
        C.log(f"[eval:asr_spk] Wrote {out_path}")

    del asr, enc
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


# ---------------- Stage 2: Acoustics + Report + HTML ----------------

def pause_gaps(wav, sr, top_db=30, min_gap=0.08):
    import librosa
    iv = librosa.effects.split(wav, top_db=top_db, frame_length=400, hop_length=160)
    gaps = []
    for i in range(1, len(iv)):
        t0, t1 = iv[i - 1][1] / sr, iv[i][0] / sr
        if t1 - t0 >= min_gap:
            gaps.append({"start": round(t0, 3), "end": round(t1, 3), "dur": round(t1 - t0, 3)})
    return gaps


def f0_features(path):
    import librosa
    from scipy.ndimage import median_filter
    w, sr = C.read_wav(path)
    if len(w) < sr * 0.15:
        return None
    f0, vf, _ = librosa.pyin(w, fmin=70, fmax=400, sr=sr,
                             frame_length=1024, hop_length=160)
    f0 = np.asarray(f0, float)
    vf = np.asarray(vf, bool)
    idx = np.where(vf)[0]
    if len(idx) > 5:
        med = median_filter(f0[idx], size=5, mode="nearest")
        st = 12 * np.log2(f0[idx] / 55.0)
        stm = 12 * np.log2(med / 55.0)
        bad = np.abs(st - stm) > 6
        st[bad] = stm[bad]
        f0[idx] = 55.0 * 2 ** (st / 12)
    v = f0[vf & np.isfinite(f0)]
    if len(v) < 5:
        return None
    st = 12 * np.log2(v / 55.0)
    idx = np.where(vf & np.isfinite(f0))[0]
    end = idx[-1]
    last = [i for i in idx if i > end - 12]
    prev = [i for i in idx if end - 42 < i <= end - 12]
    if len(last) > 3 and len(prev) > 3:
        final_delta = float(np.mean(12 * np.log2(f0[last] / 55.0))
                            - np.mean(12 * np.log2(f0[prev] / 55.0)))
    else:
        final_delta = float("nan")
    rms = librosa.feature.rms(y=w, frame_length=400, hop_length=160)[0]
    db = 20 * np.log10(rms + 1e-8)
    return {
        "f0_mean": float(np.mean(v)),
        "st_std": float(np.std(st)),
        "st_range": float(np.percentile(st, 95) - np.percentile(st, 5)),
        "final_delta": final_delta,
        "dur": float(len(w)) / float(sr),
        "db": float(np.mean(db)),
        "db_std": float(np.std(db)),
    }


def eta2(groups, vals):
    vals = np.asarray(vals, float)
    grand = vals.mean()
    ssb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups if len(g) > 0)
    sst = ((vals - grand) ** 2).sum()
    return float(ssb / sst) if sst > 0 else 0.0


def omega2(groups, vals):
    vals = np.asarray(vals, float)
    k = len(groups)
    n = len(vals)
    if n <= k:
        return 0.0
    grand = vals.mean()
    ssb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups if len(g) > 0)
    sst = ((vals - grand) ** 2).sum()
    msw = (sst - ssb) / (n - k)
    return float((ssb - (k - 1) * msw) / (sst + msw)) if (sst + msw) > 0 else 0.0


def perm_p(groups, vals, n_perm=5000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, float)
    sizes = [len(g) for g in groups]
    obs = eta2(groups, vals)
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(vals)
        gs = []
        i = 0
        for s in sizes:
            gs.append(p[i:i + s])
            i += s
        if eta2(gs, p) >= obs - 1e-12:
            cnt += 1
    return float((cnt + 1) / (n_perm + 1))


def binom_two_sided(k, n):
    if n <= 0:
        return 1.0
    if k > n / 2:
        k = n - k
    p = sum(math.comb(n, i) * 0.5 ** n for i in range(k + 1))
    return min(1.0, 2.0 * p)


def span_per_char(r):
    nch = r.get("norm_chars") or 0
    sp = r.get("speech_span_s") or 0.0
    if not nch or sp <= 0.1:
        return None
    return float(sp) / float(nch)


def pearson_r(xs, ys):
    if len(xs) < 2:
        return None
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return round(float(np.corrcoef(x, y)[0, 1]), 4)


def spearman_r(xs, ys):
    if len(xs) < 2:
        return None
    rx = np.argsort(np.argsort(xs))
    ry = np.argsort(np.argsort(ys))
    return pearson_r(rx, ry)


def safe_mean(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return round(float(np.mean(v)), 4) if v else None


def eval_baseline(rows_by_cfg, gt_by_lvl):
    out = {}
    for cid in CONFIGS:
        rows = sorted([r for r in rows_by_cfg[cid] if r["sub"] == "baseline" and r["status"] == "ok"],
                      key=lambda x: x["level"])
        per_lvl = []
        for r in rows:
            lvl = int(r["level"])
            gt_cpm = gt_by_lvl[lvl]["target_metrics"]["actual_cpm"]
            per_lvl.append({
                "level": lvl,
                "gt_cpm": gt_cpm,
                "actual_cpm": r["actual_cpm"],
                "delta_cpm": round(r["actual_cpm"] - gt_cpm, 2),
                "cer": r.get("cer"),
                "spk_sim_to_ref": r.get("spk_sim_to_ref"),
                "spk_sim_to_gt": r.get("spk_sim_to_gt"),
                "dur_s": r["dur_s"],
                "rtf": r["rtf"],
                "vram_gib": r["vram_gib"],
            })
        out[cid] = {
            "n": len(rows),
            "mean_cpm": safe_mean([x["actual_cpm"] for x in per_lvl]),
            "mean_abs_delta_cpm": safe_mean([abs(x["delta_cpm"]) for x in per_lvl]),
            "mean_cer": safe_mean([x["cer"] for x in per_lvl]),
            "mean_spk_sim_to_ref": safe_mean([x["spk_sim_to_ref"] for x in per_lvl]),
            "mean_spk_sim_to_gt": safe_mean([x["spk_sim_to_gt"] for x in per_lvl]),
            "mean_rtf": safe_mean([x["rtf"] for x in per_lvl]),
            "max_vram_gib": max([x["vram_gib"] for x in per_lvl], default=None),
            "per_level": per_lvl,
        }
    return out


def eval_rate(rows_by_cfg, gt_by_lvl):
    out = {}
    for cid in CONFIGS:
        rows = [r for r in rows_by_cfg[cid] if r["sub"] == "rate" and r["status"] == "ok"]
        tgts = [r["target_cpm"] for r in rows]
        acts = [r["actual_cpm"] for r in rows]
        errs = [abs(a - t) for a, t in zip(acts, tgts)]
        gt_rows = [r for r in rows if r.get("rate_tag") == "gt"]
        gt_abs_errs = [abs(r["actual_cpm"] - gt_by_lvl[int(r["level"])]["target_metrics"]["actual_cpm"]) for r in gt_rows]
        gt_rel_errs = [
            abs(r["actual_cpm"] - gt_by_lvl[int(r["level"])]["target_metrics"]["actual_cpm"])
            / gt_by_lvl[int(r["level"])]["target_metrics"]["actual_cpm"] * 100.0
            for r in gt_rows
        ]
        by_lvl = defaultdict(list)
        for r in rows:
            by_lvl[int(r["level"])].append(r)
        lvl_corrs = []
        monotonic_ok = 0
        for lvl in range(1, 7):
            grid = sorted([x for x in by_lvl[lvl] if x.get("rate_tag") != "gt"], key=lambda x: x["target_cpm"])
            if len(grid) >= 3:
                xs = [x["target_cpm"] for x in grid]
                ys = [x["actual_cpm"] for x in grid]
                lvl_corrs.append(pearson_r(xs, ys))
                if all(ys[i] < ys[i + 1] for i in range(len(ys) - 1)):
                    monotonic_ok += 1
        out[cid] = {
            "n": len(rows),
            "control_mode": rows[0]["control_mode"] if rows else None,
            "pearson_r_all": pearson_r(tgts, acts),
            "spearman_rho_all": spearman_r(tgts, acts),
            "mae_cpm_all": safe_mean(errs),
            "gt_match_mae_cpm": safe_mean(gt_abs_errs),
            "gt_match_mape_pct": safe_mean(gt_rel_errs),
            "per_level_pearson_mean": safe_mean(lvl_corrs),
            "strictly_monotonic_levels": f"{monotonic_ok}/6",
            "mean_cer": safe_mean([r.get("cer") for r in rows]),
            "mean_spk_sim_to_ref": safe_mean([r.get("spk_sim_to_ref") for r in rows]),
            "mean_spk_sim_to_gt": safe_mean([r.get("spk_sim_to_gt") for r in rows]),
        }
    return out


def eval_pause(rows_by_cfg):
    SETTINGS = ["none", "comma", "period"]
    out = {}
    for cid in CONFIGS:
        rows = [r for r in rows_by_cfg[cid] if r["sub"] == "pause" and r["status"] == "ok"]
        by_lvl = defaultdict(lambda: defaultdict(list))
        for r in rows:
            w, sr = C.read_wav(r["out_wav"])
            gaps = pause_gaps(w, sr)
            r["pause_gap_count"] = len(gaps)
            r["pause_gap_total_s"] = round(sum(g["dur"] for g in gaps), 3)
            r["pause_gap_max_s"] = max([g["dur"] for g in gaps], default=0.0)
            by_lvl[int(r["level"])][r["pause_setting"]].append(r)

        per_lvl = []
        d_comma, d_period, d_per_none = [], [], []
        gap_means = {s: [] for s in SETTINGS}
        for lvl in range(1, 7):
            d = by_lvl[lvl]
            if not all(s in d and d[s] for s in SETTINGS):
                continue
            med_spc = {}
            med_gap_dur = {}
            med_gap_cnt = {}
            for s in SETTINGS:
                spcs = [span_per_char(x) for x in d[s]]
                spcs = [v for v in spcs if v is not None]
                med_spc[s] = float(np.median(spcs))
                med_gap_dur[s] = float(np.median([x["pause_gap_total_s"] for x in d[s]]))
                med_gap_cnt[s] = float(np.median([x["pause_gap_count"] for x in d[s]]))
                gap_means[s].append(med_gap_dur[s])
            nch = d["none"][0]["norm_chars"]
            dc = (med_spc["comma"] - med_spc["none"]) * nch
            dp = (med_spc["period"] - med_spc["comma"]) * nch
            dpn = (med_spc["period"] - med_spc["none"]) * nch
            d_comma.append(dc)
            d_period.append(dp)
            d_per_none.append(dpn)
            per_lvl.append({
                "level": lvl,
                "nch": nch,
                "span_per_char": {s: round(med_spc[s], 5) for s in SETTINGS},
                "delta_comma_s": round(dc, 4),
                "delta_period_vs_comma_s": round(dp, 4),
                "delta_period_vs_none_s": round(dpn, 4),
                "median_internal_gap_dur_s": {s: round(med_gap_dur[s], 3) for s in SETTINGS},
                "median_internal_gap_cnt": {s: round(med_gap_cnt[s], 1) for s in SETTINGS},
            })

        pos_c = sum(1 for x in d_comma if x > 0)
        pos_pn = sum(1 for x in d_per_none if x > 0)
        n = len(d_comma)
        out[cid] = {
            "n_rows": len(rows),
            "n_levels": n,
            "delta_comma_median_s": round(float(np.median(d_comma)), 4) if d_comma else None,
            "delta_period_vs_comma_median_s": round(float(np.median(d_period)), 4) if d_period else None,
            "delta_period_vs_none_median_s": round(float(np.median(d_per_none)), 4) if d_per_none else None,
            "positive_comma": f"{pos_c}/{n}",
            "sign_p_comma": round(binom_two_sided(pos_c, n), 4) if n else None,
            "positive_period_vs_none": f"{pos_pn}/{n}",
            "sign_p_period_vs_none": round(binom_two_sided(pos_pn, n), 4) if n else None,
            "mean_internal_gap_dur_s": {s: round(float(np.mean(gap_means[s])), 3) for s in SETTINGS if gap_means[s]},
            "mean_cer": safe_mean([r.get("cer") for r in rows]),
            "mean_spk_sim_to_ref": safe_mean([r.get("spk_sim_to_ref") for r in rows]),
            "per_level": per_lvl,
        }
    return out


def eval_intonation(rows_by_cfg, gt_by_lvl):
    TYPES = ["declarative", "interrogative", "exclamatory"]
    gt_feats = {}
    for lvl, m in gt_by_lvl.items():
        gt_feats[lvl] = f0_features(m["target_gt_wav"])

    out = {"data_gt_f0": {str(lvl): ({k: round(v, 3) if np.isfinite(v) else None for k, v in f.items()} if f else None) for lvl, f in gt_feats.items()}}
    for cid in CONFIGS:
        rows = [r for r in rows_by_cfg[cid] if r["sub"] == "intonation" and r["status"] == "ok"]
        data = defaultdict(list)
        feats = []
        for r in rows:
            f = f0_features(r["out_wav"])
            if f is None:
                continue
            r["f0_features"] = {k: round(v, 3) if np.isfinite(v) else None for k, v in f.items()}
            f["type"] = r["intonation_type"]
            f["level"] = int(r["level"])
            feats.append(f)
            data[r["intonation_type"]].append(f)
        res = {"n": len(feats), "features": {}}
        for key in ["f0_mean", "st_std", "st_range", "final_delta"]:
            groups = [[f[key] for f in data[t] if np.isfinite(f[key])] for t in TYPES]
            vals = [x for g in groups for x in g]
            if len(vals) < 6 or any(len(g) < 2 for g in groups):
                continue
            e2 = eta2(groups, vals)
            w2 = omega2(groups, vals)
            p = perm_p(groups, vals)
            res["features"][key] = {
                "means": {t: round(float(np.mean(g)), 3) for t, g in zip(TYPES, groups)},
                "eta2": round(e2, 4),
                "omega2": round(w2, 4),
                "perm_p": round(p, 4),
            }
        if res["features"]:
            max_key = max(res["features"], key=lambda k: res["features"][k]["eta2"])
            res["max_eta2_feature"] = max_key
            res["max_eta2"] = res["features"][max_key]["eta2"]
            res["max_omega2"] = res["features"][max_key]["omega2"]
            res["effect"] = round(float(max(0.0, min(1.0, res["max_omega2"] / 0.12))), 4)
        res["mean_cer"] = safe_mean([r.get("cer") for r in rows])
        res["mean_spk_sim_to_ref"] = safe_mean([r.get("spk_sim_to_ref") for r in rows])
        out[cid] = res
    return out


def eval_emotion(rows_by_cfg):
    EMOS = ["calm", "happy", "sad", "angry"]
    FEAT_KEYS = ["f0_mean", "st_std", "st_range", "db", "db_std", "dur"]
    out = {}
    for cid in CONFIGS:
        rows = [r for r in rows_by_cfg[cid] if r["sub"] == "emotion" and r["status"] == "ok"]
        feats = []
        for r in rows:
            f = f0_features(r["out_wav"])
            if f is None:
                continue
            r["f0_features"] = {k: round(v, 3) if np.isfinite(v) else None for k, v in f.items()}
            f["emo"] = r["emotion"]
            feats.append(f)
        if len(feats) < 12:
            out[cid] = {"status": "insufficient", "n": len(feats)}
            continue
        X = np.array([[f[k] for k in FEAT_KEYS] for f in feats], float)
        for j in range(X.shape[1]):
            col = X[:, j]
            med = np.nanmedian(col)
            col[~np.isfinite(col)] = med
            X[:, j] = col
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        y = np.array([f["emo"] for f in feats])
        correct = 0
        conf = defaultdict(int)
        for i in range(len(y)):
            tr = np.arange(len(y)) != i
            cents = {e: X[tr & (y == e)].mean(0) for e in EMOS if (tr & (y == e)).any()}
            pred = min(cents, key=lambda e: np.linalg.norm(X[i] - cents[e]))
            correct += int(pred == y[i])
            conf[f"{y[i]}->{pred}"] += 1
        acc = correct / len(y)
        out[cid] = {
            "status": "ok",
            "control_mode": rows[0]["control_mode"] if rows else None,
            "n": len(y),
            "loo_acc": round(acc, 4),
            "chance": 0.25,
            "obj": round(float(max(0.0, min(1.0, (acc - 0.25) / 0.75))), 4),
            "confusion": dict(conf),
            "per_emotion_means": {
                e: {k: round(float(np.mean([f[k] for f in feats if f["emo"] == e])), 3)
                    for k in ["f0_mean", "st_std", "st_range", "db", "dur"]}
                for e in EMOS
            },
            "mean_cer": safe_mean([r.get("cer") for r in rows]),
            "mean_spk_sim_to_ref": safe_mean([r.get("spk_sim_to_ref") for r in rows]),
            "mean_spk_sim_to_gt": safe_mean([r.get("spk_sim_to_gt") for r in rows]),
        }
    return out


def eval_pronunciation(rows_by_cfg):
    out = {}
    for cid in CONFIGS:
        rows = [r for r in rows_by_cfg[cid] if r["sub"] == "pronunciation" and r["status"] == "ok"]
        pos = [r for r in rows if not r.get("negative_control")]
        neg = [r for r in rows if r.get("negative_control")]
        hsk_pos = [r for r in pos if r.get("poly_group") == "hsk_dialogue"]
        pr_pos = [r for r in pos if r.get("poly_group") in ("standard_pr", "pr_benchmark")]

        pos_by_key = {}
        for r in pos:
            pos_by_key[(int(r["level"]), r["poly_id"])] = r
        neg_diff = 0
        neg_pairs = 0
        for r in neg:
            base_poly = r["poly_id"].replace("_NEG", "")
            p = pos_by_key.get((int(r["level"]), base_poly))
            if p and p.get("asr_hyp_norm") is not None and r.get("asr_hyp_norm") is not None:
                neg_pairs += 1
                if p["asr_hyp_norm"] != r["asr_hyp_norm"]:
                    neg_diff += 1

        out[cid] = {
            "control_mode": rows[0]["control_mode"] if rows else None,
            "n_pos": len(pos),
            "n_neg": len(neg),
            "pos_cer_mean": safe_mean([r.get("cer") for r in pos]),
            "pos_zero_cer_rate": round(sum(1 for r in pos if r.get("cer") == 0.0) / max(1, len(pos)), 4),
            "hsk_pos_cer_mean": safe_mean([r.get("cer") for r in hsk_pos]),
            "pr_pos_cer_mean": safe_mean([r.get("cer") for r in pr_pos]),
            "neg_cer_mean": safe_mean([r.get("cer") for r in neg]) if neg else None,
            "neg_asr_diff_vs_pos": f"{neg_diff}/{neg_pairs}" if neg_pairs else None,
            "mean_spk_sim_to_ref": safe_mean([r.get("spk_sim_to_ref") for r in pos]),
            "mean_spk_sim_to_gt": safe_mean([r.get("spk_sim_to_gt") for r in pos]),
        }
    return out


def build_markdown(summary, manifest, rows_by_cfg=None):
    lines = []
    lines.append("# P2plus 真实数据对照全维可控性实验报告")
    lines.append("")
    lines.append("本实验以 `/data/shangguanrz/projectsss/speakgenerate/data/` 中 HSK 1–6 真实女性对话音频作为 **Ground-Truth 对照 (`data_gt`)**，并在同等级题目中选取更长的女性录音作为 **克隆参考 (`refs`)**，对 **VoxCPM2 无指令 (`voxcpm2`)**、**VoxCPM2 官方括号可控克隆 (`voxcpm2_ctrl`)** 与 **IndexTTS-2.5 (`indextts25`)** 进行 P2 全套 6 项可控性评测。")
    lines.append("")
    lines.append("## 0. HSK 1–6 样本与参考音频对照表 (`data_gt` vs `refs`)")
    lines.append("")
    lines.append("| 等级 | 目标对话 (`data_gt`) | 目标文本 | GT时长/语速 | GT CER | 克隆参考 (`refs`) | 参考时长 | REF-GT CAM++相似度 |")
    lines.append("|---:|---|---|---:|---:|---|---:|---:|")
    for m in manifest:
        tm = m["target_metrics"]
        rm = m["ref_metrics"]
        lines.append(
            f"| HSK{m['level']} | `{m['target_sample_id']}` | {m['target_text']} | "
            f"{tm['duration_s']}s / {tm['actual_cpm']} cpm | {tm['cer']} | "
            f"`{m['ref_sample_id']}` | {rm['duration_s']}s | {m['ref_target_spk_sim']} |"
        )
    lines.append("")

    b = summary["01_baseline"]
    lines.append("## 1. `01_默认语速基线` (Natural Rate vs `data_gt`)")
    lines.append("")
    lines.append("| 模型 | 样本数 | 平均默认语速 (cpm) | 相对GT平均\\|ΔCPM\\| | CER | Sim(to Ref) | Sim(to GT) | Mean RTF | Peak VRAM (GiB) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cid in CONFIGS:
        x = b[cid]
        lines.append(
            f"| `{cid}` | {x['n']} | {x['mean_cpm']} | {x['mean_abs_delta_cpm']} | "
            f"{x['mean_cer']} | {x['mean_spk_sim_to_ref']} | {x['mean_spk_sim_to_gt']} | "
            f"{x['mean_rtf']} | {x['max_vram_gib']} |"
        )
    lines.append("")
    lines.append("| 等级 | `data_gt` CPM | `voxcpm2` CPM (Δ) | `indextts25` CPM (Δ) |")
    lines.append("|---:|---:|---:|---:|")
    vox_b = {r["level"]: r for r in b["voxcpm2"]["per_level"]}
    idx_b = {r["level"]: r for r in b["indextts25"]["per_level"]}
    for m in manifest:
        lvl = m["level"]
        vb = vox_b.get(lvl, {})
        ib = idx_b.get(lvl, {})
        vd = vb.get("delta_cpm")
        id_ = ib.get("delta_cpm")
        vds = f"{vd:+}" if vd is not None else "N/A"
        ids = f"{id_:+}" if id_ is not None else "N/A"
        lines.append(
            f"| HSK{lvl} | {m['target_metrics']['actual_cpm']} | "
            f"{vb.get('actual_cpm')} ({vds}) | "
            f"{ib.get('actual_cpm')} ({ids}) |"
        )
    lines.append("")

    rt = summary["02_rate"]
    lines.append("## 2. `02_语速控制` (GT对标 + 160–320 cpm 网格)")
    lines.append("")
    lines.append("- **控制方法对比**：`voxcpm2` 无原生参数（仅默认输出对照）；`voxcpm2_ctrl` 在文本前拼接括号定性语速指令（如 `(语速非常缓慢)` ~ `(语速非常快)`）；`indextts25` 通过 `duration_factor = baseline_cpm / target_cpm` 在 `length_regulator` 层做连续时长缩放。")
    lines.append("")
    lines.append("| 模型 | 控制模式 | Pearson `r` | Spearman `ρ` | 分级均值 `r` | 全部MAE (cpm) | 还原GT MAE / MAPE | 单调递增等级数 | CER | Sim(to Ref) | Sim(to GT) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cid in CONFIGS:
        x = rt[cid]
        lines.append(
            f"| `{cid}` | `{x['control_mode']}` | {x['pearson_r_all']} | {x['spearman_rho_all']} | "
            f"{x.get('per_level_pearson_mean') if x.get('per_level_pearson_mean') is not None else '—'} | "
            f"{x['mae_cpm_all']} | {x['gt_match_mae_cpm']} / {x['gt_match_mape_pct']}% | "
            f"{x['strictly_monotonic_levels']} | {x['mean_cer']} | {x['mean_spk_sim_to_ref']} | {x['mean_spk_sim_to_gt']} |"
        )
    lines.append("")
    if rows_by_cfg:
        rate_map = {
            cid: {(int(r["level"]), r.get("rate_tag")): r for r in rows_by_cfg[cid] if r.get("sub") == "rate" and r.get("status") == "ok"}
            for cid in CONFIGS
        }
        lines.append("### 2.1 还原真题原声语速明细 (`tgt-gt`)")
        lines.append("")
        lines.append("| 等级 | GT语速 (cpm) | `voxcpm2` 实测 (Δ) | `voxcpm2_ctrl` 实测 (Δ / Sim) | `indextts25` 实测 (Δ / `df` / Sim) |")
        lines.append("|---:|---:|---:|---:|---:|")
        for m in manifest:
            lvl = int(m["level"])
            gt_c = float(m["target_metrics"]["actual_cpm"])
            vr = rate_map["voxcpm2"].get((lvl, "gt"), {})
            vcr = rate_map["voxcpm2_ctrl"].get((lvl, "gt"), {})
            ir = rate_map["indextts25"].get((lvl, "gt"), {})
            vc_d = round(vcr["actual_cpm"] - gt_c, 2) if "actual_cpm" in vcr else None
            vr_d = round(vr["actual_cpm"] - gt_c, 2) if "actual_cpm" in vr else None
            ir_d = round(ir["actual_cpm"] - gt_c, 2) if "actual_cpm" in ir else None
            lines.append(
                f"| HSK{lvl} | {gt_c} | {vr.get('actual_cpm')} ({vr_d:+}) | "
                f"{vcr.get('actual_cpm')} ({vc_d:+} / {vcr.get('spk_sim_to_ref')}) | "
                f"{ir.get('actual_cpm')} ({ir_d:+} / df={ir.get('opts', {}).get('duration_factor')} / {ir.get('spk_sim_to_ref')}) |"
            )
        lines.append("")
        lines.append("### 2.2 160–320 cpm 五档网格实测语速 (`voxcpm2_ctrl` vs `indextts25`)")
        lines.append("")
        lines.append("| 等级 | `voxcpm2` 默认 | `voxcpm2_ctrl` (160 / 200 / 240 / 280 / 320) | `indextts25` (160 / 200 / 240 / 280 / 320) |")
        lines.append("|---:|---:|---|---|")
        for m in manifest:
            lvl = int(m["level"])
            vdef = rate_map["voxcpm2"].get((lvl, "160"), {}).get("actual_cpm")
            vc_vals = " / ".join(str(rate_map["voxcpm2_ctrl"].get((lvl, str(t)), {}).get("actual_cpm")) for t in (160, 200, 240, 280, 320))
            idx_vals = " / ".join(str(rate_map["indextts25"].get((lvl, str(t)), {}).get("actual_cpm")) for t in (160, 200, 240, 280, 320))
            lines.append(f"| HSK{lvl} | {vdef} | {vc_vals} | {idx_vals} |")
        lines.append("")

    ps = summary["03_pause"]
    lines.append("## 3. `03_停顿控制` (`none` / `comma` / `period` 标点结构 × 3 seeds)")
    lines.append("")
    lines.append("- **控制方法对比**：`voxcpm2` 与 `indextts25` 仅依赖标点符号；`voxcpm2_ctrl` 在标点基础上额外叠加停顿括号指令（`(连贯紧凑地说，句中不停顿)` / `(在逗号处自然短暂停顿)` / `(在每个句号处明显停顿换气)`）。")
    lines.append("")
    lines.append("| 模型 | 水平数 | Δ_comma 中位数 (s) | Δ(period-comma) (s) | Δ(period-none) 中位数 (s) | comma>none 正向 (Sign p) | period>none 正向 (Sign p) | 内部静音总长 none / comma / period (s) | CER | Sim(to Ref) |")
    lines.append("|---|---:|---:|---:|---:|---|---|---|---:|---:|")
    for cid in CONFIGS:
        x = ps[cid]
        gm = x["mean_internal_gap_dur_s"]
        lines.append(
            f"| `{cid}` | {x['n_levels']} | {x['delta_comma_median_s']} | {x['delta_period_vs_comma_median_s']} | "
            f"{x['delta_period_vs_none_median_s']} | {x['positive_comma']} ({x['sign_p_comma']}) | "
            f"{x['positive_period_vs_none']} ({x['sign_p_period_vs_none']}) | "
            f"{gm.get('none')} / {gm.get('comma')} / {gm.get('period')} | {x['mean_cer']} | {x.get('mean_spk_sim_to_ref')} |"
        )
    lines.append("")
    lines.append("### 3.1 HSK 1–6 分等级停顿明细 (3 seeds 中位数)")
    lines.append("")
    lines.append("| 等级 | `voxcpm2` Δ_comma / Δ(p-n) | `voxcpm2_ctrl` Δ_comma / Δ(p-n) | `indextts25` Δ_comma / Δ(p-n) | 静音总长 `vox_ctrl` (n/c/p) | 静音总长 `index` (n/c/p) |")
    lines.append("|---:|---|---|---|---|---|")
    ps_lvl = {cid: {r["level"]: r for r in ps[cid]["per_level"]} for cid in CONFIGS}
    for lvl in range(1, 7):
        v1 = ps_lvl["voxcpm2"].get(lvl, {})
        v2 = ps_lvl["voxcpm2_ctrl"].get(lvl, {})
        i1 = ps_lvl["indextts25"].get(lvl, {})
        g2 = v2.get("median_internal_gap_dur_s", {})
        gi = i1.get("median_internal_gap_dur_s", {})
        lines.append(
            f"| HSK{lvl} | {v1.get('delta_comma_s')} / {v1.get('delta_period_vs_none_s')}s | "
            f"{v2.get('delta_comma_s')} / {v2.get('delta_period_vs_none_s')}s | "
            f"{i1.get('delta_comma_s')} / {i1.get('delta_period_vs_none_s')}s | "
            f"{g2.get('none')} / {g2.get('comma')} / {g2.get('period')}s | "
            f"{gi.get('none')} / {gi.get('comma')} / {gi.get('period')}s |"
        )
    lines.append("")

    it = summary["04_intonation"]
    lines.append("## 4. `04_语气控制` (陈述 `.` / 疑问 `?` / 感叹 `!` F0 声学统计)")
    lines.append("")
    lines.append("- **控制方法对比**：`voxcpm2` 与 `indextts25` 仅靠陈述/疑问/感叹文本与标点触发；`voxcpm2_ctrl` 显式叠加 `(用平淡陈述的语气)` / `(用疑惑询问、句尾上扬的语气)` / `(用惊讶感叹、强烈的语气)`。")
    lines.append("")
    lines.append("| 模型 | 样本数 | 最强区分特征 | `η²` | `ω²` | Perm `p` | Effect | `f0_mean` (陈述/疑问/感叹 Hz) | `final_delta` (陈述/疑问/感叹 st) | CER | Sim(to Ref) |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---|---|---:|---:|")
    for cid in CONFIGS:
        x = it[cid]
        fm = x.get("features", {}).get("f0_mean", {}).get("means", {})
        fm_str = f"{fm.get('declarative')} / {fm.get('interrogative')} / {fm.get('exclamatory')}"
        fd = x.get("features", {}).get("final_delta", {}).get("means", {})
        fd_str = f"{fd.get('declarative')} / {fd.get('interrogative')} / {fd.get('exclamatory')}"
        mk = x.get("max_eta2_feature", "")
        pp = x.get("features", {}).get(mk, {}).get("perm_p")
        lines.append(
            f"| `{cid}` | {x['n']} | `{mk}` | {x.get('max_eta2')} | "
            f"{x.get('max_omega2')} | {pp} | "
            f"{x.get('effect')} | {fm_str} | {fd_str} | {x['mean_cer']} | {x.get('mean_spk_sim_to_ref')} |"
        )
    lines.append("")

    em = summary["05_emotion"]
    lines.append("## 5. `05_情绪控制` (平静 / 开心 / 悲伤 / 愤怒 LOO-CV)")
    lines.append("")
    lines.append("- **控制方法对比**：`voxcpm2` 无指令基线；`voxcpm2_ctrl` 使用括号情绪描述（如 `(用悲伤、低落、缓慢的语气)`）；`indextts25` 使用解耦的 8 维情绪向量 `emo_vector`（`calm[7]/happy[0]/sad[2]/angry[1]`，强度 1.0）。")
    lines.append("")
    lines.append("| 模型 | 控制模式 | 样本数 | 4类LOO准确率 (chance=0.25) | `obj` 区分度 | CER | Sim(to Ref) | Sim(to GT) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for cid in CONFIGS:
        x = em[cid]
        lines.append(
            f"| `{cid}` | `{x.get('control_mode')}` | {x.get('n')} | {x.get('loo_acc')} | "
            f"{x.get('obj')} | {x.get('mean_cer')} | {x.get('mean_spk_sim_to_ref')} | {x.get('mean_spk_sim_to_gt')} |"
        )
    lines.append("")
    lines.append("### 5.1 分情绪声学特征与音色对比 (`voxcpm2_ctrl` vs `indextts25`)")
    lines.append("")
    lines.append("| 情绪 | 模型 | `f0_mean` (Hz) | `st_std` (st) | `st_range` (st) | 响度 `db` | 时长 `dur` (s) | Sim(to Ref) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    emo_sim = {}
    if rows_by_cfg:
        for cid in ("voxcpm2_ctrl", "indextts25"):
            for e in ("calm", "happy", "sad", "angry"):
                vals = [r.get("spk_sim_to_ref") for r in rows_by_cfg[cid] if r.get("sub") == "emotion" and r.get("emotion") == e and r.get("spk_sim_to_ref") is not None]
                emo_sim[(cid, e)] = safe_mean(vals)
    for e, elab in (("calm", "平静 (calm)"), ("happy", "开心 (happy)"), ("sad", "悲伤 (sad)"), ("angry", "愤怒 (angry)")):
        for cid in ("voxcpm2_ctrl", "indextts25"):
            pm = em[cid].get("per_emotion_means", {}).get(e, {})
            lines.append(
                f"| {elab} | `{cid}` | {pm.get('f0_mean')} | {pm.get('st_std')} | {pm.get('st_range')} | "
                f"{pm.get('db')} | {pm.get('dur')} | {emo_sim.get((cid, e), '—')} |"
            )
    lines.append("")

    pr = summary["06_pronunciation"]
    lines.append("## 6. `06_多音字读音纠错` (HSK对话多音字 + PR1–PR6 + 负对照)")
    lines.append("")
    lines.append("- **控制方法对比**：`voxcpm2` 无读音提示；`voxcpm2_ctrl` 在开头附加括号字音提示（如 `(请注意字音：第1处“行”读hang2...)`）；`indextts25` 使用官方原生 `<字|PINYIN+声调>` 音素替换标签（词表校验自 `pinyin.vocab`）。")
    lines.append("")
    lines.append("| 模型 | 控制模式 | 正例数 | 正例平均CER | 零CER率 | HSK对话CER | PR1–6 CER | 负对照数 | 负对照CER | 负对照ASR文本变化率 | Sim(to Ref) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for cid in CONFIGS:
        x = pr[cid]
        lines.append(
            f"| `{cid}` | `{x['control_mode']}` | {x['n_pos']} | {x['pos_cer_mean']} | "
            f"{x['pos_zero_cer_rate']} | {x['hsk_pos_cer_mean']} | {x['pr_pos_cer_mean']} | "
            f"{x['n_neg']} | {x.get('neg_cer_mean') if x.get('neg_cer_mean') is not None else '—'} | "
            f"{x['neg_asr_diff_vs_pos']} | {x['mean_spk_sim_to_ref']} |"
        )
    lines.append("")
    if rows_by_cfg:
        pr_rows = {cid: [r for r in rows_by_cfg[cid] if r.get("sub") == "pronunciation" and r.get("status") == "ok"] for cid in CONFIGS}
        lines.append("### 6.1 分组正例 CER 明细 (`n = 42`)")
        lines.append("")
        lines.append("| 用例组 | 测试文本 | 正例拼音标注 | `voxcpm2` CER (零CER率) | `voxcpm2_ctrl` CER (零CER率) | `indextts25` CER (零CER率) |")
        lines.append("|---|---|---|---:|---:|---:|")
        pos_ids = []
        seen_pid = set()
        for r in pr_rows["indextts25"]:
            if not r.get("negative_control") and r["poly_id"] not in seen_pid:
                seen_pid.add(r["poly_id"])
                pos_ids.append((r["poly_id"], r["text"], r["pinyin_hints"]))
        for pid, ptxt, phints in pos_ids:
            cols = []
            for cid in CONFIGS:
                sub_r = [r for r in pr_rows[cid] if not r.get("negative_control") and r["poly_id"] == pid]
                c_m = safe_mean([r.get("cer") for r in sub_r])
                zr = f"{sum(1 for r in sub_r if r.get('cer') == 0.0)}/{len(sub_r)}"
                cols.append(f"{c_m} ({zr})")
            lines.append(f"| `{pid}` | {ptxt} | `{phints}` | {cols[0]} | {cols[1]} | {cols[2]} |")
        lines.append("")
        lines.append("### 6.2 负对照反例控制生效明细 (`n = 18`，故意标反拼音测试是否真的改读)")
        lines.append("")
        lines.append("| 等级 / 用例 | 正例标注 → 负对照反标 | `voxcpm2_ctrl` ASR变化 | `indextts25` ASR变化 | `indextts25` ASR 回转写变化（正例 → 负对照） |")
        lines.append("|---|---|---|---|---|")
        neg_idx = [r for r in pr_rows["indextts25"] if r.get("negative_control")]
        pos_idx_map = {(int(r["level"]), r["poly_id"]): r for r in pr_rows["indextts25"] if not r.get("negative_control")}
        pos_vc_map = {(int(r["level"]), r["poly_id"]): r for r in pr_rows["voxcpm2_ctrl"] if not r.get("negative_control")}
        neg_vc_map = {r["item_id"]: r for r in pr_rows["voxcpm2_ctrl"] if r.get("negative_control")}
        for nr in sorted(neg_idx, key=lambda x: (x["poly_id"], int(x["level"]))):
            base_pid = nr["poly_id"].replace("_NEG", "")
            pr_i = pos_idx_map[(int(nr["level"]), base_pid)]
            pr_v = pos_vc_map[(int(nr["level"]), base_pid)]
            nr_v = neg_vc_map.get(nr["item_id"], {})
            chg_i = "✅ 改变" if pr_i.get("asr_hyp_norm") != nr.get("asr_hyp_norm") else "— 未变(同字纠偏)"
            chg_v = "✅ 改变" if pr_v.get("asr_hyp_norm") != nr_v.get("asr_hyp_norm") else "❌ 0变化"
            hyp_show = f"`{pr_i.get('asr_hyp')}` → `{nr.get('asr_hyp')}`" if pr_i.get("asr_hyp_norm") != nr.get("asr_hyp_norm") else f"`{nr.get('asr_hyp')}`"
            lines.append(f"| HSK{nr['level']} `{base_pid}` | `{pr_i.get('pinyin_hints')}` → `{nr.get('pinyin_hints')}` | {chg_v} | {chg_i} | {hyp_show} |")
    lines.append("")
    lines.append("> 注：SenseVoice ASR 以中文汉字转写为主，部分同字异音负对照（如 `银行` 标 `XING2` 仍被 ASR 语言模型强行纠成 `银行`、`了|LIAO3` vs `了|LE5`）请结合 `audio/P2plus/index.html` 在浏览器中直接试听对比；`PR3` 中 `玩得开心` 的轻声 `得` 被 ASR 一律转写为同音 `的`，占正例 CER 误差主要来源。")
    return "\n".join(lines) + "\n"


def rel_from_p2plus(abs_path):
    return os.path.relpath(abs_path, P2PLUS_DIR).replace("\\", "/")


def build_html(summary, manifest, rows_by_cfg):
    gt_by_lvl = {int(m["level"]): m for m in manifest}
    sub_titles = [
        ("01_默认语速基线", "01 · 默认语速基线"),
        ("02_语速控制", "02 · 语速控制"),
        ("03_停顿控制", "03 · 停顿控制"),
        ("04_语气控制", "04 · 语气控制"),
        ("05_情绪控制", "05 · 情绪控制"),
        ("06_多音字读音纠错", "06 · 多音字读音纠错"),
    ]
    by_id = {cid: {r["item_id"]: r for r in rows_by_cfg[cid]} for cid in CONFIGS}

    def audio_cell(r):
        if not r or r.get("status") != "ok":
            return "<td class='na'>—</td>"
        src = html.escape(rel_from_p2plus(r["out_wav"]))
        meta_parts = [f"{r.get('dur_s')}s", f"{r.get('actual_cpm')} cpm"]
        if r.get("cer") is not None:
            meta_parts.append(f"CER={r['cer']}")
        if r.get("spk_sim_to_gt") is not None:
            meta_parts.append(f"simGT={r['spk_sim_to_gt']}")
        if r.get("opts", {}).get("duration_factor") is not None:
            meta_parts.append(f"df={r['opts']['duration_factor']}")
        if r.get("opts", {}).get("vox_control"):
            meta_parts.append(f"({html.escape(r['opts']['vox_control'])})")
        if r.get("pinyin_hints"):
            meta_parts.append(f"{html.escape(r['pinyin_hints'])}")
        meta = " | ".join(meta_parts)
        return (
            f"<td><audio controls preload='none' src='{src}'></audio>"
            f"<div class='meta'>{meta}</div></td>"
        )

    parts = [
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>",
        "<title>P2plus 真实 HSK 1–6 对照试听面板</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;background:#f8fafc;color:#0f172a;line-height:1.5;}",
        "h1,h2{color:#0f172a;} nav{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px;}",
        "nav a{padding:6px 12px;background:#e2e8f0;border-radius:6px;text-decoration:none;color:#1e293b;font-weight:600;font-size:14px;}",
        "nav a:hover{background:#cbd5e1;} section{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:18px;margin-bottom:24px;}",
        "table{width:100%;border-collapse:collapse;font-size:13px;} th,td{border:1px solid #e2e8f0;padding:8px;vertical-align:top;text-align:left;}",
        "th{background:#f1f5f9;position:sticky;top:0;} audio{width:220px;height:32px;} .meta{font-size:11px;color:#475569;margin-top:4px;}",
        ".na{color:#94a3b8;text-align:center;} .badge{display:inline-block;padding:2px 6px;border-radius:4px;background:#dbeafe;color:#1e40af;font-size:11px;margin-right:4px;}",
        "</style></head><body>",
        "<h1>P2plus · HSK 1–6 真实语音对照与全维可控性试听</h1>",
        "<nav>" + "".join(f"<a href='#{html.escape(f)}'>{html.escape(t)}</a>" for f, t in sub_titles) + "</nav>",
        "<section><h2>HSK 1–6 Ground-Truth (`data_gt`) 与克隆参考 (`refs`)</h2><table><tr><th>等级</th><th>目标对话 (`data_gt`)</th><th>文本</th><th>克隆参考 (`refs`)</th><th>REF-GT Sim</th></tr>",
    ]
    for m in manifest:
        gt_src = html.escape(rel_from_p2plus(m["target_gt_wav"]))
        ref_src = html.escape(rel_from_p2plus(m["ref_wav"]))
        parts.append(
            f"<tr><td>HSK{m['level']}</td>"
            f"<td><audio controls preload='none' src='{gt_src}'></audio><div class='meta'>{html.escape(m['target_sample_id'])} | {m['target_metrics']['actual_cpm']} cpm</div></td>"
            f"<td>{html.escape(m['target_text'])}</td>"
            f"<td><audio controls preload='none' src='{ref_src}'></audio><div class='meta'>{html.escape(m['ref_sample_id'])} | {m['ref_metrics']['duration_s']}s</div></td>"
            f"<td>{m['ref_target_spk_sim']}</td></tr>"
        )
    parts.append("</table></section>")

    master_rows = load_jsonl(os.path.join(C.E, "texts", "p2plus.jsonl"))
    for folder, title in sub_titles:
        parts.append(f"<section id='{html.escape(folder)}'><h2>{html.escape(title)}</h2>")
        parts.append("<table><tr><th>Item</th><th>等级 / 条件</th><th>合成文本</th><th>Ground Truth (`data_gt`)</th><th>VoxCPM2 (`vox`)</th><th>VoxCPM2 可控克隆 (`vox_ctrl`)</th><th>IndexTTS-2.5 (`index`)</th></tr>")
        for mr in master_rows:
            if mr["folder"] != folder:
                continue
            iid = mr["item_id"]
            vr = by_id["voxcpm2"].get(iid)
            vcr = by_id["voxcpm2_ctrl"].get(iid)
            ir = by_id["indextts25"].get(iid)
            ref_r = ir or vcr or vr or mr
            lvl = int(ref_r["level"])
            gt_src = html.escape(rel_from_p2plus(gt_by_lvl[lvl]["target_gt_wav"]))
            gt_cpm = gt_by_lvl[lvl]["target_metrics"]["actual_cpm"]
            cond_bits = [f"<span class='badge'>HSK{lvl}</span>"]
            for k in ["rate_tag", "pause_setting", "replicate", "intonation_label", "emotion_label", "poly_id"]:
                if ref_r.get(k) is not None:
                    cond_bits.append(f"<span class='badge'>{html.escape(str(ref_r[k]))}</span>")
            if ref_r.get("negative_control"):
                cond_bits.append("<span class='badge'>NEG</span>")
            parts.append(
                f"<tr><td><code>{html.escape(iid)}</code></td>"
                f"<td>{''.join(cond_bits)}</td>"
                f"<td>{html.escape(ref_r['text'])}</td>"
                f"<td><audio controls preload='none' src='{gt_src}'></audio><div class='meta'>{gt_cpm} cpm</div></td>"
                f"{audio_cell(vr)}{audio_cell(vcr)}{audio_cell(ir)}</tr>"
            )
        parts.append("</table></section>")

    parts.append("</body></html>")
    return "\n".join(parts)


def run_acoustics_report():
    manifest = load_manifest()
    gt_by_lvl = {int(m["level"]): m for m in manifest}
    rows_by_cfg = {cid: load_jsonl(os.path.join(RESULTS_DIR, f"{cid}.eval.jsonl")) for cid in CONFIGS}
    for cid in CONFIGS:
        if not rows_by_cfg[cid]:
            rows_by_cfg[cid] = load_jsonl(os.path.join(RESULTS_DIR, f"{cid}.jsonl"))

    C.log("[eval:acoustics] Computing all 6 sub-experiment evaluations...")
    summary = {
        "manifest": manifest,
        "01_baseline": eval_baseline(rows_by_cfg, gt_by_lvl),
        "02_rate": eval_rate(rows_by_cfg, gt_by_lvl),
        "03_pause": eval_pause(rows_by_cfg),
        "04_intonation": eval_intonation(rows_by_cfg, gt_by_lvl),
        "05_emotion": eval_emotion(rows_by_cfg),
        "06_pronunciation": eval_pronunciation(rows_by_cfg),
    }

    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    for sp in (summary_path, os.path.join(P2PLUS_DIR, "summary.json")):
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    report_md = build_markdown(summary, manifest, rows_by_cfg)
    report_path = os.path.join(P2PLUS_DIR, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    index_html = build_html(summary, manifest, rows_by_cfg)
    html_path = os.path.join(P2PLUS_DIR, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(index_html)

    C.log(f"[eval:acoustics] Wrote {summary_path}, {report_path}, {html_path}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["asr_spk", "acoustics_report"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.stage == "asr_spk":
        return run_asr_spk(args.device)
    return run_acoustics_report()


if __name__ == "__main__":
    sys.exit(main())
