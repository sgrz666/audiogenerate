#!/usr/bin/env python3
"""P2/P2b objective evaluation.

  --pause    pause v2 pairs: speech-span delta (primary) + gap description,
             v1 same-text noise floor, sign test.
  --prosody  intonation eta2/omega2/permutation and IndexTTS emotion LOO.

Run with an env that has librosa + soundfile (envs/indextts is verified).
Does not write outside results/P2b/eval/.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_common as C  # noqa: E402

VALID = ["voxcpm2", "indextts25", "qwen3tts_base",
         "fireredtts3_base", "fireredtts3_instruct"]
SETTINGS = ["none", "comma", "period"]
OUTDIR = os.path.join(C.E, "results", "P2b", "eval")


def log(m):
    print(m, flush=True)


def load_jsonl(path):
    if not os.path.isfile(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_meta():
    meta = {}
    for r in load_jsonl(os.path.join(C.E, "texts", "p2b.jsonl")):
        meta[r["item_id"]] = r
    return meta


def load_wav(p):
    import librosa
    import soundfile as sf
    w, sr = sf.read(p, dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != 16000:
        w = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000
    return w.astype(np.float32), sr


def pause_gaps(wav, sr, top_db=30, min_gap=0.10):
    """Descriptive only: librosa.effects.split is not sensitive to shallow TTS
    pauses; the score uses speech_span instead (see P2_RUBRIC.md)."""
    import librosa
    iv = librosa.effects.split(wav, top_db=top_db, frame_length=400,
                               hop_length=160)
    out = []
    prev = None
    for s, e in iv:
        if prev is not None:
            g = (s - prev) / sr
            if g >= min_gap:
                out.append({"start": round(prev / sr, 3),
                            "end": round(s / sr, 3),
                            "dur": round(float(g), 3)})
        prev = e
    return out


def binom_two_sided(k, n, p=0.5):
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    obs = probs[k]
    return float(min(1.0, sum(x for x in probs if x <= obs + 1e-12)))


def span_per_char(row):
    nch = row.get("norm_chars")
    span = row.get("speech_span_s")
    if not nch or span is None:
        return None
    return float(span) / float(nch)


def pause_analysis():
    meta = load_meta()
    rows = []
    for cfg in VALID:
        for r in load_jsonl(os.path.join(C.E, "results", "P2b", cfg + ".jsonl")):
            if r.get("sub") != "pause" or r.get("status") != "ok":
                continue
            m = meta.get(r["item_id"], {})
            r = dict(r)
            r["base_text_id"] = m.get("base_text_id", r.get("base_text_id"))
            r["pause_setting"] = m.get("pause_setting", r.get("pause_setting"))
            r["replicate"] = m.get("replicate", r.get("replicate"))
            rows.append(r)
    out = {"n_rows": len(rows), "configs": {}, "params": {
        "primary": "delta_s=(span_per_char_comma-span_per_char_none)*norm_chars",
        "anchor_s": 0.2, "seeds": [42, 43, 44], "gap_detector": "librosa.effects.split(top_db=30)",
        "note": "gap stats are descriptive; shallow pauses can be missed"}}
    for cfg in VALID:
        cr = [r for r in rows if r["config_id"] == cfg]
        if not cr:
            out["configs"][cfg] = {"status": "not_tested"}
            continue
        groups = defaultdict(dict)
        for r in cr:
            key = (r["voice_id"], r["base_text_id"])
            groups[key].setdefault(r["pause_setting"], []).append(r)
        triples = []
        for key, d in sorted(groups.items()):
            if not all(s in d and d[s] for s in SETTINGS):
                continue
            med = {}
            for s in SETTINGS:
                vals = [span_per_char(x) for x in d[s]]
                vals = [v for v in vals if v is not None]
                if not vals:
                    break
                med[s] = float(np.median(vals))
            if len(med) != 3:
                continue
            nch = d["none"][0].get("norm_chars") or 0
            delta_comma = (med["comma"] - med["none"]) * nch
            delta_period = (med["period"] - med["comma"]) * nch
            triples.append({"voice_id": key[0], "base_text_id": key[1],
                            "nch": nch, "span_per_char": {s: round(med[s], 6) for s in SETTINGS},
                            "delta_comma_s": round(delta_comma, 4),
                            "delta_period_s": round(delta_period, 4)})
        if not triples:
            out["configs"][cfg] = {"status": "no_complete_triples", "n_rows": len(cr)}
            continue
        deltas = np.array([t["delta_comma_s"] for t in triples])
        pos = int((deltas > 0).sum())
        n = len(deltas)
        # Median, not the mean of clipped positives: VoxCPM2 had median -0.04 s
        # but the clipped mean was 0.44 because half the triples were positive
        # by noise.  The sign test is reported alongside.
        effect = float(max(0.0, min(1.0, float(np.median(deltas)) / 0.2)))
        # v1 noise floor: same text across three settings, one render each.
        v1 = []
        for r in load_jsonl(os.path.join(C.E, "results", "P2", cfg + ".jsonl")):
            if r.get("sub") != "pause" or r.get("status") != "ok":
                continue
            v1.append(r)
        v1g = defaultdict(dict)
        for r in v1:
            v1g[(r["voice_id"], r["text"])][r["pause_setting"]] = r
        v1_deltas = []
        for key, d in v1g.items():
            if all(s in d for s in SETTINGS):
                a, b = span_per_char(d["none"]), span_per_char(d["comma"])
                nch = d["none"].get("norm_chars") or 0
                if a is not None and b is not None and nch:
                    v1_deltas.append(abs((b - a) * nch))
        out["configs"][cfg] = {
            "status": "ok", "n_triples": n,
            "delta_comma_median_s": round(float(np.median(deltas)), 4),
            "delta_comma_mean_s": round(float(np.mean(deltas)), 4),
            "positive_comma": pos, "sign_p": round(binom_two_sided(pos, n), 4),
            "pause_effect": round(effect, 4),
            "v1_same_text_abs_delta_median_s": round(float(np.median(v1_deltas)), 4) if v1_deltas else None,
            "gate_v2_gt_v1_noise": bool(v1_deltas and float(np.median(deltas)) > 0
                                         and float(np.median(deltas)) > float(np.median(v1_deltas))),
            "triples": triples,
        }
        # gap description on one representative voice/text (all settings, all reps)
        gap_stats = {}
        for s in SETTINGS:
            gs = []
            for r in cr:
                if r["pause_setting"] == s and r["base_text_id"] == "P1" and r["voice_id"] == "MALE":
                    w, sr = load_wav(r["out_wav"])
                    g = pause_gaps(w, sr)
                    gs.append({"n": len(g), "total": round(sum(x["dur"] for x in g), 3),
                               "max": max([x["dur"] for x in g], default=0)})
            gap_stats[s] = gs
        out["configs"][cfg]["gap_stats_P1_MALE"] = gap_stats
        log(f"[pause] {cfg} n={n} median_delta={np.median(deltas):.3f}s "
            f"effect={effect:.3f} pos={pos}/{n} p={out['configs'][cfg]['sign_p']} "
            f"v1noise={out['configs'][cfg]['v1_same_text_abs_delta_median_s']}")
    os.makedirs(OUTDIR, exist_ok=True)
    p = os.path.join(OUTDIR, "pause_pairs.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"[pause] -> {p}")


# ---------------- prosody ----------------

def f0_features(path):
    import librosa
    from scipy.ndimage import median_filter
    w, sr = load_wav(path)
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
    return {"f0_mean": float(np.mean(v)), "st_std": float(np.std(st)),
            "st_range": float(np.percentile(st, 95) - np.percentile(st, 5)),
            "final_delta": final_delta, "dur": len(w) / sr,
            "db": float(np.mean(db)), "db_std": float(np.std(db))}


def eta2(groups, vals):
    vals = np.asarray(vals, float)
    grand = vals.mean()
    ssb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    sst = ((vals - grand) ** 2).sum()
    return float(ssb / sst) if sst > 0 else 0.0


def omega2(groups, vals):
    vals = np.asarray(vals, float)
    k = len(groups)
    n = len(vals)
    if n <= k:
        return 0.0
    grand = vals.mean()
    ssb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
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


def prosody_analysis():
    out = {"intonation": {}, "emotion": {}}
    # intonation
    for cfg in VALID:
        rows = [r for r in load_jsonl(os.path.join(C.E, "results", "P2", cfg + ".jsonl"))
                if r.get("sub") == "intonation" and r.get("status") == "ok"]
        if not rows:
            out["intonation"][cfg] = {"status": "not_tested"}
            continue
        data = defaultdict(list)
        feats = []
        for r in rows:
            f = f0_features(r["out_wav"])
            if f is None:
                continue
            f["type"] = r["intonation_type"]
            f["voice"] = r["voice_id"]
            f["text_id"] = r["text_id"].split("-")[0] if "-" in r["text_id"] else r["text_id"]
            feats.append(f)
            data[r["intonation_type"]].append(f)
        if len(data) < 3:
            out["intonation"][cfg] = {"status": "insufficient", "n": len(feats)}
            continue
        feats_clean = [f for f in feats if np.isfinite(f["final_delta"])]
        res = {"status": "ok", "n": len(feats), "n_final_delta": len(feats_clean),
               "features": {}}
        for key in ["f0_mean", "st_std", "st_range", "final_delta"]:
            groups = [[f[key] for f in data[t] if np.isfinite(f[key])] for t in ["declarative", "interrogative", "exclamatory"]]
            vals = [x for g in groups for x in g]
            if len(vals) < 9:
                continue
            e2 = eta2(groups, vals)
            w2 = omega2(groups, vals)
            p = perm_p(groups, vals)
            res["features"][key] = {"means": [round(float(np.mean(g)), 3) for g in groups],
                                    "eta2": round(e2, 4), "omega2": round(w2, 4),
                                    "perm_p": round(p, 4)}
        max_key = max(res["features"], key=lambda k: res["features"][k]["eta2"]) if res["features"] else None
        if max_key:
            max_eta = res["features"][max_key]["eta2"]
            max_w2 = res["features"][max_key]["omega2"]
            res["max_eta2_feature"] = max_key
            res["max_eta2"] = max_eta
            res["max_omega2"] = max_w2
            # ω² is the small-sample-corrected effect; using η² alone gives a
            # near-zero ω² (Qwen 0.0038) a 0.58 score.  Anchor 0.12 ~ Cohen f=0.4.
            res["effect"] = round(float(max(0.0, min(1.0, max_w2 / 0.12))), 4)
        out["intonation"][cfg] = res
        log(f"[intonation] {cfg} n={len(feats)} max_eta2={res.get('max_eta2')} ({max_key}) "
            f"omega2={res.get('max_omega2')} effect={res.get('effect')}")

    # emotion (IndexTTS only; other configs have no signal)
    cfg = "indextts25"
    rows = [r for r in load_jsonl(os.path.join(C.E, "results", "P2", cfg + ".jsonl"))
            if r.get("sub") == "emotion" and r.get("status") == "ok"]
    feats = []
    for r in rows:
        f = f0_features(r["out_wav"])
        if f is None:
            continue
        f["emo"] = r["emotion"]
        f["voice"] = r["voice_id"]
        f["text_id"] = r["text_id"]
        feats.append(f)
    if len(feats) >= 12:
        # Emotion is pitch level/range + energy + rate; final_delta is an
        # intonation feature and its inclusion blurred sad/happy (0.583 -> 0.472).
        keys = ["f0_mean", "st_std", "st_range", "db", "db_std", "dur"]
        X = np.array([[f[k] for k in keys] for f in feats], float)
        # replace non-finite with column median
        for j in range(X.shape[1]):
            col = X[:, j]
            med = np.nanmedian(col)
            col[~np.isfinite(col)] = med
            X[:, j] = col
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        y = np.array([f["emo"] for f in feats])
        emos = ["calm", "happy", "sad", "angry"]
        correct = 0
        conf = defaultdict(int)
        for i in range(len(y)):
            tr = np.arange(len(y)) != i
            cents = {e: X[tr & (y == e)].mean(0) for e in emos if (tr & (y == e)).any()}
            pred = min(cents, key=lambda e: np.linalg.norm(X[i] - cents[e]))
            correct += int(pred == y[i])
            conf[f"{y[i]}->{pred}"] += 1
        acc = correct / len(y)
        out["emotion"][cfg] = {"status": "ok", "n": len(y), "acc": round(acc, 4),
                               "chance": 0.25,
                               "obj": round(float(max(0.0, min(1.0, (acc - 0.25) / 0.75))), 4),
                               "confusion": dict(conf),
                               "per_emotion_means": {
                                   e: {k: round(float(np.mean([f[k] for f in feats if f["emo"] == e])), 3)
                                       for k in ["f0_mean", "st_std", "st_range", "dur"]}
                                   for e in emos}}
        log(f"[emotion] {cfg} acc={acc:.3f} obj={out['emotion'][cfg]['obj']}")
    else:
        out["emotion"][cfg] = {"status": "insufficient", "n": len(feats)}
    os.makedirs(OUTDIR, exist_ok=True)
    p = os.path.join(OUTDIR, "prosody.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"[prosody] -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause", action="store_true")
    ap.add_argument("--prosody", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all or args.pause:
        pause_analysis()
    if args.all or args.prosody:
        prosody_analysis()
    if not (args.all or args.pause or args.prosody):
        ap.error("choose --pause / --prosody / --all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
