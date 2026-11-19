#!/usr/bin/env python3
"""Run all 6 P2plus control sub-experiments for voxcpm2 or indextts25."""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_adapters as A
import mt_common as C
import mt_control as K

P2PLUS_DIR = os.path.join(C.E, "audio", "P2plus")
MANIFEST_PATH = os.path.join(P2PLUS_DIR, "manifest.json")
TEXTS_PATH = os.path.join(C.E, "texts", "p2plus.jsonl")
RESULTS_DIR = os.path.join(C.E, "results", "P2plus")

SUB_DIR_MAP = {
    "voxcpm2": "vox",
    "voxcpm2_ctrl": "vox_ctrl",
    "indextts25": "index",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(x["level"]): x for x in data}


def load_items(cid):
    items = []
    with open(TEXTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if cid == "voxcpm2" and r.get("index_only"):
                continue
            items.append(r)
    baselines = [x for x in items if x["sub"] == "baseline"]
    others = [x for x in items if x["sub"] != "baseline"]
    return baselines + others


def _vox_rate_prompt(target_cpm):
    c = float(target_cpm)
    if c <= 165.0:
        return "语速非常缓慢，一字一顿慢慢说"
    if c <= 205.0:
        return "语速偏慢"
    if c <= 250.0:
        return "语速适中，自然平稳"
    if c <= 295.0:
        return "语速偏快"
    return "语速非常快，紧凑流利"


def _vox_pinyin_prompt(hints):
    parts = []
    for idx, pair in enumerate(hints.split(";"), 1):
        pair = pair.strip()
        if not pair or "|" not in pair:
            continue
        w, r = pair.split("|", 1)
        parts.append(f"第{idx}处“{w}”读{r.strip().lower()}")
    return "请注意字音：" + "，".join(parts) if parts else ""


def build_control(cid, item, baseline_cpm, manifest_entry):
    sub = item["sub"]
    seed = int(item.get("seed", 42))
    text = item["text"]
    opts = {"seed": seed}
    if sub == "baseline":
        return text, opts, "baseline"
    if cid == "voxcpm2_ctrl":
        ctrl = None
        mode = "native_prompt"
        if sub == "rate":
            ctrl = _vox_rate_prompt(item["target_cpm"])
        elif sub == "pause":
            ctrl = {
                "none": "连贯紧凑地说，句中不停顿",
                "comma": "在逗号处自然短暂停顿",
                "period": "在每个句号处明显停顿换气",
            }[item["pause_setting"]]
        elif sub == "intonation":
            ctrl = {
                "declarative": "用平淡陈述的语气",
                "interrogative": "用疑惑询问、句尾上扬的语气",
                "exclamatory": "用惊讶感叹、强烈的语气",
            }[item["intonation_type"]]
        elif sub == "emotion":
            ctrl = {
                "calm": "用平静、平稳的语气",
                "happy": "用高兴、欢快的语气",
                "sad": "用悲伤、低落、缓慢的语气",
                "angry": "用生气、愤怒、强烈的语气",
            }[item["emotion"]]
        elif sub == "pronunciation":
            ctrl = _vox_pinyin_prompt(item.get("pinyin_hints", ""))
            mode = "prompt_hint"
        if ctrl:
            opts["vox_control"] = ctrl
            return f"({ctrl}){text}", opts, mode
        return text, opts, "text_fallback"
    if sub == "rate":
        if cid == "indextts25":
            base_cpm = baseline_cpm.get(int(item["level"]), manifest_entry["target_metrics"]["actual_cpm"])
            df = round(float(base_cpm) / float(item["target_cpm"]), 4)
            opts["duration_factor"] = df
            return text, opts, "native"
        return text, opts, "text_fallback"
    if sub in ("pause", "intonation"):
        return text, opts, "text_punctuation"
    if sub == "emotion":
        if cid == "indextts25":
            opts["emo_vector"] = item["emo_vector"]
            return text, opts, "native"
        return text, opts, "text_fallback"
    if sub == "pronunciation":
        if cid == "indextts25":
            synth_text, applied = K._pinyin_text(text, item["pinyin_hints"])
            opts["applied_hints"] = applied
            return synth_text, opts, "native"
        return text, opts, "text_fallback"
    return text, opts, "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["voxcpm2", "voxcpm2_ctrl", "indextts25"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", action="store_true", help="Keep existing status=ok rows in out_jsonl")
    args = ap.parse_args()

    cid = args.config
    base_cid = "voxcpm2" if cid == "voxcpm2_ctrl" else cid
    cfg = C.cfg_by_id(base_cid)
    manifest = load_manifest()
    items = load_items(cid)

    model_root = os.path.join(P2PLUS_DIR, SUB_DIR_MAP[cid])
    os.makedirs(model_root, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_jsonl = os.path.join(RESULTS_DIR, f"{cid}.jsonl")

    done_by_id = {}
    baseline_cpm = {}
    if args.resume and os.path.isfile(out_jsonl):
        with open(out_jsonl, "r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("status") == "ok" and os.path.isfile(r.get("out_wav", "")):
                    done_by_id[r["item_id"]] = r
                    if r.get("sub") == "baseline" and r.get("actual_cpm"):
                        baseline_cpm[int(r["level"])] = float(r["actual_cpm"])
        C.log(f"[P2plus] Resuming {cid}: {len(done_by_id)} already ok, {len(items) - len(done_by_id)} remaining")

    pending = [it for it in items if it["item_id"] not in done_by_id]
    if not pending:
        C.log(f"[P2plus] Nothing to run for {cid}; all {len(items)} items ok")
        return 0

    C.log(f"[P2plus] Loading {cid} ({cfg['family']}) on {args.device}; tasks={len(pending)} (total={len(items)})")
    h = A.LOAD[base_cid](cfg, args.device)
    ok_cnt = 0
    err_cnt = 0

    try:
        mode = "a" if (args.resume and done_by_id) else "w"
        with open(out_jsonl, mode, encoding="utf-8") as fout:
            for idx, item in enumerate(pending, 1):
                lvl = int(item["level"])
                m = manifest[lvl]
                ref = {
                    "ref_wav": m["ref_wav"],
                    "ref_text": m["ref_text"],
                    "ref_sample_id": m["ref_sample_id"],
                }
                sub_dir = os.path.join(model_root, item["folder"])
                os.makedirs(sub_dir, exist_ok=True)
                out_wav = os.path.join(sub_dir, item["filename"])

                synth_text, opts, control_mode = build_control(cid, item, baseline_cpm, m)
                row = dict(item)
                row.update({
                    "config_id": cid,
                    "model_name": cfg["family"],
                    "control_mode": control_mode,
                    "synth_text": synth_text,
                    "opts": opts,
                    "ref_sample_id": m["ref_sample_id"],
                    "ref_wav": m["ref_wav"],
                    "gt_sample_id": m["target_sample_id"],
                    "gt_wav": m["target_gt_wav"],
                    "out_wav": out_wav,
                })

                try:
                    set_seed(int(item.get("seed", 42)))
                    C.vram_reset()
                    t0 = time.perf_counter()
                    wav, sr, meta = A.SYNTH[base_cid](h, synth_text, ref, out_wav, opts)
                    gen_s = time.perf_counter() - t0
                    if not meta.get("written_by_model"):
                        C.write_wav(out_wav, wav, sr)
                    dur_s = float(len(wav)) / float(sr) if sr else 0.0
                    sp0, sp1 = C.speech_span(wav, sr)
                    span_s = max(0.0, float(sp1) - float(sp0))
                    nch = C.norm_chars(item["text"])
                    denom = span_s if span_s > 0.2 else dur_s
                    actual_cpm = round(float(nch) / (denom / 60.0), 2) if denom > 0 else 0.0
                    rtf = round(gen_s / dur_s, 4) if dur_s > 0 else 0.0
                    if item["sub"] == "baseline":
                        baseline_cpm[lvl] = actual_cpm
                    row.update({
                        "status": "ok",
                        "sr": int(sr),
                        "dur_s": round(dur_s, 3),
                        "speech_span_s": round(float(span_s), 3),
                        "norm_chars": nch,
                        "actual_cpm": actual_cpm,
                        "gen_s": round(gen_s, 3),
                        "rtf": rtf,
                        "vram_gib": (round(C.vram_peak_gib(), 3) if C.vram_peak_gib() is not None else None),
                        "rss_gib": (round(C.rss_gib(), 3) if C.rss_gib() is not None else None),
                    })
                    ok_cnt += 1
                except Exception as exc:
                    row.update({
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    err_cnt += 1
                    C.log(f"[ERR] {cid} {item['item_id']}: {row['error']}")

                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()

                if idx % 15 == 0 or idx == len(pending) or item["sub"] == "baseline":
                    C.log(
                        f"[{cid}] {idx}/{len(pending)} {item['item_id']} "
                        f"status={row['status']} dur={row.get('dur_s')} cpm={row.get('actual_cpm')} "
                        f"gen={row.get('gen_s')}s"
                    )
    finally:
        C.release(h)

    C.log(f"[P2plus] Finished {cid}: ok={ok_cnt}, err={err_cnt} -> {out_jsonl}")
    return 0 if err_cnt == 0 else 1


if __name__ == "__main__":
    sys.exit(main())