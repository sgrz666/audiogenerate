#!/usr/bin/env python3
"""P2 control-experiment worker (one config per process, serial).

Usage: mt_run_p2.py --config indextts25 [--sub rate,emotion] [--limit N]

Rate items need a baseline 字/分钟 to convert the target cpm into the model's
multiplier (IndexTTS duration_factor is inverse, CosyVoice speed is direct).
The baseline is measured once per (config, voice) on the first rate text and
cached to results/P2/baseline_<config>.json; the conversion is recorded per item.
"""
import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sub", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--texts-file", default="texts/p2.jsonl")
    ap.add_argument("--tag", default="P2")
    args = ap.parse_args()

    cfg = C.cfg_by_id(args.config)
    cid = cfg["config_id"]
    tf = args.texts_file
    if not os.path.isabs(tf):
        tf = os.path.join(C.E, tf)
    tag = args.tag
    items = [json.loads(l) for l in open(tf, encoding="utf-8")]
    items = [i for i in items if i["config_id"] == cid]
    if args.sub:
        want = set(args.sub.split(","))
        items = [i for i in items if i["sub"] in want]
    if args.limit:
        items = items[: args.limit]
    refs = C.load_refs()

    import torch
    torch.set_num_threads(min(8, os.cpu_count() or 8))
    import mt_adapters as A
    import mt_control as K

    rp = C.result_path(cid, tag)
    done = set()
    if os.path.isfile(rp):
        for l in open(rp, encoding="utf-8"):
            try:
                d = json.loads(l)
            except json.JSONDecodeError:
                continue
            if d.get("status") == "ok":
                done.add(d["item_id"])

    C.log(f"[{tag} {cid}] items={len(items)} already_ok={len(done)}")
    t0 = time.time()
    try:
        C.vram_reset()
        handle = A.LOAD[cid](cfg, args.device)
        C.log(f"[{tag} {cid}] loaded {time.time()-t0:.1f}s vram={C.vram_peak_gib()}GiB")
    except Exception as e:
        with open(rp, "a", encoding="utf-8") as f:
            f.write(json.dumps({"config_id": cid, "item_id": "", "status": "failed",
                                "stage": "load", "error_type": type(e).__name__,
                                "error_msg": str(e)[:1200]}, ensure_ascii=False) + "\n")
        C.log(f"[{tag} {cid}] LOAD FAILED {type(e).__name__}: {str(e)[:200]}")
        return 2

    bpath = os.path.join(C.E, "results", tag, f"baseline_{cid}.json")
    baseline = json.load(open(bpath, encoding="utf-8")) if os.path.isfile(bpath) else {}

    def get_baseline(voice, ref):
        if voice in baseline:
            return baseline[voice]
        # measure on the first rate text at default settings
        probe = next((i for i in items if i["sub"] == "rate" and i["voice_id"] == voice), None)
        if probe is None:
            return None
        tmp = os.path.join(C.E, "audio", tag, cid, "_baseline_" + voice + ".wav")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        w, sr, _ = A.SYNTH[cid](handle, probe["text"], ref, tmp, {"seed": 42})
        if not os.path.isfile(tmp):
            C.write_wav(tmp, w, sr)
        s0, s1 = C.speech_span(w, sr)
        cpm = C.norm_chars(probe["text"]) / max(s1 - s0, 1e-6) * 60
        baseline[voice] = round(cpm, 2)
        os.makedirs(os.path.dirname(bpath), exist_ok=True)
        json.dump(baseline, open(bpath, "w", encoding="utf-8"), ensure_ascii=False)
        C.log(f"[{tag} {cid}] baseline {voice} = {cpm:.1f} cpm")
        return baseline[voice]

    n_ok = n_fail = 0
    fout = open(rp, "a", encoding="utf-8")
    for i, it in enumerate(items, 1):
        if it["item_id"] in done:
            continue
        v = it["voice_id"]
        ref = refs[v]
        seed = int(it.get("seed", 42))
        out = C.out_path(cid, tag, v, it["item_id"])
        row = {"config_id": cid, "family": cfg["family"], "variant": cfg["variant"],
               "phase": tag, "item_id": it["item_id"], "sub": it["sub"],
               "voice_id": v, "text_id": it["text_id"], "text": it["text"],
               "seed": seed, "ref_sample_id": ref["ref_sample_id"], "out_wav": out}
        for k in ("target_cpm", "emotion", "emotion_label", "intonation_type",
                  "pause_setting", "pause_text", "base_text_id", "base_text",
                  "pinyin_hints", "hint_text", "replicate",
                  "negative_control", "negative_note"):
            if k in it:
                row[k] = it[k]
        try:
            base = get_baseline(v, ref) if it["sub"] == "rate" else None
            row["baseline_cpm"] = base
            C.vram_reset()
            torch.manual_seed(seed)
            ts = time.time()
            wav, sr, extra = K.synth_control(cid, handle, it, ref, out, base)
            wall = time.time() - ts
            if wav is None or len(wav) == 0:
                raise RuntimeError("empty waveform")
            if not extra.get("written_by_model"):
                C.write_wav(out, wav, sr)
            dur = len(wav) / float(sr)
            s0, s1 = C.speech_span(wav, sr)
            span = max(s1 - s0, 1e-6)
            nch = C.norm_chars(it["text"])
            cpm = nch / span * 60
            row.update({"status": "ok", "sample_rate": sr,
                        "duration_s": round(dur, 4), "speech_span_s": round(span, 4),
                        "norm_chars": nch, "actual_cpm": round(cpm, 2),
                        "gen_wall_s": round(wall, 3),
                        "rtf": round(wall / dur, 4) if dur else None,
                        "peak_vram_gib": C.vram_peak_gib(), "rss_gib": C.rss_gib(),
                        "ts": time.time()})
            row.update(extra)
            if it["sub"] == "rate" and it.get("target_cpm"):
                row["rate_rel_err_pct"] = round(
                    abs(cpm - it["target_cpm"]) / it["target_cpm"] * 100, 2)
            n_ok += 1
        except Exception as e:
            row.update({"status": "failed", "error_type": type(e).__name__,
                        "error_msg": str(e)[:1200],
                        "traceback": traceback.format_exc()[-1500:],
                        "peak_vram_gib": C.vram_peak_gib(), "ts": time.time()})
            n_fail += 1
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        fout.flush()
        if i % 20 == 0 or i == len(items):
            C.log(f"[{tag} {cid}] {i}/{len(items)} ok={n_ok} fail={n_fail} "
                  f"vram={C.vram_peak_gib()}GiB rss={C.rss_gib()}GiB")
    fout.close()
    C.release(handle)
    C.log(f"[{tag} {cid}] DONE ok={n_ok} fail={n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())