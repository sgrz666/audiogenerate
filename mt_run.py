#!/usr/bin/env python3
"""Single-config inference worker. Runs in its own process; exit frees all VRAM/RAM.

Usage: mt_run.py --config voxcpm2 --phase P0 [--limit N] [--device cuda:0]
"""
import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=["P0", "P1", "P2", "P3"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = C.cfg_by_id(args.config)
    cid = cfg["config_id"]
    texts = C.load_texts(args.phase)
    refs = C.load_refs()
    rp = C.result_path(cid, args.phase)

    # resume: keep only rows already ok
    done = set()
    if os.path.isfile(rp):
        with open(rp, encoding="utf-8") as f:
            for ln in f:
                try:
                    d = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if d.get("status") == "ok":
                    done.add((d["text_id"], d["voice_id"]))

    import torch
    torch.set_num_threads(min(8, os.cpu_count() or 8))
    C.log(f"[{cid}] phase={args.phase} texts={len(texts)} voices={len(refs)} "
          f"already_ok={len(done)} device={args.device}")
    C.log(f"[{cid}] gpu={torch.cuda.get_device_name(0)} "
          f"total={torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GiB "
          f"rss={C.rss_gib()}GiB")

    import mt_adapters as A
    base = {"config_id": cid, "family": cfg["family"], "variant": cfg["variant"],
            "phase": args.phase, "model_dir": cfg["model_dir"],
            "device": args.device, "seed": args.seed,
            "vram_method": "torch.cuda.max_memory_allocated (NVML unavailable)",
            "span_method": "energy RMS 20ms frames, rel_thresh=0.02, head/tail trimmed"}

    t0 = time.time()
    try:
        C.vram_reset()
        handle = A.LOAD[cid](cfg, args.device)
        load_s = round(time.time() - t0, 2)
        load_vram = C.vram_peak_gib()
        C.log(f"[{cid}] model loaded in {load_s}s peak_vram={load_vram}GiB "
              f"rss={C.rss_gib()}GiB")
    except Exception as e:
        row = dict(base, text_id="", voice_id="", status="failed",
                   stage="load", error_type=type(e).__name__,
                   error_msg=str(e)[:1500], traceback=traceback.format_exc()[-2500:],
                   load_s=round(time.time() - t0, 2), ts=time.time())
        with open(rp, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        C.log(f"[{cid}] LOAD FAILED {type(e).__name__}: {str(e)[:300]}")
        return 2

    n_ok = n_fail = 0
    fout = open(rp, "a", encoding="utf-8")
    items = [(t, v) for t in texts for v in C.VOICES]
    if args.limit:
        items = items[: args.limit]
    for i, (t, v) in enumerate(items, 1):
        tid = t["text_id"]
        if (tid, v) in done:
            continue
        ref = refs[v]
        out = C.out_path(cid, args.phase, v, tid)
        row = dict(base, text_id=tid, voice_id=v, level=t["level"],
                   category=t["category"], text=t["text"],
                   pinyin_hint=t.get("pinyin_hint", ""),
                   text_source=t["source"], review_status=t["review_status"],
                   ref_sample_id=ref["ref_sample_id"],
                   ref_wav=ref["ref_wav"], out_wav=out,
                   load_s=load_s, load_vram_gib=load_vram)
        try:
            C.vram_reset()
            ts = time.time()
            wav, sr, extra = A.SYNTH[cid](handle, t["text"], ref, out,
                                         {"seed": args.seed})
            wall = time.time() - ts
            if wav is None or len(wav) == 0:
                raise RuntimeError("empty waveform returned")
            if not extra.get("written_by_model"):
                C.write_wav(out, wav, sr)
            dur = len(wav) / float(sr)
            s0, s1 = C.speech_span(wav, sr)
            span = max(s1 - s0, 1e-6)
            nch = C.norm_chars(t["text"])
            row.update({
                "status": "ok", "sample_rate": sr,
                "duration_s": round(dur, 4),
                "speech_start_s": round(s0, 4), "speech_end_s": round(s1, 4),
                "speech_span_s": round(span, 4),
                "head_silence_s": round(s0, 4),
                "tail_silence_s": round(max(dur - s1, 0), 4),
                "norm_chars": nch, "han_chars": C.han_count(t["text"]),
                "chars_per_min": round(nch / span * 60, 2),
                "gen_wall_s": round(wall, 3),
                "rtf": round(wall / dur, 4) if dur > 0 else None,
                "peak_vram_gib": C.vram_peak_gib(),
                "reserved_vram_gib": C.vram_reserved_gib(),
                "rss_gib": C.rss_gib(), "ts": time.time(),
            })
            row.update(extra)
            n_ok += 1
        except NotImplementedError as e:
            row.update(status="unsupported", error_type="NotImplementedError",
                       error_msg=str(e)[:800], ts=time.time())
            n_fail += 1
        except Exception as e:
            row.update(status="failed", error_type=type(e).__name__,
                       error_msg=str(e)[:1500],
                       traceback=traceback.format_exc()[-2000:],
                       peak_vram_gib=C.vram_peak_gib(), rss_gib=C.rss_gib(),
                       ts=time.time())
            n_fail += 1
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        fout.flush()
        if i % 10 == 0 or i == len(items):
            C.log(f"[{cid}] {i}/{len(items)} ok={n_ok} fail={n_fail} "
                  f"vram={C.vram_peak_gib()}GiB rss={C.rss_gib()}GiB")

    fout.close()
    C.release(handle)
    C.log(f"[{cid}] DONE ok={n_ok} fail={n_fail} "
          f"final_rss={C.rss_gib()}GiB vram_after_release={C.vram_reserved_gib()}GiB")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())