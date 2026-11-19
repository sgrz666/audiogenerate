#!/usr/bin/env python3
"""P3 stability: repeat generations with different seeds and measure spread.

Track A's 稳定性 (5%) needs repeated generation, which P0/P1/P2 do not provide.
For a stratified subset of P1 texts, generate N times per (config, voice, text)
with distinct seeds and record the spread of duration / chars-per-minute /
speaker-embedding similarity. One config per process, serial.

Usage: mt_run_p3.py --config voxcpm2 [--reps 3] [--texts 6]
"""
import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--texts", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = C.cfg_by_id(args.config)
    cid = cfg["config_id"]
    # reuse P1 texts: spread across levels + pronunciation specials
    texts = [json.loads(l) for l in
             open(os.path.join(C.E, "texts", "all.jsonl"), encoding="utf-8")]
    p1 = [t for t in texts if t["phase"] == "P1"]
    chosen, seen_lv = [], set()
    for t in p1:
        if t["level"] > 0 and t["level"] not in seen_lv:
            chosen.append(t); seen_lv.add(t["level"])
        if len(chosen) >= args.texts - 2:
            break
    chosen += [t for t in p1 if t["level"] == 0][:2]
    chosen = chosen[: args.texts]
    refs = C.load_refs()

    import torch
    torch.set_num_threads(min(8, os.cpu_count() or 8))
    import mt_adapters as A

    rp = C.result_path(cid, "P3")
    C.log(f"[P3 {cid}] texts={len(chosen)} reps={args.reps} voices={len(C.VOICES)}")
    try:
        handle = A.LOAD[cid](cfg, args.device)
    except Exception as e:
        with open(rp, "a", encoding="utf-8") as f:
            f.write(json.dumps({"config_id": cid, "status": "failed", "stage": "load",
                                "error_type": type(e).__name__, "error_msg": str(e)[:800]},
                               ensure_ascii=False) + "\n")
        return 2

    fout = open(rp, "a", encoding="utf-8")
    n_ok = n_fail = 0
    for t in chosen:
        for v in C.VOICES:
            ref = refs[v]
            runs = []
            for rep in range(args.reps):
                seed = 42 + rep
                out = C.out_path(cid, "P3", v, f"{t['text_id']}_r{rep}")
                row = {"config_id": cid, "phase": "P3", "text_id": t["text_id"],
                       "voice_id": v, "rep": rep, "seed": seed, "text": t["text"],
                       "out_wav": out}
                try:
                    C.vram_reset()
                    ts = time.time()
                    wav, sr, extra = A.SYNTH[cid](handle, t["text"], ref, out,
                                                 {"seed": seed})
                    wall = time.time() - ts
                    if not extra.get("written_by_model"):
                        C.write_wav(out, wav, sr)
                    dur = len(wav) / float(sr)
                    s0, s1 = C.speech_span(wav, sr)
                    span = max(s1 - s0, 1e-6)
                    row.update({"status": "ok", "duration_s": round(dur, 4),
                                "speech_span_s": round(span, 4),
                                "chars_per_min": round(C.norm_chars(t["text"]) / span * 60, 2),
                                "gen_wall_s": round(wall, 3),
                                "peak_vram_gib": C.vram_peak_gib(), "ts": time.time()})
                    runs.append(row)
                    n_ok += 1
                except Exception as e:
                    row.update({"status": "failed", "error_type": type(e).__name__,
                                "error_msg": str(e)[:800]})
                    n_fail += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
            # spread summary across successful reps
            good = [r for r in runs if r.get("status") == "ok"]
            if len(good) >= 2:
                import statistics as st
                cpms = [r["chars_per_min"] for r in good]
                durs = [r["duration_s"] for r in good]
                summ = {"config_id": cid, "phase": "P3", "text_id": t["text_id"],
                        "voice_id": v, "status": "summary", "n_reps": len(good),
                        "cpm_mean": round(st.mean(cpms), 2),
                        "cpm_stdev": round(st.stdev(cpms), 2),
                        "cpm_cv_pct": round(st.stdev(cpms) / st.mean(cpms) * 100, 2),
                        "dur_stdev_s": round(st.stdev(durs), 3)}
                fout.write(json.dumps(summ, ensure_ascii=False) + "\n")
                fout.flush()
    fout.close()
    C.release(handle)
    C.log(f"[P3 {cid}] DONE ok={n_ok} fail={n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())