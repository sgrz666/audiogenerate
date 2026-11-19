#!/usr/bin/env python3
"""Objective metrics for the TTS comparison: speaker similarity, CER, aggregates.

Speaker similarity uses CAM++ with the weights ALREADY on disk
(model/IndexTTS-2.5/hf_cache/campplus_cn_common.bin) and the CAMPPlus class from
the index-tts repo. Feature recipe copied from indextts/infer_v2_5.py:640 so the
measurement basis matches how the corpus speaker labels were produced:
    kaldi.fbank(16k, num_mel_bins=80, dither=0) -> subtract per-utt mean -> CAM++

CER needs an ASR model. If none is available the CER column is recorded as
not_tested rather than silently filled with a guess.

Usage:
  mt_metrics.py --phase P0 --speaker-only
  mt_metrics.py --phase P1 --asr-dir /path/to/asr
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_common as C  # noqa: E402

CAMPPLUS_BIN = C.MODEL + "/IndexTTS-2.5/hf_cache/campplus_cn_common.bin"
PUNCT = r"[\s，。！？、；：""''（）《》【】…—～·,.!?;:\"'()<>\[\]{}\-~/\\|@#$%^&*_+=`]"


def norm_text(s, t2s=False):
    s = s.strip()
    if t2s:
        try:
            from zhconv import convert
            s = convert(s, "zh-cn")
        except ImportError:
            pass
    s = re.sub(PUNCT, "", s)
    return s.lower()


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class SpeakerEncoder:
    def __init__(self, device="cuda:0"):
        import torch
        import torchaudio
        self.torch, self.ta = torch, torchaudio
        sys.path.insert(0, os.path.join(C.E, "repos", "index-tts"))
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
        self.model = CAMPPlus(feat_dim=80, embedding_size=192)
        sd = torch.load(CAMPPLUS_BIN, map_location="cpu")
        self.model.load_state_dict(sd)
        self.model.eval().to(device)
        self.device = device

    def emb(self, wav_path):
        torch, ta = self.torch, self.ta
        w, sr = C.read_wav(wav_path)
        t = torch.from_numpy(w).float().unsqueeze(0)
        if sr != 16000:
            t = ta.functional.resample(t, sr, 16000)
        with torch.no_grad():
            feat = ta.compliance.kaldi.fbank(t.to(self.device), num_mel_bins=80,
                                             dither=0, sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            e = self.model(feat.unsqueeze(0))
        return torch.nn.functional.normalize(e, dim=-1).squeeze(0).cpu().numpy()


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--configs", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--speaker-only", action="store_true")
    ap.add_argument("--asr-dir", default="")
    args = ap.parse_args()

    refs = C.load_refs()
    enc = SpeakerEncoder(args.device)
    ref_emb = {}
    for v, r in refs.items():
        ref_emb[v] = enc.emb(r["ref_wav"])
        C.log(f"[spk] reference {v} embedded from {r['ref_sample_id']}")

    asr = None
    if not args.speaker_only and args.asr_dir:
        try:
            from funasr import AutoModel
            asr = AutoModel(model=args.asr_dir, disable_update=True,
                            device=args.device)
            C.log(f"[asr] loaded {args.asr_dir}")
        except Exception as e:
            C.log(f"[asr] UNAVAILABLE {type(e).__name__}: {str(e)[:200]} "
                  f"-> CER recorded as not_tested")

    cfgs = C.CONFIGS
    if args.configs:
        want = [x.strip() for x in args.configs.split(",")]
        cfgs = [c for c in cfgs if c["config_id"] in want]

    summary = []
    for c in cfgs:
        cid = c["config_id"]
        rp = C.result_path(cid, args.phase)
        if not os.path.isfile(rp):
            C.log(f"[{cid}] no results -> not_tested")
            summary.append({"config_id": cid, "status": "not_tested"})
            continue
        rows = [json.loads(l) for l in open(rp, encoding="utf-8") if l.strip()]
        # Dedup by (text_id, voice_id): re-runs append rows, and earlier failed
        # attempts must not inflate n_rows / deflate success_rate. Keep ok over
        # failed; among ok keep the most recent. Load-failure rows (no text_id)
        # are kept as-is so a config that never loaded is still visible.
        best = {}
        load_fail = []
        for r in rows:
            k = (r.get("text_id"), r.get("voice_id"))
            if not r.get("text_id"):
                load_fail.append(r)
                continue
            prev = best.get(k)
            if prev is None:
                best[k] = r
            elif r.get("status") == "ok" and prev.get("status") != "ok":
                best[k] = r
            elif r.get("status") == "ok" and prev.get("status") == "ok":
                if r.get("ts", 0) >= prev.get("ts", 0):
                    best[k] = r
        rows = list(best.values()) + load_fail
        ok = [r for r in rows if r.get("status") == "ok"]
        sims, cers, cpms, rtfs, vrams = [], [], [], [], []
        outp = os.path.join(C.E, "results", args.phase, cid + ".metrics.jsonl")
        with open(outp, "w", encoding="utf-8") as fo:
            for r in ok:
                m = {"text_id": r["text_id"], "voice_id": r["voice_id"]}
                try:
                    e = enc.emb(r["out_wav"])
                    m["spk_sim"] = round(cos(e, ref_emb[r["voice_id"]]), 4)
                    sims.append(m["spk_sim"])
                except Exception as ex:
                    m["spk_sim_error"] = f"{type(ex).__name__}: {str(ex)[:150]}"
                if asr is not None:
                    try:
                        # use_itn=False: reference texts are in Chinese-character
                        # form ("一三八…", "三月五号"). With ITN on, ASR rewrites
                        # correct speech to "138…"/"3月5号" and inflates CER
                        # (measured: P0-07 CER 0.611 despite correct reading).
                        res = asr.generate(input=r["out_wav"], cache={},
                                           language="zh", use_itn=False,
                                           batch_size_s=60)
                        hyp = res[0]["text"] if res else ""
                        # SenseVoice emits rich-text tags like <|zh|><|NEUTRAL|>
                        hyp = re.sub(r"<\|[^|]*\|>", "", hyp)
                        ref_t = norm_text(r["text"], t2s=True)
                        hyp_t = norm_text(hyp, t2s=True)
                        d = edit_distance(ref_t, hyp_t)
                        m["asr_hyp"] = hyp
                        m["cer"] = round(d / max(len(ref_t), 1), 4)
                        cers.append(m["cer"])
                    except Exception as ex:
                        m["cer_error"] = f"{type(ex).__name__}: {str(ex)[:150]}"
                else:
                    m["cer"] = None
                    m["cer_status"] = "not_tested"
                if r.get("chars_per_min"):
                    cpms.append(r["chars_per_min"])
                if r.get("rtf") is not None:
                    rtfs.append(r["rtf"])
                if r.get("peak_vram_gib") is not None:
                    vrams.append(r["peak_vram_gib"])
                fo.write(json.dumps(m, ensure_ascii=False) + "\n")

        def agg(x):
            return (round(float(np.mean(x)), 4) if x else None)

        s = {"config_id": cid, "family": c["family"], "variant": c["variant"],
             "phase": args.phase, "n_rows": len(rows), "n_ok": len(ok),
             "n_failed": sum(1 for r in rows if r.get("status") == "failed"),
             "n_unsupported": sum(1 for r in rows if r.get("status") == "unsupported"),
             "success_rate": round(len(ok) / len(rows), 4) if rows else None,
             "mean_spk_sim": agg(sims), "mean_cer": agg(cers),
             "cer_status": "measured" if cers else "not_tested",
             "mean_chars_per_min": agg(cpms), "mean_rtf": agg(rtfs),
             "max_peak_vram_gib": round(max(vrams), 3) if vrams else None,
             "fits_24gb": (max(vrams) <= 24.0) if vrams else None,
             "metrics_file": outp}
        summary.append(s)
        C.log(f"[{cid}] ok={len(ok)}/{len(rows)} sim={s['mean_spk_sim']} "
              f"cer={s['mean_cer']}({s['cer_status']}) "
              f"cpm={s['mean_chars_per_min']} rtf={s['mean_rtf']} "
              f"vram={s['max_peak_vram_gib']}GiB fits24={s['fits_24gb']}")

    sp = os.path.join(C.E, "results", f"metrics_{args.phase}.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    C.log(f"metrics -> {sp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())