#!/usr/bin/env python3
"""Build the P2b repair set.

  pause:        5 configs x 3 voices x 3 settings x 3 texts x 3 seeds = 405
  pronunciation: 5? only IndexTTS needs hints; 6 texts x 3 voices = 18
  pronunciation negative controls: 2 texts x 3 voices = 6

Pause text variants are the whole point of the repair: the v1 builder defined
PAUSE_SETTINGS marks but never inserted them, so all three settings used the
same sentence.  Here:
  none   : remove internal commas
  comma  : keep the original text
  period : replace internal commas with full stops

Pronunciation hints use IndexTTS's <字|PINYIN+TONE> syntax (e.g. 银<行|HANG2>)
and are validated against repos/index-tts/checkpoints/pinyin.vocab.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_common as C  # noqa: E402
from build_p2 import (PAUSE_TEXTS, PRON_TEXTS, validate_hints)  # noqa: E402

VALID_CONFIGS = ["voxcpm2", "indextts25", "qwen3tts_base",
                 "fireredtts3_base", "fireredtts3_instruct"]
SEEDS = [42, 43, 44]
PAUSE_LABELS = {"none": "无额外停顿", "comma": "分句间停顿", "period": "句间停顿"}


def pause_variant(text, setting):
    if setting == "none":
        return text.replace("，", "")
    if setting == "comma":
        return text
    if setting == "period":
        return text.replace("，", "。")
    raise ValueError(setting)


def add(items, **kw):
    items.append(kw)


def build():
    items = []
    for cfg in VALID_CONFIGS:
        for voice in C.VOICES:
            for setting in ("none", "comma", "period"):
                for tid, base in PAUSE_TEXTS:
                    txt = pause_variant(base, setting)
                    for rep, seed in enumerate(SEEDS, 1):
                        add(items,
                            item_id=f"P2b-pause-{cfg}-{voice}-{tid}-{setting}-r{rep}",
                            phase="P2b", sub="pause", config_id=cfg,
                            voice_id=voice, text_id=f"{tid}-{setting}",
                            base_text_id=tid, text=txt, pause_text=txt,
                            base_text=base, pause_setting=setting,
                            pause_label=PAUSE_LABELS[setting],
                            replicate=rep, seed=seed, control="pause")
    # Pronunciation only for IndexTTS (the only native hint interface).
    for tid, base, hints in PRON_TEXTS:
        validate_hints(hints)
        for voice in C.VOICES:
            add(items,
                item_id=f"P2b-pron-indextts25-{voice}-{tid}",
                phase="P2b", sub="pronunciation", config_id="indextts25",
                voice_id=voice, text_id=tid, text=base,
                pinyin_hints=hints, control="pronunciation", seed=42)
    # Deliberately wrong readings: used to measure chance / false-accept.
    neg = {
        "PR1": ("行|XING2;长|CHANG2", "银行的行故意读 xíng"),
        "PR4": ("还|HUAN2;还|HAI2;还|HAI2", "还有/没还故意对调"),
    }
    for tid, (hints, note) in neg.items():
        text = dict((t, s) for t, s, _ in PRON_TEXTS)[tid]
        validate_hints(hints)
        for voice in C.VOICES:
            add(items,
                item_id=f"P2b-pronneg-indextts25-{voice}-{tid}",
                phase="P2b", sub="pronunciation", config_id="indextts25",
                voice_id=voice, text_id=f"{tid}-NEG", text=text,
                pinyin_hints=hints, control="pronunciation",
                negative_control=True, negative_note=note, seed=42)

    # Pilot: PR2 + PR4, one voice, IndexTTS only.
    pilot = [i for i in items if i["config_id"] == "indextts25"
             and i["sub"] == "pronunciation" and i["voice_id"] == "NARRATOR"
             and i["text_id"] in ("PR2", "PR4")]
    return items, pilot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="texts/p2b.jsonl")
    ap.add_argument("--pilot-out", default="texts/p2b_pilot.jsonl")
    args = ap.parse_args()
    items, pilot = build()
    from collections import Counter
    cnt = Counter((i["sub"], i["config_id"]) for i in items)
    n_pause = sum(1 for i in items if i["sub"] == "pause")
    n_pron = sum(1 for i in items if i["sub"] == "pronunciation"
                 and not i.get("negative_control"))
    n_neg = sum(1 for i in items if i.get("negative_control"))
    assert n_pause == 405, n_pause
    assert n_pron == 18, n_pron
    assert n_neg == 6, n_neg
    assert len(pilot) == 2, [p["item_id"] for p in pilot]
    # Every pause triple must have three distinct texts.
    by = {}
    for i in items:
        if i["sub"] == "pause":
            by.setdefault((i["config_id"], i["voice_id"], i["base_text_id"]),
                          set()).add(i["text"])
    for k, v in by.items():
        assert len(v) == 3, (k, v)
    out = os.path.join(C.E, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i in items:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    pout = os.path.join(C.E, args.pilot_out)
    with open(pout, "w", encoding="utf-8") as f:
        for i in pilot:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    C.log(f"P2b items={len(items)} -> {out}")
    for k, v in sorted(cnt.items()):
        C.log(f"  {k[0]:14s} {k[1]:22s} {v}")
    C.log(f"pilot={len(pilot)} -> {pout}")
    C.log("counts: pause=405 pron=18 neg=6; pause triples distinct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
