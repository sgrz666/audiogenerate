#!/usr/bin/env python3
"""Build an ANONYMOUS blind-listening package from P1 outputs (CPU only).

The plan requires 自然度/语气/情绪 to be judged by anonymous human listening, and
says the existing review Skill can locate errors but cannot replace human judgement.
This prepares the package; it does NOT produce any subjective scores.

  blind/clips/S001.wav ...      opaque ids, model identity hidden
  blind/refs/<voice>.wav        reference timbre for comparison
  blind/rating_sheet.csv        rater fills scores; no model column
  blind/answer_key.json         id -> (config, text, voice); kept separate

Stratified sample: one text per HSK level 1/3/5/6 plus two pronunciation specials,
crossed with the 3 voices and every config that has P1 output.
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_common as C  # noqa: E402

SEED = 20261111
DIMS = ["naturalness", "intonation", "emotion_appropriateness",
        "voice_consistency", "audio_quality", "content_pronunciation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="P1")
    ap.add_argument("--per-level", type=int, default=1)
    args = ap.parse_args()

    texts = [json.loads(l) for l in
             open(os.path.join(C.E, "texts", "all.jsonl"), encoding="utf-8")]
    by_level = {}
    for t in texts:
        if t["phase"] != args.phase:
            continue
        by_level.setdefault(t["level"], []).append(t)
    rng = random.Random(SEED)
    chosen = []
    for lv in (1, 3, 5, 6):
        pool = by_level.get(lv, [])
        if pool:
            chosen.extend(rng.sample(pool, min(args.per_level, len(pool))))
    pron = [t for t in by_level.get(0, []) if t["category"].startswith("多音字")]
    chosen.extend(pron[:2])
    if not chosen:
        raise SystemExit("no texts selected")

    # collect ok rows per config
    rows = []
    for c in C.CONFIGS:
        p = C.result_path(c["config_id"], args.phase)
        if not os.path.isfile(p):
            continue
        best = {}
        for l in open(p, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("status") == "ok":
                best[(r["text_id"], r["voice_id"])] = r
        for t in chosen:
            for v in C.VOICES:
                r = best.get((t["text_id"], v))
                if r and os.path.isfile(r["out_wav"]):
                    rows.append((c["config_id"], t, v, r["out_wav"]))
    if not rows:
        raise SystemExit("no P1 audio available yet; run P1 first")

    rng.shuffle(rows)
    bdir = os.path.join(C.E, "blind")
    cdir = os.path.join(bdir, "clips")
    os.makedirs(cdir, exist_ok=True)
    rdir = os.path.join(bdir, "refs")
    os.makedirs(rdir, exist_ok=True)
    refs = C.load_refs()
    for v, r in refs.items():
        shutil.copy2(r["ref_wav"], os.path.join(rdir, v + ".wav"))

    key, sheet = [], []
    for i, (cid, t, v, src) in enumerate(rows, 1):
        sid = f"S{i:03d}"
        dst = os.path.join(cdir, sid + ".wav")
        shutil.copy2(src, dst)
        key.append({"blind_id": sid, "config_id": cid, "text_id": t["text_id"],
                    "voice_id": v, "text": t["text"], "level": t["level"],
                    "category": t["category"], "source_wav": src})
        sheet.append({"blind_id": sid, "text": t["text"], "voice": v,
                      "reference_clip": f"refs/{v}.wav",
                      **{d: "" for d in DIMS}, "notes": ""})

    with open(os.path.join(bdir, "answer_key.json"), "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=1)
    cols = ["blind_id", "text", "voice", "reference_clip"] + DIMS + ["notes"]
    with open(os.path.join(bdir, "rating_sheet.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sheet)

    with open(os.path.join(bdir, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# 匿名盲听包\n\n"
            f"共 {len(rows)} 条，来自 {len(set(k['config_id'] for k in key))} 个配置。\n"
            "clip 文件名是随机不透明编号，**不含模型信息**；对照表在 answer_key.json，"
            "评分完成前不要打开。\n\n"
            "## 评分方法\n"
            "每条按 1–5 分打六个维度（naturalness 自然度 / intonation 语气 / "
            "emotion_appropriateness 情绪适切 / voice_consistency 音色一致 / "
            "audio_quality 音质 / content_pronunciation 发音与内容）。\n"
            "音色一致性请对照 refs/<voice>.wav 参考音色判断。\n"
            "考试播报以清晰适度为准，不因缺乏戏剧性扣分。\n\n"
            "## 注意\n"
            "参考音色本身跨试卷、身份未人工确认；语料文本未经人工校对，"
            "发音评分以听感为准，文本只作线索。\n"
            "本包只提供材料，不产生任何主观分数。\n")

    C.log(f"blind package: {len(rows)} clips, "
          f"{len(set(k['config_id'] for k in key))} configs -> {bdir}")
    C.log("rater fills rating_sheet.csv; answer_key.json stays sealed until done")
    return 0


if __name__ == "__main__":
    sys.exit(main())