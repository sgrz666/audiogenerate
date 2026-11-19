#!/usr/bin/env python3
"""Build the P2 control-experiment item set (918 items, per the plan).

Sub-experiments and counts:
  rate      6 configs x 3 voices x 5 levels x 3 texts = 270
  pause     6 configs x 3 voices x 3 settings x 3 texts = 162
  intonation6 configs x 3 voices x 3 types  x 3 texts = 162
  emotion   6 configs x 3 voices x 4 emotions x 3 texts = 216
  pronunciation 6 configs x 3 voices x 6 texts = 108
                                             total = 918

Items carry the *target* parameter; whether a config can honour it natively is
decided at run time from the verified interface matrix (see CONTROL.md) and
recorded as ok / unsupported / failed. Nothing is silently skipped.

Rate targets are in 字/分钟. IndexTTS duration_factor and CosyVoice speed are
multipliers, so the runner must first measure a baseline and convert; the item
records the target cpm and the conversion is logged, not assumed.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_common as C  # noqa: E402

RATE_CPM = [160, 200, 240, 280, 320]
RATE_TEXTS = [
    ("R1", "今天的天气很好，我们一起去公园散步吧。"),
    ("R2", "请你把这本书翻到第二十五页，然后开始朗读。"),
    ("R3", "虽然外面下着大雨，但是他还是坚持按时到了学校。"),
]
PAUSE_SETTINGS = [
    ("none", "无额外停顿", ""),
    ("comma", "分句间停顿", "，"),
    ("period", "句间停顿", "。"),
]
PAUSE_TEXTS = [
    ("P1", "春天来了，花都开了，小草也绿了。"),
    ("P2", "他先去了图书馆，然后去了食堂，最后回了宿舍。"),
    ("P3", "如果你明天有空，我们就一起去看看那场电影。"),
]
INTONATION = [
    ("declarative", "陈述", "你明天去图书馆。"),
    ("interrogative", "疑问", "你明天去图书馆吗？"),
    ("exclamatory", "感叹", "你明天居然去图书馆！"),
]
INTO_TEXTS = [
    ("I1", "他昨天买了一本新书。"),
    ("I2", "这道题其实一点都不难。"),
    ("I3", "我们终于完成了这个任务。"),
]
# task book's four emotions; IndexTTS 8-dim order is
# [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
EMOTIONS = [
    ("calm", "平静", 7, "用平静、平稳的语气朗读。"),
    ("happy", "高兴", 0, "用高兴、愉快的语气朗读。"),
    ("sad", "悲伤", 2, "用悲伤、低落的语气朗读。"),
    ("angry", "生气", 1, "用生气、不满的语气朗读。"),
]
EMO_TEXTS = [
    ("E1", "这件事终于有了结果。"),
    ("E2", "他把我的书弄坏了。"),
    ("E3", "明天就要考试了。"),
]
PRON_TEXTS = [
    # IndexTTS syntax is <字|PINYIN+TONE>, e.g. 银<行|HANG2>; the vocab is
    # repos/index-tts/checkpoints/pinyin.vocab.  Old tone-mark hints (xíng)
    # were read as letters and corrupted the audio.  Hints are consumed in
    # text order, one occurrence per hint.
    ("PR1", "他骑车去银行，路上走了很长时间。", "行|HANG2;长|CHANG2"),
    ("PR2", "这个箱子太重了，请你重新搬一次。", "重|ZHONG4;重|CHONG2"),
    ("PR3", "你得把作业写完，才能出去玩得开心。", "得|DEI3;得|DE5"),
    ("PR4", "他还有一本书没还给我，说明天还。", "还|HAI2;还|HUAN2;还|HUAN2"),
    ("PR5", "我们都去过首都，那里的人都很热情。", "都|DOU1;都|DU1;都|DOU1"),
    ("PR6", "我觉得睡觉以前应该少看手机。", "觉|JVE2;觉|JIAO4"),
]
PINYIN_VOCAB = os.path.join(C.E, "repos", "index-tts", "checkpoints", "pinyin.vocab")


def validate_hints(hints):
    vocab = set(open(PINYIN_VOCAB, encoding="utf-8").read().split())
    for pair in hints.split(";"):
        w, r = pair.split("|", 1)
        assert r.strip().upper() in vocab, \
            f"pinyin {r!r} for {w!r} not in {PINYIN_VOCAB}"


def main():
    items = []
    for tid, _, hints in PRON_TEXTS:
        validate_hints(hints)

    def add(sub, cfg, voice, tid, text, **kw):
        items.append({"item_id": f"P2-{sub}-{cfg}-{voice}-{tid}",
                      "phase": "P2", "sub": sub, "config_id": cfg,
                      "voice_id": voice, "text_id": tid, "text": text, **kw})

    for c in C.CONFIGS:
        cid = c["config_id"]
        for v in C.VOICES:
            for cpm in RATE_CPM:
                for tid, t in RATE_TEXTS:
                    add("rate", cid, v, f"{tid}-{cpm}", t,
                        target_cpm=cpm, control="speed")
            for setting, label, mark in PAUSE_SETTINGS:
                for tid, t in PAUSE_TEXTS:
                    add("pause", cid, v, f"{tid}-{setting}", t,
                        pause_setting=setting, pause_label=label, control="pause")
            for itype, label, _ in INTONATION:
                for tid, t in INTO_TEXTS:
                    # apply the sentence-type punctuation to the base text stem
                    stem = t.rstrip("。！？")
                    punct = {"declarative": "。", "interrogative": "吗？",
                             "exclamatory": "！"}[itype]
                    txt = stem + punct if itype != "interrogative" else stem + "吗？"
                    add("intonation", cid, v, f"{tid}-{itype}", txt,
                        intonation_type=itype, intonation_label=label,
                        control="intonation")
            for emo, label, idx, instr in EMOTIONS:
                for tid, t in EMO_TEXTS:
                    vec = [0.0] * 8
                    vec[idx] = 0.8
                    add("emotion", cid, v, f"{tid}-{emo}", t,
                        emotion=emo, emotion_label=label,
                        emo_vector=vec, instruct_text=instr, control="emotion")
            for tid, t, hints in PRON_TEXTS:
                add("pronunciation", cid, v, tid, t,
                    pinyin_hints=hints, control="pronunciation")

    out = os.path.join(C.E, "texts", "p2.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter
    cnt = Counter(i["sub"] for i in items)
    C.log(f"P2 items: {len(items)} -> {out}")
    for k, v in sorted(cnt.items()):
        C.log(f"  {k:14s} {v}")
    assert len(items) == 918, f"expected 918, got {len(items)}"
    C.log("count matches plan (918)")
    return 0


if __name__ == "__main__":
    sys.exit(main())