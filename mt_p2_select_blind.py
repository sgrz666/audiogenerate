#!/usr/bin/env python3
"""Create a 30-clip stratified P2 human-review spec (no model names in HTML).

Covers: pronunciation 8 (6 hint + 2 negative), pause 10 (5 configs x none/period),
intonation 6 (IndexTTS + FireRed Base x 3 types), emotion 6 (IndexTTS E1 four
emotions + E2 happy/sad).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_common as C  # noqa: E402

PAUSE_TEXT = {
    "none": "春天来了花都开了小草也绿了。",
    "period": "春天来了。花都开了。小草也绿了。",
}
INTO_TEXT = {
    "declarative": "这道题其实一点都不难。",
    "interrogative": "这道题其实一点都不难吗？",
    "exclamatory": "这道题其实一点都不难！",
}
PRON_TEXT = {
    "PR1": ("他骑车去银行，路上走了很长时间。", "行 háng / 长 cháng"),
    "PR2": ("这个箱子太重了，请你重新搬一次。", "重 zhòng / 重 chóng"),
    "PR3": ("你得把作业写完，才能出去玩得开心。", "得 děi / 得 de"),
    "PR4": ("他还有一本书没还给我，说明天还。", "还 hái / 还 huán / 还 huán"),
    "PR5": ("我们都去过首都，那里的人都很热情。", "都 dōu / 都 dū / 都 dōu"),
    "PR6": ("我觉得睡觉以前应该少看手机。", "觉 jué / 觉 jiào"),
}


def add(spec, blind_id, path, text, target, kind):
    if os.path.isfile(path):
        spec.append({"blind_id": blind_id, "audio_path": path,
                     "source_text": text, "target": target, "kind": kind,
                     "model": os.path.basename(os.path.dirname(os.path.dirname(path)))})
    else:
        print(f"[skip] {path}")


def main():
    spec = []
    # pronunciation: 6 hint + 2 negative
    for tid in ["PR1", "PR2", "PR3", "PR4", "PR5", "PR6"]:
        p = os.path.join(C.E, "audio", "P2b", "indextts25", "NARRATOR",
                         f"P2b-pron-indextts25-NARRATOR-{tid}.wav")
        text, target = PRON_TEXT[tid]
        add(spec, f"pron_{tid}", p, text, f"读音提示后：{target}", "pron")
    for tid in ["PR1", "PR4"]:
        p = os.path.join(C.E, "audio", "P2b", "indextts25", "NARRATOR",
                         f"P2b-pronneg-indextts25-NARRATOR-{tid}.wav")
        text, _ = PRON_TEXT[tid]
        add(spec, f"pronneg_{tid}", p, text, "故意错读负对照（不计入正样本）", "pron_neg")
    # pause: 5 configs x none/period
    for cfg in ["voxcpm2", "indextts25", "qwen3tts_base",
                "fireredtts3_base", "fireredtts3_instruct"]:
        for setting in ["none", "period"]:
            p = os.path.join(C.E, "audio", "P2b", cfg, "NARRATOR",
                             f"P2b-pause-{cfg}-NARRATOR-P1-{setting}-r1.wav")
            add(spec, f"pause_{cfg}_{setting}", p, PAUSE_TEXT[setting],
                f"停顿设置：{setting}（none=去逗号，period=逗号改句号）", "pause")
    # intonation: IndexTTS + FireRed Base, 3 types
    for cfg in ["indextts25", "fireredtts3_base"]:
        for typ in ["declarative", "interrogative", "exclamatory"]:
            p = os.path.join(C.E, "audio", "P2", cfg, "NARRATOR",
                             f"P2-intonation-{cfg}-NARRATOR-I2-{typ}.wav")
            add(spec, f"into_{cfg}_{typ}", p, INTO_TEXT[typ],
                f"语气：{typ}", "intonation")
    # emotion: IndexTTS E1 four + E2 happy/sad
    for emo in ["calm", "happy", "sad", "angry"]:
        p = os.path.join(C.E, "audio", "P2", "indextts25", "FEMALE",
                         f"P2-emotion-indextts25-FEMALE-E1-{emo}.wav")
        add(spec, f"emo_E1_{emo}", p, "这件事终于有了结果。",
            f"目标情绪：{emo}", "emotion")
    for emo in ["happy", "sad"]:
        p = os.path.join(C.E, "audio", "P2", "indextts25", "FEMALE",
                         f"P2-emotion-indextts25-FEMALE-E2-{emo}.wav")
        add(spec, f"emo_E2_{emo}", p, "他把我的书弄坏了。",
            f"目标情绪：{emo}", "emotion")
    out = os.path.join(C.E, "results", "P2b", "blind_spec.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(spec, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"spec n={len(spec)} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
