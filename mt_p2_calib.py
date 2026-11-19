#!/usr/bin/env python3
"""Calibration set for MiniCPM-o P2 review (10-12 clips, planted defects).

Expected labels are deliberately coarse; the point is to catch a reviewer that
says "usable" to everything, not to estimate MOS.
"""
import os

E = "/data/shangguanrz/projectsss/speakgenerate/experiments/model-test"


def build_calibration():
    tasks = []

    def add(aid, rel, text, req, expected):
        p = rel if os.path.isabs(rel) else os.path.join(E, rel)
        if not os.path.isfile(p):
            print(f"[calib] skip missing {p}")
            return
        tasks.append({"kind": "calib", "audio_id": aid, "audio_path": p,
                      "source_text": text, "requirements": req,
                      "expected": expected})

    add("calib-normal", "audio/P1/indextts25/NARRATOR/P1-HSK1-01.wav",
        "那家饭店是我朋友开的。",
        "正常样本；按规范完整评价。",
        {"decision_in": ["usable"], "content_pronunciation_min": 4})
    add("calib-letter-read", "audio/P2/indextts25/NARRATOR/P2-pronunciation-indextts25-NARRATOR-PR1.wav",
        "他骑车去银行，路上走了很长时间。",
        "重点听是否把拼音提示逐字读成字母或插入怪音；在 JSON 里加 target_reading。",
        {"decision_in": ["revise", "human_review"], "content_pronunciation_max": 3,
         "issue_keywords": ["字母", "拼音", "xing", "怪", "增字"]})
    add("calib-wrong-tone", "audio/P2b/indextts25/NARRATOR/P2b-pronneg-indextts25-NARRATOR-PR4.wav",
        "他还有一本书没还给我，说明天还。",
        "重点核对多音字读音；在 JSON 里加 target_reading。",
        {"decision_in": ["revise", "human_review"],
         "issue_keywords": ["读音", "声调", "多音", "梅", "海", "hái", "huán", "hai", "huan", "还"]})
    add("calib-clipped", "results/P2b/calib/clipped.wav",
        "那家饭店是我朋友开的。",
        "重点听音质、削波、失真。",
        {"decision_in": ["revise", "human_review"], "audio_quality_max": 3,
         "soft": True})
    add("calib-dropout", "results/P2b/calib/dropout.wav",
        "那家饭店是我朋友开的。",
        "重点听是否有异常断音、缺口、衔接不自然。",
        {"decision_in": ["revise", "human_review"], "pauses_boundaries_max": 3,
         "soft": True})
    add("calib-pause-long", "audio/P2b/indextts25/NARRATOR/P2b-pause-indextts25-NARRATOR-P1-period-r1.wav",
        "春天来了。花都开了。小草也绿了。",
        "重点听分句/句间是否有可感知停顿；在 JSON 里加 pause_present。",
        {"pause_present_in": ["有"], "soft": True})
    add("calib-pause-short", "audio/P2b/indextts25/NARRATOR/P2b-pause-indextts25-NARRATOR-P1-none-r1.wav",
        "春天来了花都开了小草也绿了。",
        "重点听分句处是否几乎没有停顿；在 JSON 里加 pause_present。",
        {"pause_present_in": ["无"], "soft": True})
    add("calib-happy", "audio/P2/indextts25/FEMALE/P2-emotion-indextts25-FEMALE-E1-happy.wav",
        "这件事终于有了结果。",
        "判断实际听感最像哪种情绪；在 JSON 里加 emotion_choice。",
        {"emotion_choice_in": ["高兴", "happy"]})
    add("calib-sad", "audio/P2/indextts25/FEMALE/P2-emotion-indextts25-FEMALE-E1-sad.wav",
        "这件事终于有了结果。",
        "判断实际听感最像哪种情绪；在 JSON 里加 emotion_choice。",
        {"emotion_choice_in": ["悲伤", "sad"]})
    add("calib-declarative", "audio/P2/indextts25/FEMALE/P2-intonation-indextts25-FEMALE-I2-declarative.wav",
        "这道题其实一点都不难。",
        "判断实际听感最像哪种句型；在 JSON 里加 type_choice。",
        {"type_choice_in": ["陈述", "declarative"]})
    add("calib-interrogative", "audio/P2/indextts25/FEMALE/P2-intonation-indextts25-FEMALE-I2-interrogative.wav",
        "这道题其实一点都不难吗？",
        "判断实际听感最像哪种句型；在 JSON 里加 type_choice。",
        {"type_choice_in": ["疑问", "interrogative"]})
    add("calib-exclamatory", "audio/P2/indextts25/FEMALE/P2-intonation-indextts25-FEMALE-I2-exclamatory.wav",
        "这道题其实一点都不难！",
        "判断实际听感最像哪种句型；在 JSON 里加 type_choice。",
        {"type_choice_in": ["感叹", "exclamatory"]})
    print(f"[calib] tasks={len(tasks)}")
    return tasks


if __name__ == "__main__":
    for t in build_calibration():
        print(t["audio_id"], t["audio_path"])
