#!/usr/bin/env python3
"""Build the reference-voice set and P0/P1 text sets for the 5-model comparison.

Evidence-based selection:
  * 3 identity anchors come from the only speaker_assignment_status=='user_confirmed'
    records in the corpus (all HSK4-1: MALE / FEMALE / NARRATOR).
  * Reference clips are chosen per global_speaker_id from clips whose speaker was
    assigned by 'confirmed_reference_voice_comparison', i.e. compared against those
    anchors, preferring high similarity and a usable 5-12 s length.
  * No text in the corpus is human-proofread (6903 automatic_unproofread, 79
    needs_review, 64 unreliable_placeholder). Placeholders are excluded outright;
    every selected corpus text keeps its real review status and is flagged
    needs_human_proofread. Nothing here is claimed to be verified ground truth.

Outputs: refs/refs.json + wav copies, texts/p0.csv, texts/p1.csv, texts/all.jsonl
"""
import csv
import json
import os
import random
import re
import shutil
import sys

DATA = "/data/shangguanrz/projectsss/speakgenerate/data"
E = "/data/shangguanrz/projectsss/speakgenerate/experiments/model-test"
SEED = 20261111
VOICES = ["MALE", "FEMALE", "NARRATOR"]
LABEL = {"MALE": "男播音", "FEMALE": "女播音", "NARRATOR": "旁白"}


def log(m):
    print(m, flush=True)


def load_all():
    rows = []
    for pkg in sorted(os.listdir(DATA)):
        p = os.path.join(DATA, pkg, "metadata", "samples.jsonl")
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                d = json.loads(ln)
                d["_pkg"] = pkg
                d["_abs"] = os.path.join(DATA, pkg, d["audio_path"])
                rows.append(d)
    return rows


def han_len(s):
    return len(re.findall(r"[\u4e00-\u9fff]", s))


def pick_refs(rows):
    anchors = {}
    for d in rows:
        if d.get("speaker_assignment_status") == "user_confirmed":
            anchors[d["global_speaker_id"]] = d
    refs = {}
    for v in VOICES:
        a = anchors.get(v)
        if a is None:
            raise SystemExit(f"no user_confirmed anchor for {v}")
        cands = [d for d in rows
                 if d.get("global_speaker_id") == v
                 and d.get("speaker_assignment_source") == "confirmed_reference_voice_comparison"
                 and d.get("text_review_status") != "unreliable_placeholder"
                 and 5.0 <= (d.get("duration_s") or 0) <= 12.0
                 and han_len(d.get("text") or "") >= 12
                 and os.path.isfile(d["_abs"])]
        # prefer the anchor's own exam (identity consistency), then similarity, then length
        cands.sort(key=lambda d: (d["exam_key"] != a["exam_key"],
                                  -(d.get("speaker_similarity") or 0),
                                  -d["duration_s"]))
        if not cands:
            raise SystemExit(f"no usable reference candidate for {v}")
        best = cands[0]
        refs[v] = {"anchor": a, "ref": best, "n_candidates": len(cands)}
    return refs, anchors


def copy_ref(v, info):
    d = info["ref"]
    dst_dir = os.path.join(E, "refs")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"{v}_{d['sample_id']}.wav")
    shutil.copy2(d["_abs"], dst)
    a = info["anchor"]
    return {
        "voice_id": v,
        "speaker_label": LABEL[v],
        "ref_wav": dst,
        "ref_sample_id": d["sample_id"],
        "ref_exam_key": d["exam_key"],
        "ref_level": d["level"],
        "ref_text": d["text"],
        "ref_duration_s": round(d["duration_s"], 3),
        "ref_speaker_similarity": d.get("speaker_similarity"),
        "ref_assignment_status": d.get("speaker_assignment_status"),
        "ref_text_review_status": d.get("text_review_status"),
        "anchor_sample_id": a["sample_id"],
        "anchor_status": a.get("speaker_assignment_status"),
        "n_candidates": info["n_candidates"],
        "identity_caveat": ("说话人身份仅由 3 条 user_confirmed 锚点 + 嵌入相似度推定，"
                            "跨试卷是否为同一真人尚未人工确认"),
    }


# ---- P0: 10 original smoke texts (known-correct, no corpus dependency) ----
P0 = [
    ("P0-01", "陈述", "现在开始第一题。", ""),
    ("P0-02", "陈述", "请你把书翻到第十五页。", ""),
    ("P0-03", "疑问", "你明天上午有空吗？", ""),
    ("P0-04", "疑问", "车上提供什么？", ""),
    ("P0-05", "感叹", "这里的风景真是太美了！", ""),
    ("P0-06", "数字", "会议改到三月五号下午两点半。", ""),
    ("P0-07", "数字", "他的电话号码是一三八零九零一二三四五。", ""),
    ("P0-08", "多音字", "他骑车去银行，路上走了很长时间。", "行 xíng / háng；长 cháng"),
    ("P0-09", "多音字", "这件事还得重新商量，你得早点回来。", "得 děi / dé；重 chóng"),
    ("P0-10", "长句", "虽然今天下雨，但是我们还是决定按原计划去公园散步。", ""),
]

# ---- P1: 12 pronunciation-focused original texts ----
PRON = [
    ("P1-PR-01", "多音字-行", "他一边骑车一边听音乐，很快就到了银行门口。", "行 xíng / háng"),
    ("P1-PR-02", "多音字-长", "这条马路很长，两边种满了高大的树。", "长 cháng / zhǎng"),
    ("P1-PR-03", "多音字-重", "这个箱子太重了，请你重新搬一次。", "重 zhòng / chóng"),
    ("P1-PR-04", "多音字-得", "你得把作业写完，才能出去玩得开心。", "得 děi / de / dé"),
    ("P1-PR-05", "多音字-还", "他还有一本书没还给我，说明天还。", "还 hái / huán"),
    ("P1-PR-06", "多音字-都", "我们都去过首都，那里的人都很热情。", "都 dōu / dū"),
    ("P1-PR-07", "多音字-发", "他发现理发店今天不开门，只好明天再理发。", "发 fā / fà"),
    ("P1-PR-08", "多音字-觉", "我觉得睡觉以前应该少看手机。", "觉 jué / jiào"),
    ("P1-PR-09", "数字-年份电话", "二零零八年八月八号，我打了电话零幺零八八八八六六六六。", "幺 yāo"),
    ("P1-PR-10", "数字-数量分数", "三百二十五个学生里，有四分之三的人通过了考试。", "分数读法"),
    ("P1-PR-11", "数字-价格小数", "这件衣服原价一百九十九块，现在打八点五折。", "小数与折扣"),
    ("P1-PR-12", "声调-易混", "妈妈骂马，马慢，妈妈骂马慢。", "四声对比 mā má mǎ mà"),
]


def pick_hsk(rows, per_level=8):
    rng = random.Random(SEED)
    out = []
    for lv in range(1, 7):
        cands = [d for d in rows
                 if d.get("level") == lv
                 and d.get("text_review_status") == "automatic_unproofread"
                 and d.get("speaker_assignment_status") == "automatic"
                 and 10 <= han_len(d.get("text") or "") <= 40
                 and re.search(r"[。？！]$", (d.get("text") or "").strip())
                 and not re.search(r"[A-Za-z]{3,}", d.get("text") or "")
                 and os.path.isfile(d["_abs"])]
        # spread across exams and questions, avoid near-duplicate texts
        rng.shuffle(cands)
        seen_text, seen_exam, picked = set(), {}, []
        for d in cands:
            t = d["text"].strip()
            if t in seen_text:
                continue
            if seen_exam.get(d["exam_key"], 0) >= 3:
                continue
            seen_text.add(t)
            seen_exam[d["exam_key"]] = seen_exam.get(d["exam_key"], 0) + 1
            picked.append(d)
            if len(picked) >= per_level:
                break
        if len(picked) < per_level:
            log(f"  [warn] level {lv}: only {len(picked)} candidates")
        for i, d in enumerate(picked, 1):
            out.append({
                "text_id": f"P1-HSK{lv}-{i:02d}",
                "phase": "P1", "level": lv,
                "category": f"HSK{lv}真题",
                "text": d["text"].strip(),
                "pinyin_hint": "",
                "source": "corpus",
                "source_sample_id": d["sample_id"],
                "source_exam_key": d["exam_key"],
                "source_audio": d["_abs"],
                "source_duration_s": round(d["duration_s"], 3),
                "review_status": d["text_review_status"],
                "needs_human_proofread": True,
            })
    return out


def main():
    rows = load_all()
    log(f"loaded {len(rows)} samples from {len(set(r['_pkg'] for r in rows))} packages")
    refs, anchors = pick_refs(rows)
    ref_meta = []
    for v in VOICES:
        m = copy_ref(v, refs[v])
        ref_meta.append(m)
        log(f"[ref] {v:8s} {m['ref_sample_id']} {m['ref_duration_s']}s "
            f"sim={m['ref_speaker_similarity']} cands={m['n_candidates']}")
    with open(os.path.join(E, "refs", "refs.json"), "w", encoding="utf-8") as f:
        json.dump(ref_meta, f, ensure_ascii=False, indent=1)

    p0 = [{"text_id": t, "phase": "P0", "level": 0, "category": c, "text": x,
           "pinyin_hint": p, "source": "original", "source_sample_id": "",
           "source_exam_key": "", "source_audio": "", "source_duration_s": 0,
           "review_status": "agent_authored", "needs_human_proofread": True}
          for t, c, x, p in P0]
    hsk = pick_hsk(rows)
    pron = [{"text_id": t, "phase": "P1", "level": 0, "category": c, "text": x,
             "pinyin_hint": p, "source": "original", "source_sample_id": "",
             "source_exam_key": "", "source_audio": "", "source_duration_s": 0,
             "review_status": "agent_authored", "needs_human_proofread": True}
            for t, c, x, p in PRON]
    p1 = hsk + pron

    cols = ["text_id", "phase", "level", "category", "text", "pinyin_hint",
            "source", "source_sample_id", "source_exam_key", "source_audio",
            "source_duration_s", "review_status", "needs_human_proofread"]
    for name, data in (("p0", p0), ("p1", p1)):
        with open(os.path.join(E, "texts", name + ".csv"), "w",
                  encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(data)
    with open(os.path.join(E, "texts", "all.jsonl"), "w", encoding="utf-8") as f:
        for d in p0 + p1:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    log(f"P0={len(p0)} texts  P1={len(p1)} texts "
        f"(HSK={len(hsk)}, pronunciation={len(pron)})")
    log(f"P0 generations = {len(p0)} x 3 voices x 6 configs = {len(p0)*18}")
    log(f"P1 generations = {len(p1)} x 3 voices x 6 configs = {len(p1)*18}")
    log("NOTE: 语料文本全部未经人工校对，已逐条保留 review_status 并标记 "
        "needs_human_proofread；CER 参考文本在人工校对前只能作为初步依据。")
    return 0


if __name__ == "__main__":
    sys.exit(main())