#!/usr/bin/env python3
"""Build P2plus: HSK 1-6 female dialogue cloning & 6-subexperiment control set."""
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_common as C
import mt_metrics as M
from build_p2 import EMOTIONS, PRON_TEXTS, validate_hints
from funasr import AutoModel

DATA = "/data/shangguanrz/projectsss/speakgenerate/data"
OUT_ROOT = os.path.join(C.E, "audio", "P2plus")

# Curated, dialogue-verified pairs: 1 female dialogue (2-clause for pause testing) +
# 1 longer same-level/same-exam female ref per HSK 1-6.
PAIRS = [
    (1, "HSK1-3_Q12_U05", "HSK1-3_Q18_U04"),
    (2, "HSK2-2_Q21_U06", "HSK2-2_Q25_U02"),
    (3, "HSK3-1_Q36_U09", "HSK3-1_Q40_U04"),
    (4, "HSK4-2_Q32_U05", "HSK4-2_Q18_U02"),
    (5, "HSK5-5_Q11_U02", "HSK5-5_Q25_U02"),
    (6, "HSK6-5_Q21_U04", "HSK6-5_Q26_U03"),
]

SUB_DIRS = [
    "01_默认语速基线",
    "02_语速控制",
    "03_停顿控制",
    "04_语气控制",
    "05_情绪控制",
    "06_多音字读音纠错",
]

RATE_TIERS = [160, 200, 240, 280, 320]
SEEDS = [42, 43, 44]

# Pause variants strictly follow the user's pattern:
#   none:   春天来了花都开了小草也绿了
#   comma:  春天来了，花都开了，小草也绿了
#   period: 春天来了。花都开了。小草也绿了。
PAUSE_VARIANTS = {
    1: {
        "none": "我下午不来了我们明天上午见",
        "comma": "我下午不来了，我们明天上午见",
        "period": "我下午不来了。我们明天上午见。",
    },
    2: {
        "none": "这么长时间了还没有公共汽车来",
        "comma": "这么长时间了，还没有公共汽车来",
        "period": "这么长时间了。还没有公共汽车来。",
    },
    3: {
        "none": "那我们到机场还要多长时间不会迟到吧",
        "comma": "那我们到机场还要多长时间，不会迟到吧",
        "period": "那我们到机场还要多长时间。不会迟到吧。",
    },
    4: {
        "none": "年底各大商场都在打折你们可以去看看",
        "comma": "年底各大商场都在打折，你们可以去看看",
        "period": "年底各大商场都在打折。你们可以去看看。",
    },
    5: {
        "none": "今年春节我们买几号的票回家呢",
        "comma": "今年春节，我们买几号的票回家呢",
        "period": "今年春节。我们买几号的票回家呢。",
    },
    6: {
        "none": "根据多年的拍摄经验你觉得生态摄影有什么需要注意的吗",
        "comma": "根据多年的拍摄经验，你觉得生态摄影有什么需要注意的吗",
        "period": "根据多年的拍摄经验。你觉得生态摄影有什么需要注意的吗。",
    },
}

INTO_VARIANTS = {
    1: {
        "declarative": ("陈述", "我下午不来了，我们明天上午见。"),
        "interrogative": ("疑问", "我下午不来了，我们明天上午见，好吗？"),
        "exclamatory": ("感叹", "我下午真的不来了，我们明天上午见！"),
    },
    2: {
        "declarative": ("陈述", "这么长时间了，还没有公共汽车来。"),
        "interrogative": ("疑问", "这么长时间了，还没有公共汽车来？"),
        "exclamatory": ("感叹", "这么长时间了，还没有公共汽车来！"),
    },
    3: {
        "declarative": ("陈述", "我们到机场还要一会儿，不会迟到的。"),
        "interrogative": ("疑问", "那我们到机场还要多长时间？不会迟到吧？"),
        "exclamatory": ("感叹", "那我们到机场还要这么长时间，可千万别迟到啊！"),
    },
    4: {
        "declarative": ("陈述", "年底各大商场都在打折，你们可以去看看。"),
        "interrogative": ("疑问", "年底各大商场都在打折，你们不去看看吗？"),
        "exclamatory": ("感叹", "年底各大商场都在大打折，你们快去看看吧！"),
    },
    5: {
        "declarative": ("陈述", "今年春节，我们决定提前买票回家。"),
        "interrogative": ("疑问", "今年春节，我们买几号的票回家呢？"),
        "exclamatory": ("感叹", "今年春节，我们终于买到回家的票了！"),
    },
    6: {
        "declarative": ("陈述", "根据多年的拍摄经验，生态摄影确实有很多需要注意的地方。"),
        "interrogative": ("疑问", "根据多年的拍摄经验，你觉得生态摄影有什么需要注意的吗？"),
        "exclamatory": ("感叹", "根据多年的拍摄经验，生态摄影需要注意的地方实在太多了！"),
    },
}

# Level-specific polyphone hints on the HSK 1-6 dialogues themselves (positive + deliberate negative control)
HSK_POLY = {
    1: ("了|LE5", "了|LIAO3", "我下午不来了，我们明天上午见。"),
    2: ("长|CHANG2;还|HAI2", "长|ZHANG3;还|HUAN2", "这么长时间了，还没有公共汽车来。"),
    3: ("还|HAI2;长|CHANG2;会|HUI4", "还|HUAN2;长|ZHANG3;会|KUAI4", "那我们到机场还要多长时间？不会迟到吧？"),
    4: ("大|DA4;折|ZHE2", "大|DAI4;折|SHE2", "年底各大商场都在打折，你们可以去看看。"),
    5: ("几|JI3", "几|JI1", "今年春节，我们买几号的票回家呢？"),
    6: ("觉|JVE2;得|DE5;要|YAO4", "觉|JIAO4;得|DEI3;要|YAO1", "根据多年的拍摄经验，你觉得生态摄影有什么需要注意的吗？"),
}


def load_corpus():
    by_id = {}
    questions = {}
    for pkg in sorted(os.listdir(DATA)):
        pdir = os.path.join(DATA, pkg)
        sp = os.path.join(pdir, "metadata", "samples.jsonl")
        qp = os.path.join(pdir, "metadata", "questions.jsonl")
        if not os.path.isfile(sp):
            continue
        if os.path.isfile(qp):
            with open(qp, encoding="utf-8") as f:
                for ln in f:
                    q = json.loads(ln)
                    questions[(q["exam_key"], q["question_no"])] = q
        with open(sp, encoding="utf-8") as f:
            for ln in f:
                r = json.loads(ln)
                r["_abs"] = os.path.join(pdir, r["audio_path"])
                by_id[r["sample_id"]] = r
    return by_id, questions


def main():
    # Validate all pinyin hints first
    for lvl, (pos_h, neg_h, _) in HSK_POLY.items():
        validate_hints(pos_h)
        validate_hints(neg_h)
    for _, _, hints in PRON_TEXTS:
        validate_hints(hints)

    by_id, questions = load_corpus()
    enc = M.SpeakerEncoder("cuda:0")
    asr_dir = next(d for d in [os.path.join(C.E, "asr", "SenseVoiceSmall"), "/data/shangguanrz/projectsss/speakgenerate/model/asr/SenseVoiceSmall"] if os.path.isdir(d))
    asr = AutoModel(model=asr_dir,
                    disable_update=True, device="cuda:0")

    gt_dir = os.path.join(OUT_ROOT, "data_gt")
    ref_dir = os.path.join(OUT_ROOT, "refs")
    for d in (gt_dir, ref_dir, os.path.join(C.E, "results", "P2plus")):
        os.makedirs(d, exist_ok=True)
    for mdir in ("vox", "index"):
        p = os.path.join(OUT_ROOT, mdir)
        os.makedirs(p, exist_ok=True)
        # Clean any legacy flat wavs in vox/ or index/
        for fn in os.listdir(p):
            fp = os.path.join(p, fn)
            if os.path.isfile(fp) and fn.endswith(".wav"):
                os.remove(fp)
        for sd in SUB_DIRS:
            os.makedirs(os.path.join(p, sd), exist_ok=True)
    # Clean legacy flat files in data_gt and refs so only current 6 remain
    for d in (gt_dir, ref_dir):
        for fn in os.listdir(d):
            if fn.endswith(".wav"):
                os.remove(os.path.join(d, fn))

    def inspect(wav_path, text):
        w, sr = C.read_wav(wav_path)
        dur = len(w) / float(sr)
        s0, s1 = C.speech_span(w, sr)
        span = max(s1 - s0, 1e-6)
        nch = C.norm_chars(text)
        cpm = nch / span * 60.0
        res = asr.generate(input=wav_path, cache={}, language="zh",
                           use_itn=False, batch_size_s=60)
        hyp = re.sub(r"<\|[^|]*\|>", "", res[0]["text"] if res else "")
        cer = M.edit_distance(M.norm_text(text, t2s=True),
                              M.norm_text(hyp, t2s=True)) / max(len(M.norm_text(text, t2s=True)), 1)
        emb = enc.emb(wav_path)
        return {
            "sample_rate": sr,
            "duration_s": round(dur, 4),
            "speech_span_s": round(span, 4),
            "norm_chars": nch,
            "actual_cpm": round(cpm, 2),
            "asr_hyp": hyp,
            "cer": round(cer, 4),
            "_emb": emb,
        }

    manifest = []
    for lvl, tgt_id, ref_id in PAIRS:
        tgt = by_id[tgt_id]
        ref = by_id[ref_id]
        assert tgt["level"] == lvl and ref["level"] == lvl
        assert tgt["global_speaker_id"] == "FEMALE" and ref["global_speaker_id"] == "FEMALE"
        assert tgt["question_no"] != ref["question_no"]

        gt_dst = os.path.join(gt_dir, f"HSK{lvl}_{tgt_id}_gt.wav")
        ref_dst = os.path.join(ref_dir, f"HSK{lvl}_{ref_id}_ref.wav")
        shutil.copy2(tgt["_abs"], gt_dst)
        shutil.copy2(ref["_abs"], ref_dst)

        ti = inspect(gt_dst, tgt["text"])
        ri = inspect(ref_dst, ref["text"])
        pair_sim = round(M.cos(ti.pop("_emb"), ri.pop("_emb")), 4)
        q = questions.get((tgt["exam_key"], tgt["question_no"]), {})

        entry = {
            "item_id": f"P2plus-HSK{lvl}",
            "level": lvl,
            "exam_key": tgt["exam_key"],
            "question_no": tgt["question_no"],
            "question_template": q.get("template"),
            "target_sample_id": tgt_id,
            "target_role": tgt.get("role"),
            "target_text": tgt["text"].strip(),
            "target_source_wav": tgt["_abs"],
            "target_gt_wav": gt_dst,
            "target_speaker_status": tgt.get("speaker_assignment_status"),
            "target_metrics": ti,
            "ref_sample_id": ref_id,
            "ref_role": ref.get("role"),
            "ref_text": ref["text"].strip(),
            "ref_source_wav": ref["_abs"],
            "ref_wav": ref_dst,
            "ref_speaker_status": ref.get("speaker_assignment_status"),
            "ref_metrics": ri,
            "ref_target_spk_sim": pair_sim,
        }
        manifest.append(entry)
        print(f"[HSK{lvl}] TGT={tgt_id} ({ti['duration_s']}s, {ti['actual_cpm']} cpm, cer={ti['cer']}) | "
              f"REF={ref_id} ({ri['duration_s']}s, cer={ri['cer']}) | sim={pair_sim}")

    # Build full P2plus task list covering all 6 subfolders
    tasks = []
    for e in manifest:
        lvl = e["level"]
        sid = e["target_sample_id"]
        gt_cpm = float(e["target_metrics"]["actual_cpm"])
        base_txt = e["target_text"]

        # 01_默认语速基线
        tasks.append({
            "item_id": f"P2plus-01-HSK{lvl}",
            "sub": "baseline",
            "folder": "01_默认语速基线",
            "level": lvl,
            "filename": f"HSK{lvl}_{sid}_baseline.wav",
            "text": base_txt,
            "seed": 42,
        })

        # 02_语速控制 (真值语速 + 160/200/240/280/320)
        rate_targets = [("gt", gt_cpm)] + [(str(r), float(r)) for r in RATE_TIERS]
        for tag, tcpm in rate_targets:
            tasks.append({
                "item_id": f"P2plus-02-HSK{lvl}-tgt_{tag}",
                "sub": "rate",
                "folder": "02_语速控制",
                "level": lvl,
                "rate_tag": tag,
                "target_cpm": tcpm,
                "filename": f"HSK{lvl}_{sid}_tgt-{tag}.wav",
                "text": base_txt,
                "seed": 42,
            })

        # 03_停顿控制 (none / comma / period x 3 seeds)
        for setting in ("none", "comma", "period"):
            ptxt = PAUSE_VARIANTS[lvl][setting]
            for rep, seed in enumerate(SEEDS, 1):
                tasks.append({
                    "item_id": f"P2plus-03-HSK{lvl}-{setting}-r{rep}",
                    "sub": "pause",
                    "folder": "03_停顿控制",
                    "level": lvl,
                    "pause_setting": setting,
                    "replicate": rep,
                    "filename": f"HSK{lvl}_{sid}_{setting}_r{rep}.wav",
                    "text": ptxt,
                    "seed": seed,
                })

        # 04_语气控制 (declarative / interrogative / exclamatory)
        for itype in ("declarative", "interrogative", "exclamatory"):
            ilabel, itxt = INTO_VARIANTS[lvl][itype]
            tasks.append({
                "item_id": f"P2plus-04-HSK{lvl}-{itype}",
                "sub": "intonation",
                "folder": "04_语气控制",
                "level": lvl,
                "intonation_type": itype,
                "intonation_label": ilabel,
                "filename": f"HSK{lvl}_{sid}_{itype}.wav",
                "text": itxt,
                "seed": 42,
            })

        # 05_情绪控制 (calm / happy / sad / angry)
        for emo, elabel, idx, _ in EMOTIONS:
            vec = [0.0] * 8
            vec[idx] = 1.0
            tasks.append({
                "item_id": f"P2plus-05-HSK{lvl}-{emo}",
                "sub": "emotion",
                "folder": "05_情绪控制",
                "level": lvl,
                "emotion": emo,
                "emotion_label": elabel,
                "emo_vector": vec,
                "filename": f"HSK{lvl}_{sid}_{emo}.wav",
                "text": base_txt,
                "seed": 42,
            })

        # 06_多音字读音纠错 - Part A: HSK 1-6 dialogue polyphones (pos + neg for index; plain for vox)
        pos_h, neg_h, poly_txt = HSK_POLY[lvl]
        tasks.append({
            "item_id": f"P2plus-06-HSK{lvl}-dlg-pos",
            "sub": "pronunciation",
            "folder": "06_多音字读音纠错",
            "level": lvl,
            "poly_group": "hsk_dialogue",
            "poly_id": f"HSK{lvl}_DLG",
            "pinyin_hints": pos_h,
            "negative_control": False,
            "filename": f"HSK{lvl}_{sid}_poly_pos.wav",
            "text": poly_txt,
            "seed": 42,
        })
        tasks.append({
            "item_id": f"P2plus-06-HSK{lvl}-dlg-neg",
            "sub": "pronunciation",
            "folder": "06_多音字读音纠错",
            "level": lvl,
            "poly_group": "hsk_dialogue",
            "poly_id": f"HSK{lvl}_DLG_NEG",
            "pinyin_hints": neg_h,
            "negative_control": True,
            "index_only": True,
            "filename": f"HSK{lvl}_{sid}_poly_neg.wav",
            "text": poly_txt,
            "seed": 42,
        })

        # 06_多音字读音纠错 - Part B: PR1-PR6 under HSK 1-6 cloning voices
        pr_neg = {
            "PR1": "行|XING2;长|CHANG2",
            "PR4": "还|HUAN2;还|HAI2;还|HAI2",
        }
        for tid, ptxt, hints in PRON_TEXTS:
            tasks.append({
                "item_id": f"P2plus-06-HSK{lvl}-{tid}-pos",
                "sub": "pronunciation",
                "folder": "06_多音字读音纠错",
                "level": lvl,
                "poly_group": "standard_pr",
                "poly_id": tid,
                "pinyin_hints": hints,
                "negative_control": False,
                "filename": f"HSK{lvl}_{tid}_pos.wav",
                "text": ptxt,
                "seed": 42,
            })
            if tid in pr_neg:
                tasks.append({
                    "item_id": f"P2plus-06-HSK{lvl}-{tid}-neg",
                    "sub": "pronunciation",
                    "folder": "06_多音字读音纠错",
                    "level": lvl,
                    "poly_group": "standard_pr",
                    "poly_id": f"{tid}_NEG",
                    "pinyin_hints": pr_neg[tid],
                    "negative_control": True,
                    "index_only": True,
                    "filename": f"HSK{lvl}_{tid}_neg.wav",
                    "text": ptxt,
                    "seed": 42,
                })

    mpath = os.path.join(OUT_ROOT, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    tpath = os.path.join(C.E, "texts", "p2plus.jsonl")
    with open(tpath, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    n_vox = sum(1 for t in tasks if not t.get("index_only"))
    n_idx = len(tasks)
    print(f"Built P2plus: {len(manifest)} HSK levels, vox tasks={n_vox}, index tasks={n_idx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())