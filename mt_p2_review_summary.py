#!/usr/bin/env python3
"""Aggregate MiniCPM-o review JSON into a compact summary for mt_score_b2.py.

Calibration is a gate: if the reviewer marks planted defects as usable or
agrees <80% with the coarse labels, calibration_pass=false and mt_score_b2.py
will not use any listening number as a score.
"""
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_common as C  # noqa: E402

OUT = os.path.join(C.E, "results", "P2b", "review")
VALID = ["voxcpm2", "indextts25", "qwen3tts_base",
         "fireredtts3_base", "fireredtts3_instruct"]

TYPE_ALIASES = {
    "陈述": "declarative", "declarative": "declarative",
    "疑问": "interrogative", "interrogative": "interrogative",
    "感叹": "exclamatory", "exclamatory": "exclamatory",
}
EMO_ALIASES = {
    "平静": "calm", "calm": "calm",
    "高兴": "happy", "happy": "happy", "开心": "happy",
    "悲伤": "sad", "sad": "sad", "难过": "sad",
    "生气": "angry", "angry": "angry", "愤怒": "angry",
}
READING = {
    "PR1": ["HANG"], "PR2": ["ZHONG"], "PR3": ["DEI"],
    "PR4": ["HAI"], "PR5": ["DOU"], "PR6": ["JVE", "JUE", "JYUE"],
    "PR1-NEG": ["XING"], "PR4-NEG": ["HUAN"],
}
READING_CORRECT = {"PR1-NEG": "HANG", "PR4-NEG": "HAI"}


def load_reviews(kind):
    d = os.path.join(OUT, kind)
    out = []
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        try:
            out.append(json.load(open(os.path.join(d, fn), encoding="utf-8")))
        except Exception:
            continue
    return out


def dim_score(parsed, dim):
    try:
        return float(parsed["dimensions"][dim]["score"])
    except Exception:
        pass
    try:
        v = parsed["scores"][dim]
        return float(v) if v is not None else None
    except Exception:
        return None


def norm_reading(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace("ü", "V").replace("Ü", "V")
    s = re.sub(r"[^A-Za-z]", "", s).upper()
    return s


def _load_syllables():
    p = os.path.join(C.E, "repos", "index-tts", "checkpoints", "pinyin.vocab")
    if not os.path.isfile(p):
        return set()
    out = set()
    for line in open(p, encoding="utf-8"):
        s = line.strip().upper()
        if not s:
            continue
        s = re.sub(r"[0-9]+$", "", s)
        if s:
            out.add(s)
    # The vocab writes ü after j/q/x as V (JVE2); ASR-style pinyin often writes
    # JUE.  Add the common U-spellings so segmentation does not fall to "E".
    out.update({"JUE", "QUE", "XUE", "YUE", "JUAN", "QUAN", "XUAN",
                "JUN", "QUN", "XUN"})
    return out


SYLLABLES = _load_syllables()


def pinyin_tokens(s):
    """Greedy segment normalized pinyin using the project's pinyin.vocab.

    Substring matching is wrong here: CHANG contains HANG.  Greedy longest
    syllable segmentation turns 'XINGCHANG' into [XING, CHANG].
    """
    s = norm_reading(s)
    toks = []
    i = 0
    while i < len(s):
        matched = False
        for L in range(min(5, len(s) - i), 0, -1):
            cand = s[i:i + L]
            if cand in SYLLABLES:
                toks.append(cand)
                i += L
                matched = True
                break
        if not matched:
            i += 1
    return toks


def first_reading(rec):
    p = rec.get("parsed") or {}
    tr = p.get("target_reading")
    if tr:
        toks = pinyin_tokens(tr)
        if toks:
            return toks[0]
    # fallback: only look inside pronunciation reasoning / issues, not the prompt
    bits = []
    try:
        bits.append(p["dimensions"]["content_pronunciation"].get("reason", ""))
    except Exception:
        pass
    for it in (p.get("issues") or []):
        if isinstance(it, dict) and it.get("dimension") in ("content_pronunciation",):
            bits.append(it.get("evidence", ""))
            bits.append(it.get("observed_text", ""))
    toks = pinyin_tokens(" ".join(str(x) for x in bits))
    return toks[0] if toks else None


def reading_matches(rec, expected):
    """Compare the FIRST target's reading, not any token in the string.

    PR4 has three 还; the model may list hai/huan/huan.  The first token is the
    one that corresponds to the first planted target.
    """
    return first_reading(rec) in expected


def calibrate():
    recs = load_reviews("calib")
    # The label dict is the current pre-registered one; old saved tasks may
    # carry stale expectations (e.g. before audio-quality was marked soft).
    try:
        from mt_p2_calib import build_calibration
        labels = {t["audio_id"]: t.get("expected", {}) for t in build_calibration()}
    except Exception:
        labels = {}
    checks = []
    defect_marked_usable = False
    for rec in recs:
        p = rec.get("parsed") or {}
        aid = (rec.get("task") or {}).get("audio_id")
        exp = labels.get(aid) or (rec.get("task") or {}).get("expected") or {}
        ok = True
        dec = str(p.get("decision", "")).lower()
        if "decision_in" in exp and dec not in [x.lower() for x in exp["decision_in"]]:
            ok = False
        cps = dim_score(p, "content_pronunciation")
        if "content_pronunciation_min" in exp and (cps is None or cps < exp["content_pronunciation_min"]):
            ok = False
        if "content_pronunciation_max" in exp and (cps is not None and cps > exp["content_pronunciation_max"]):
            ok = False
        aqs = dim_score(p, "audio_quality")
        if "audio_quality_max" in exp and (aqs is None or aqs > exp["audio_quality_max"]):
            ok = False
        pbs = dim_score(p, "pauses_boundaries")
        if "pauses_boundaries_max" in exp and (pbs is None or pbs > exp["pauses_boundaries_max"]):
            ok = False
        if "emotion_choice_in" in exp:
            got = EMO_ALIASES.get(str(p.get("emotion_choice", "")).strip().lower(), str(p.get("emotion_choice", "")).strip().lower())
            if got not in [EMO_ALIASES.get(x.lower(), x.lower()) for x in exp["emotion_choice_in"]]:
                ok = False
        if "type_choice_in" in exp:
            got = TYPE_ALIASES.get(str(p.get("type_choice", "")).strip().lower(), str(p.get("type_choice", "")).strip().lower())
            if got not in [TYPE_ALIASES.get(x.lower(), x.lower()) for x in exp["type_choice_in"]]:
                ok = False
        if "pause_present_in" in exp:
            got = str(p.get("pause_present", "")).strip()
            if got not in exp["pause_present_in"]:
                ok = False
        if "issue_keywords" in exp:
            blob = json.dumps(p, ensure_ascii=False)
            if not any(k.lower() in blob.lower() for k in exp["issue_keywords"]):
                ok = False
        # Only tasks whose labels say revise/human_review are defects. Emotion /
        # intonation calibration tasks are classification checks, not defects.
        defect = ("decision_in" in exp
                  and "usable" not in [x.lower() for x in exp["decision_in"]])
        soft = bool(exp.get("soft"))
        if defect and not soft and dec == "usable":
            defect_marked_usable = True
        checks.append({"audio_id": rec["task"]["audio_id"], "ok": ok,
                       "defect": defect, "soft": soft, "decision": dec})
    hard = [c for c in checks if not c["soft"]]
    n = len(hard)
    agreement = sum(c["ok"] for c in hard) / n if n else 0.0
    passed = bool(n >= 8 and agreement >= 0.8 and not defect_marked_usable)
    note = (f"n_hard={n} agreement={agreement:.2f} defect_marked_usable={defect_marked_usable} "
            f"soft={sum(1 for c in checks if c['soft'])}; "
            f"gate={'pass' if passed else 'fail'}")
    return passed, agreement, note, checks


def intonation():
    recs = load_reviews("intonation")
    by_cfg = defaultdict(lambda: {"n": 0, "correct": 0, "dim": []})
    for rec in recs:
        t = rec.get("task") or {}
        p = rec.get("parsed") or {}
        cfg = t.get("config_id")
        target = TYPE_ALIASES.get(str(t.get("target", "")).lower(), str(t.get("target", "")).lower())
        got = TYPE_ALIASES.get(str(p.get("type_choice", "")).strip().lower(), str(p.get("type_choice", "")).strip().lower())
        if not cfg:
            continue
        by_cfg[cfg]["n"] += 1
        by_cfg[cfg]["correct"] += int(got == target)
        s = dim_score(p, "intonation")
        if s is not None:
            by_cfg[cfg]["dim"].append(s)
    out = {}
    for cfg in VALID:
        d = by_cfg.get(cfg)
        if not d or not d["n"]:
            out[cfg] = {"n": 0}
            continue
        acc = d["correct"] / d["n"]
        out[cfg] = {"n": d["n"], "acc": round(acc, 4),
                    "acc_norm": round(max(0.0, min(1.0, (acc - 1 / 3) / (2 / 3))), 4),
                    "dim_mean": round(sum(d["dim"]) / len(d["dim"]), 3) if d["dim"] else None}
    return out


def emotion():
    recs = load_reviews("emotion")
    conf = defaultdict(int)
    correct = n = 0
    dims = []
    for rec in recs:
        t = rec.get("task") or {}
        p = rec.get("parsed") or {}
        target = EMO_ALIASES.get(str(t.get("target", "")).strip().lower(), str(t.get("target", "")).strip().lower())
        got = EMO_ALIASES.get(str(p.get("emotion_choice", "")).strip().lower(), str(p.get("emotion_choice", "")).strip().lower())
        n += 1
        correct += int(got == target)
        conf[f"{target}->{got}"] += 1
        s = dim_score(p, "emotion")
        if s is not None:
            dims.append(s)
    acc = correct / n if n else 0.0
    return {"n": n, "acc": round(acc, 4),
            "acc_norm": round(max(0.0, min(1.0, (acc - 0.25) / 0.75)), 4),
            "confusion": dict(conf),
            "dim_mean": round(sum(dims) / len(dims), 3) if dims else None}


def pause():
    recs = load_reviews("pause")
    by_cfg = defaultdict(lambda: {"n": 0, "present_correct": 0, "dim": []})
    for rec in recs:
        t = rec.get("task") or {}
        p = rec.get("parsed") or {}
        cfg = t.get("config_id")
        setting = str(t.get("target", ""))
        got = str(p.get("pause_present", "")).strip()
        expected = "无" if setting == "none" else "有"
        by_cfg[cfg]["n"] += 1
        by_cfg[cfg]["present_correct"] += int(got == expected)
        s = dim_score(p, "pauses_boundaries")
        if s is not None:
            by_cfg[cfg]["dim"].append(s)
    out = {}
    for cfg in VALID:
        d = by_cfg.get(cfg)
        if not d or not d["n"]:
            out[cfg] = {"n": 0}
            continue
        dim_mean = sum(d["dim"]) / len(d["dim"]) if d["dim"] else None
        present_acc = d["present_correct"] / d["n"]
        # Calibration showed the 1-5 pauses dimension was constant (2 for both
        # long and short); the binary probe (有/无) did distinguish them.
        present_norm = max(0.0, min(1.0, (present_acc - 0.5) / 0.5))
        out[cfg] = {"n": d["n"], "dim_mean": round(dim_mean, 3) if dim_mean is not None else None,
                    "dim_norm": round(float(present_norm), 4),
                    "present_acc": round(present_acc, 4),
                    "source": "pause_present"}
    return out


def pron():
    after = load_reviews("pron")
    before = load_reviews("pron_before")
    a_ok = a_n = 0
    b_ok = b_n = 0
    neg_n = neg_says_correct = neg_detects_wrong = 0
    for rec in after:
        t = rec.get("task") or {}
        tid = str(t.get("target", ""))
        is_neg = "pronneg" in rec["task"]["audio_id"]
        if is_neg:
            neg_n += 1
            if reading_matches(rec, [READING_CORRECT.get(tid, "___")]):
                neg_says_correct += 1
            if reading_matches(rec, READING.get(tid, ["___"])):
                neg_detects_wrong += 1
        else:
            a_n += 1
            a_ok += int(reading_matches(rec, READING.get(tid, ["___"])))
    for rec in before:
        t = rec.get("task") or {}
        tid = str(t.get("target", ""))
        b_n += 1
        b_ok += int(reading_matches(rec, READING.get(tid, ["___"])))
    raw_after = round(a_ok / a_n, 4) if a_n else None
    raw_before = round(b_ok / b_n, 4) if b_n else None
    raw_chance = round(neg_says_correct / neg_n, 4) if neg_n else None
    detect = round(neg_detects_wrong / neg_n, 4) if neg_n else None
    # A reviewer that reports the text-correct reading on planted-wrong clips
    # is not audio-grounded.  Require it to catch >=2/3 of the planted wrong
    # readings before its tone accuracy enters the score.
    reliable = bool(detect is not None and detect >= 0.6667)
    return {"n_after": a_n,
            "after_acc": raw_after if reliable else None,
            "n_before": b_n,
            "before_acc": raw_before if reliable else None,
            "n_neg": neg_n,
            "chance": raw_chance if reliable else None,
            "neg_detects_wrong": detect,
            "tone_reliable": reliable,
            "raw_after_acc": raw_after,
            "raw_before_acc": raw_before,
            "raw_chance": raw_chance,
            "tone_note": ("tone scores used" if reliable else
                          "MiniCPM reports text-correct readings on planted-wrong clips; "
                          "tone scores are not audio-grounded, content CER only")}


def main():
    passed, agreement, note, checks = calibrate()
    summary = {"calibration_pass": passed, "calibration_agreement": round(agreement, 4),
               "calibration_note": note, "calibration_checks": checks,
               "intonation": {"by_config": intonation()},
               "emotion": emotion(),
               "pause": {"by_config": pause()},
               "pron": pron()}
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "summary.json")
    json.dump(summary, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "calibration_checks"},
                     ensure_ascii=False, indent=1))
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
