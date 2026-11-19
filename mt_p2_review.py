#!/usr/bin/env python3
"""MiniCPM-o 4.5 review pass for P2/P2b (triage, not MOS).

Builds a stratified task list, runs one clip per request with the
speech-synthesis-audio-review SKILL.md as the system prompt, saves raw and
parsed JSON, and runs a small calibration set with planted defects.

Usage:
  python mt_p2_review.py --calibrate
  python mt_p2_review.py --run [--kind intonation,emotion,pause,pron,content]
  python mt_p2_review.py --smoke /path/to.wav
"""
import argparse
import json
import os
import re
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_common as C  # noqa: E402

MODEL_DIR = "/data/shangguanrz/projectsss/speakgenerate/model/MiniCPM-o-4_5"
SKILL = "/data/shangguanrz/projectsss/speakgenerate/skills/语音评价Skill_20261109/speech-synthesis-audio-review/SKILL.md"
OUT = os.path.join(C.E, "results", "P2b", "review")
VALID = ["voxcpm2", "indextts25", "qwen3tts_base",
         "fireredtts3_base", "fireredtts3_instruct"]
PRON_OPTIONS = {
    "PR1": "‘银行’的‘行’（xíng/háng）和‘很长’的‘长’（cháng/zhǎng）",
    "PR2": "‘太重’的‘重’和‘重新’的‘重’（zhòng/chóng）",
    "PR3": "‘你得’的‘得’和‘玩得’的‘得’（děi/dé/de）",
    "PR4": "‘还有’的‘还’、‘没还’的‘还’、‘明天还’的‘还’（hái/huán）",
    "PR5": "‘我们’的‘都’、‘首都’的‘都’、‘都很’的‘都’（dōu/dū）",
    "PR6": "‘觉得’的‘觉’和‘睡觉’的‘觉’（jué/jiào）",
}


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def load_jsonl(p):
    if not os.path.isfile(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_skill():
    text = open(SKILL, encoding="utf-8").read()
    # Keep the full rubric; the model needs the status vocabulary and schema.
    return text


def load_model():
    import torch
    from transformers import AutoModel
    log(f"loading {MODEL_DIR}")
    # model.chat() sets use_tts_template=True whenever an audio ndarray is in
    # the message, so load the TTS modules even though we only want text out.
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16, init_vision=False, init_audio=True,
        init_tts=True)
    try:
        model.init_tts()
    except Exception as e:
        log(f"init_tts warning: {type(e).__name__}: {str(e)[:200]}")
    model.eval().cuda()
    log("loaded")
    return model


def parse_json(text):
    s = text.strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("no JSON object")
    return json.loads(s[i:j + 1])


def ask(model, system, prompt, audio_path, max_new_tokens=1536, temperature=0.2):
    import librosa
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": [prompt, audio]}]
    kw = dict(max_new_tokens=max_new_tokens, do_sample=False,
              use_tts_template=False, generate_audio=False,
              enable_thinking=False, max_inp_length=32768)
    try:
        res = model.chat(msgs=msgs, **kw)
    except TypeError:
        # older/newer chat signatures may reject generate_audio when TTS is off
        kw.pop("generate_audio", None)
        res = model.chat(msgs=msgs, **kw)
    if isinstance(res, (list, tuple)):
        res = res[0] if res else ""
    return res if isinstance(res, str) else str(res)


def review_one(model, system, task):
    prompt = (
        "请按 system 里的《中文合成语音评价》规范评价下面这一个音频。"
        "为了批量初筛，只返回一个**紧凑 JSON**（不要代码围栏，不要 JSON 外解释），字段如下：\n"
        '{"decision":"usable|revise|human_review|unavailable",'
        '"scores":{"content_pronunciation":1-5,"speaking_rate":1-5,'
        '"pauses_boundaries":1-5,"intonation":1-5,"emotion":1-5,'
        '"naturalness":1-5,"audio_quality":1-5,"voice_consistency":1-5},'
        '"issues":["每个问题一句，带原文短语定位"],'
        '"emotion_choice":"平静|高兴|悲伤|生气|无法判断",'
        '"type_choice":"陈述|疑问|感叹|无法判断",'
        '"pause_present":"有|无|无法判断","target_reading":"多音字实际读音，没有就空字符串"}\n'
        "不确定的分数填 null，并在 issues 里说明；不要虚构数值。\n"
        f"audio_id: {task['audio_id']}\n"
        f"source_text: {task['source_text']}\n"
        f"requirements: {task.get('requirements', '')}\n"
        "reference_audio: 无\n"
        "measured_metrics: 无可信外部测量。\n"
    )
    raw = ask(model, system, prompt, task["audio_path"])
    rec = {"task": task, "raw": raw, "ts": time.time()}
    try:
        rec["parsed"] = parse_json(raw)
        rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "parse_failed"
        rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        # one strict retry
        try:
            raw2 = ask(model, system, prompt + "\n严格只输出一个 JSON 对象。", task["audio_path"])
            rec["raw_retry"] = raw2
            rec["parsed"] = parse_json(raw2)
            rec["status"] = "ok_retry"
        except Exception as e2:
            rec["retry_error"] = f"{type(e2).__name__}: {str(e2)[:300]}"
    return rec


def build_tasks(kinds):
    tasks = []
    if "intonation" in kinds:
        for cfg in VALID:
            rows = [r for r in load_jsonl(os.path.join(C.E, "results", "P2", cfg + ".jsonl"))
                    if r.get("sub") == "intonation" and r.get("status") == "ok"
                    and r.get("voice_id") == "NARRATOR"]
            for r in rows:
                tasks.append({
                    "kind": "intonation", "audio_id": r["item_id"],
                    "audio_path": r["out_wav"], "source_text": r["text"],
                    "requirements": f"目标语气：{r.get('intonation_type')}。"
                                    "只判断声音本身；不要因为文本里有问号就假定句尾上扬。"
                                    "在 JSON 里加顶级字段 type_choice，取值 陈述/疑问/感叹/无法判断。",
                    "target": r.get("intonation_type"), "config_id": cfg})
    if "emotion" in kinds:
        rows = [r for r in load_jsonl(os.path.join(C.E, "results", "P2", "indextts25.jsonl"))
                if r.get("sub") == "emotion" and r.get("status") == "ok"
                and r.get("voice_id") in ("FEMALE", "NARRATOR")]
        for r in rows:
            tasks.append({
                "kind": "emotion", "audio_id": r["item_id"],
                "audio_path": r["out_wav"], "source_text": r["text"],
                "requirements": f"目标情绪：{r.get('emotion')}。请判断实际听感，"
                                "并从 平静/高兴/悲伤/生气/无法判断 中选一个最接近的，"
                                "在 JSON 里加顶级字段 emotion_choice。",
                "target": r.get("emotion"), "config_id": "indextts25"})
    if "pause" in kinds:
        for cfg in VALID:
            rows = [r for r in load_jsonl(os.path.join(C.E, "results", "P2b", cfg + ".jsonl"))
                    if r.get("sub") == "pause" and r.get("status") == "ok"
                    and r.get("voice_id") == "NARRATOR"
                    and r.get("item_id", "").endswith("-r1")]
            for r in rows:
                tasks.append({
                    "kind": "pause", "audio_id": r["item_id"],
                    "audio_path": r["out_wav"], "source_text": r["text"],
                    "requirements": "只评价停顿与边界：标点处是否有可感知停顿、"
                                    "是否出现词内断开/异常断音、衔接是否自然。"
                                    "在 JSON 里加顶级字段 pause_present，取值 有/无/无法判断。",
                    "target": r.get("pause_setting"), "config_id": cfg})
    if "pron" in kinds:
        # P2b hint + negative + P1 no-hint (paired texts only)
        for r in load_jsonl(os.path.join(C.E, "results", "P2b", "indextts25.jsonl")):
            if r.get("sub") != "pronunciation" or r.get("status") != "ok":
                continue
            is_neg = "pronneg" in r.get("item_id", "")
            # positives: 2 voices; negatives: all 3, to estimate chance better
            if not is_neg and r.get("voice_id") not in ("FEMALE", "NARRATOR"):
                continue
            tid = r.get("text_id") or ""
            base = tid.split("-")[0]
            opts = PRON_OPTIONS.get(base, "")
            tasks.append({
                "kind": "pron", "audio_id": r["item_id"],
                "audio_path": r["out_wav"], "source_text": r["text"],
                "requirements": f"重点核对多音字目标：{opts}。只报告实际听到的读音，"
                                "不要按文本猜测；并判断是否有逐字读字母/增字/漏字。"
                                "在 JSON 里加顶级字段 target_reading，按目标字顺序写拼音。",
                "target": tid, "config_id": "indextts25"})
        p1 = {}
        for r in load_jsonl(os.path.join(C.E, "results", "P1", "indextts25.jsonl")):
            if r.get("status") == "ok":
                p1[(r["text_id"], r["voice_id"])] = r
        pairs = [("PR2", "P1-PR-03"), ("PR3", "P1-PR-04"), ("PR4", "P1-PR-05"),
                 ("PR5", "P1-PR-06"), ("PR6", "P1-PR-08")]
        for p2, p1id in pairs:
            for v in ("FEMALE", "NARRATOR"):
                r = p1.get((p1id, v))
                if not r:
                    continue
                opts = PRON_OPTIONS.get(p2, "")
                tasks.append({
                    "kind": "pron_before",
                    "audio_id": r.get("item_id") or f"P1-{r['text_id']}-{v}",
                    "audio_path": r["out_wav"], "source_text": r["text"],
                    "requirements": f"重点核对多音字目标：{opts}。只报告实际听到的读音，"
                                    "不要按文本猜测；并判断是否有逐字读字母/增字/漏字。"
                                    "在 JSON 里加顶级字段 target_reading，按目标字顺序写拼音。",
                    "target": p2, "config_id": "indextts25"})
    if "content" in kinds:
        for cfg in ["voxcpm2", "qwen3tts_base", "fireredtts3_base", "fireredtts3_instruct"]:
            rows = [r for r in load_jsonl(os.path.join(C.E, "results", "P2", cfg + ".jsonl"))
                    if r.get("sub") == "pronunciation" and r.get("status") == "ok"
                    and r.get("voice_id") == "NARRATOR"]
            for r in rows:
                tasks.append({
                    "kind": "content", "audio_id": r["item_id"],
                    "audio_path": r["out_wav"], "source_text": r["text"],
                    "requirements": "核对内容完整性与可懂度；没有读音提示接口，不要求声调结论。",
                    "target": None, "config_id": cfg})
    return tasks


def save(rec, kind):
    d = os.path.join(OUT, kind)
    os.makedirs(d, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", rec["task"]["audio_id"])[:160]
    p = os.path.join(d, safe + ".json")
    json.dump(rec, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return p


def run_tasks(model, system, tasks, tag):
    os.makedirs(OUT, exist_ok=True)
    done = 0
    fails = 0
    t0 = time.time()
    for i, t in enumerate(tasks, 1):
        try:
            rec = review_one(model, system, t)
            rec["tag"] = tag
            save(rec, t["kind"])
            done += 1
            if rec["status"] != "ok":
                fails += 1
            log(f"[{tag}] {i}/{len(tasks)} {t['kind']} {t['audio_id']} "
                f"{rec['status']} {time.time()-t0:.0f}s")
        except Exception as e:
            fails += 1
            rec = {"task": t, "status": "run_failed", "tag": tag,
                   "error": f"{type(e).__name__}: {str(e)[:500]}",
                   "traceback": traceback.format_exc()[-1500:], "ts": time.time()}
            save(rec, t["kind"])
            log(f"[{tag}] {i}/{len(tasks)} {t['audio_id']} RUN_FAILED {e}")
    log(f"[{tag}] done={done} fails={fails} total={len(tasks)}")
    return done, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--smoke", default="")
    ap.add_argument("--kind", default="intonation,emotion,pause,pron,content")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    system = load_skill()
    model = load_model()
    if args.smoke:
        t = {"kind": "smoke", "audio_id": os.path.basename(args.smoke),
             "audio_path": args.smoke, "source_text": "请转写这段音频。",
             "requirements": "只做冒烟，不要求完整评价。"}
        rec = review_one(model, system, t)
        print(json.dumps(rec, ensure_ascii=False)[:4000])
        return 0
    if args.calibrate:
        from mt_p2_calib import build_calibration
        tasks = build_calibration()
        if args.limit:
            tasks = tasks[:args.limit]
        run_tasks(model, system, tasks, "calibration")
        return 0
    if args.run:
        kinds = set(x.strip() for x in args.kind.split(",") if x.strip())
        tasks = build_tasks(kinds)
        if args.limit:
            tasks = tasks[:args.limit]
        log(f"tasks={len(tasks)} kinds={sorted(kinds)}")
        run_tasks(model, system, tasks, "run")
        return 0
    ap.error("choose --smoke / --calibrate / --run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
