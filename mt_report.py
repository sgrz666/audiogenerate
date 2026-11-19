#!/usr/bin/env python3
"""Generate the horizontal-comparison report from measured results only.

Reads metrics_/score_ JSON for P0 and P1, adds per-level and per-category CER
breakdowns for P1, and writes RESULTS.md. Anything not measured is listed as
not_tested rather than estimated.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_common as C  # noqa: E402


def load(name):
    p = os.path.join(C.E, "results", name)
    if not os.path.isfile(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def per_item(phase, cid):
    p = os.path.join(C.E, "results", phase, cid + ".metrics.jsonl")
    if not os.path.isfile(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    texts = {t["text_id"]: t for t in
             [json.loads(l) for l in
              open(os.path.join(C.E, "texts", "all.jsonl"), encoding="utf-8")]}
    m0, m1 = load("metrics_P0.json"), load("metrics_P1.json")
    scb = load("score_trackB_v2.json")
    s0, s1 = load("score_P0.json"), load("score_P1.json")
    sc1 = {c["config_id"]: c for c in (s1["configs"] if s1 else [])}
    sc0 = {c["config_id"]: c for c in (s0["configs"] if s0 else [])}

    L = []
    L.append("# 五模型零样本横向评测结果\n")
    L.append("本文件由 `scripts/mt_report.py` 从实测结果生成。"
             "未测到的维度一律标 not_tested，不做估算。\n")

    L.append("## 一、P0 冒烟（10 文本 × 3 音色 = 30 条/配置）\n")
    L.append("| 配置 | 成功 | 音色相似度 | CER | 字/分钟 | RTF | 峰值显存 GiB |  fits 24GB |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for m in (m0 or []):
        L.append(f"| {m['config_id']} | {m.get('n_ok')}/{m.get('n_rows')} | "
                 f"{m.get('mean_spk_sim')} | {m.get('mean_cer')} | "
                 f"{m.get('mean_chars_per_min')} | {m.get('mean_rtf')} | "
                 f"{m.get('max_peak_vram_gib')} | {m.get('fits_24gb')} |")
    L.append("\nP0 文本短而干净，CER 区分度低（天花板效应），只用于排查运行故障。\n")

    L.append("## 二、P1 横评（48 HSK 分级 + 12 发音专项 = 60 文本 × 3 音色 = 180 条/配置）\n")
    if not m1:
        L.append("**P1 尚未产出结果。**\n")
    else:
        L.append("| 配置 | 成功 | 音色相似度 | CER | 字/分钟 | RTF | 峰值显存 GiB | fits 24GB |")
        L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for m in m1:
            L.append(f"| {m['config_id']} | {m.get('n_ok')}/{m.get('n_rows')} | "
                     f"{m.get('mean_spk_sim')} | {m.get('mean_cer')} | "
                     f"{m.get('mean_chars_per_min')} | {m.get('mean_rtf')} | "
                     f"{m.get('max_peak_vram_gib')} | {m.get('fits_24gb')} |")

        L.append("\n### P1 按 HSK 等级的 CER\n")
        levels = sorted({t["level"] for t in texts.values() if t["level"] > 0})
        header = "| 配置 | " + " | ".join(f"HSK{l}" for l in levels) + " | 发音专项 |"
        L.append(header)
        L.append("| --- |" + " --- |" * (len(levels) + 1))
        for m in m1:
            cid = m["config_id"]
            items = per_item("P1", cid)
            by = defaultdict(list)
            for it in items:
                t = texts.get(it["text_id"])
                if not t or it.get("cer") is None:
                    continue
                key = f"L{t['level']}" if t["level"] > 0 else "PRON"
                by[key].append(it["cer"])
            cells = []
            for l in levels:
                v = by.get(f"L{l}")
                cells.append(f"{np.mean(v):.4f}" if v else "n/a")
            pv = by.get("PRON")
            cells.append(f"{np.mean(pv):.4f}" if pv else "n/a")
            L.append(f"| {cid} | " + " | ".join(cells) + " |")

        L.append("\n### P1 赛道 A 客观分项（归一化已在 mt_score.py 预先固定）\n")
        L.append("| 配置 | 发音(30%) | 音色(25%) | 音质(10%) | 客观部分分 | 权重覆盖率 |")
        L.append("| --- | --- | --- | --- | --- | --- |")
        for m in m1:
            c = sc1.get(m["config_id"], {})
            sub = c.get("trackA_subscores", {})
            def f(k):
                v = sub.get(k)
                return f"{v:.3f}" if v is not None else "n/a"
            L.append(f"| {m['config_id']} | {f('content_pronunciation')} | "
                     f"{f('voice_consistency')} | {f('audio_quality')} | "
                     f"{c.get('trackA_objective_partial')} | "
                     f"{c.get('trackA_weight_coverage')} |")
        L.append("\n**客观部分分只覆盖赛道 A 的 65% 权重**（发音 30 + 音色 25 + 音质 10）。"
                 "自然度 20%、语气 10%、稳定性 5% 未测，"
                 "因此 `trackA_full_score` 为 null，不能把部分分当成总分排名。\n")

    L.append("## 三、未测 / 受阻维度（如实记录）\n")
    L.append("- 自然度、语气自然性：需人工匿名盲听，未做。")
    L.append("- 稳定性：需 P3 多次重复生成，未做。")
    if scb:
        L.append("- 赛道 B 已出 v2 部分分（见第七节与 `RESULTS_P2.md`）；"
                 "停顿/语气/情绪的听评与读音声调仍需人工抽检，"
                 "`no_interface` 维度不补 0，完整分只在覆盖率=1.0 时计算。")
    else:
        L.append("- 赛道 B 语速/停顿/语气/情绪/读音纠错控制：需 P2 控制扫描，未做。")
        L.append("- 赛道 B 仅显存与效率有实测值（见上表 RTF / 峰值显存 / fits_24gb）。")
    L.append("- 语料文本全部未经人工校对，CER 只能作初步筛查；"
             "三个参考音色跨试卷、是否同一真人未确认。\n")

    L.append("## 四、选型建议（基于已测部分，非最终结论）\n")
    if m1:
        ranked = sorted([m for m in m1 if m.get("mean_spk_sim") is not None],
                        key=lambda m: -m["mean_spk_sim"])
        L.append("按音色相似度（赛道 A 权重最高的可测客观项之一）排序：\n")
        for i, m in enumerate(ranked, 1):
            L.append(f"{i}. {m['config_id']} — sim={m['mean_spk_sim']}, "
                     f"CER={m['mean_cer']}, RTF={m['mean_rtf']}, "
                     f"峰值显存={m['max_peak_vram_gib']} GiB")
        if scb:
            L.append("\n任务书已定 VoxCPM2 为主线基座；第二个微调名额应在其余四个中竞争。"
                     "**P2 v2 已补齐可测部分，但完整选型仍需人工盲听确认自然度/语气；"
                     "停顿/语气维度存在明显配置差异，不能把部分分当总分。**")
        else:
            L.append("\n任务书已定 VoxCPM2 为主线基座；第二个微调名额应在其余四个中竞争。"
                     "**最终选择必须等人工盲听与 P2 控制实验补齐后再定**，"
                     "当前只有客观部分分，不足以覆盖赛道 A 的全部权重，更未覆盖赛道 B。")
        L.append("\n另需注意：FireRedTTS3-Base 峰值显存显著高于其余配置，"
                 "虽然在本机 83 GiB 卡上跑通，但是否塞进 24 GB 需看 fits_24gb 列；"
                 "Instruct 的编辑收益不得计入 Base 的零样本能力。")
    else:
        L.append("P1 未完成，暂不给排序。")

    if scb:
        L.append("\n## 七、P2 控制实验 v2（Track B，5 个有效配置）\n")
        L.append("v1 的停顿三档传入同一句文本、IndexTTS 读音提示用了带声调符号的拼音，"
                 "两者已作废；v2 用 `texts/p2b.jsonl` 重跑。"
                 "`no_interface` 维度不补 0；`trackB_full_score` 只在覆盖率=1.0 时计算。\n")
        L.append("| 配置 | 语速 | 停顿 | 语气 | 情绪 | 读音纠错 | 显存 | 权重覆盖 | 完整分 |")
        L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for c in scb.get("configs", []):
            sub = c.get("trackB_subscores", {})

            def f(k):
                v = sub.get(k)
                return f"{v:.3f}" if v is not None else "n/a"
            L.append(f"| {c['config_id']} | {f('speaking_rate_control')} | "
                     f"{f('pause_control')} | {f('intonation_control')} | "
                     f"{f('emotion_control')} | {f('pronunciation_correction')} | "
                     f"{f('vram_efficiency')} | {c.get('trackB_weight_coverage')} | "
                     f"{c.get('trackB_full_score') if c.get('trackB_full_score') is not None else 'null'} |")
        L.append("\n分维度证据、失效项与限制见 `RESULTS_P2.md`；"
                 "停顿主效应用同文本 speech-span 中位数增量："
                 "FireRed Base/Instruct 显示稳定效应，VoxCPM2/IndexTTS/Qwen 未显示；"
                 "未显示时实际组卷应使用模板级静音后处理并单独标注 postproc。\n")

    out = os.path.join(C.E, "RESULTS.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    C.log(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())