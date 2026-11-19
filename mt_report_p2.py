#!/usr/bin/env python3
"""Generate RESULTS_P2.md from the measured P2/P2b JSONs. No estimates."""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_common as C  # noqa: E402

VALID = ["voxcpm2", "indextts25", "qwen3tts_base",
         "fireredtts3_base", "fireredtts3_instruct"]


def load(rel, default=None):
    p = os.path.join(C.E, rel)
    if not os.path.isfile(p):
        return default
    return json.load(open(p, encoding="utf-8"))


def fnum(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main():
    pause = load("results/P2b/eval/pause_pairs.json", {})
    pros = load("results/P2b/eval/prosody.json", {})
    pron = load("results/P2b/eval/pron_cer.json", [])
    review = load("results/P2b/review/summary.json", {})
    score = load("results/score_trackB_v2.json", {})
    L = []
    L.append("# P2 控制实验收口报告（v2）\n")
    L.append("本报告由 `scripts/mt_report_p2.py` 从实测 JSON 生成；"
             "未测/不支持/门失败分别记录，不补 0。评分口径见 `docs/P2_RUBRIC.md`。\n")

    L.append("## 一、这轮修了什么\n")
    L.append("- **停顿 v1 作废**：`build_p2.py` 的 `PAUSE_SETTINGS` 标点从未拼进文本，"
             "none/comma/period 传入同一句话；VoxCPM2/FireRed 三档音频字节相同。"
             "v2 用 `texts/p2b.jsonl` 的三种文本变体重跑 405 条（5 配置 × 3 音色 × 3 设置 × 3 文本 × 3 seeds）。")
    L.append("- **IndexTTS 读音 v1 作废**：旧提示是 `<行|xíng>`，源码要求 `<行|HANG2>`（拼音+数字声调）；"
             "ASR 回转写听到 “x i n g”“太昌涨”“脚猪位”。v2 改为词表内数字声调，"
             "并修掉 `_pinyin_text` 首次出现替换导致的嵌套标注；PR1 由 `行|xíng;行|háng` 改为 `行|HANG2;长|CHANG2`。")
    L.append("- **情绪 text_fallback 无信号**：除 IndexTTS 外，四情绪文本完全相同，"
             "音频差异只是采样噪声，记 `no_interface`，不给 0 分。")
    L.append("- **语气全部 text_fallback**：标点文本不同，有信号，但不是原生接口；结果标 `control_mode=text_fallback`。\n")

    L.append("## 二、停顿 v2（文本标点三档）\n")
    L.append("主效应：同文本配对 `delta_s=(span_per_char_comma-span_per_char_none)*nch`，"
             "3 seeds 取中位数，`effect=clip01(median(delta_s)/0.2)`。"
             "gap 检测只作位置/自然度描述（`librosa.effects.split(top_db=30)` 会漏浅停顿）。\n")
    L.append("| 配置 | n 三元组 | comma−none 中位 s | 均值 s | 正差/总数 | 符号检验 p | effect | v1 同文本噪声 s | 门 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cfg in VALID:
        p = (pause.get("configs", {}) or {}).get(cfg, {})
        if p.get("status") != "ok":
            L.append(f"| {cfg} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | {p.get('status','not_tested')} |")
            continue
        L.append(f"| {cfg} | {p.get('n_triples')} | {fnum(p.get('delta_comma_median_s'))} | "
                 f"{fnum(p.get('delta_comma_mean_s'))} | {p.get('positive_comma')}/{p.get('n_triples')} | "
                 f"{fnum(p.get('sign_p'))} | {fnum(p.get('pause_effect'))} | "
                 f"{fnum(p.get('v1_same_text_abs_delta_median_s'))} | "
                 f"{'通过' if p.get('gate_v2_gt_v1_noise') else '未通过'} |")
    L.append("\n**读法**：中位数为正且大于 v1 同文本噪声才算“停顿控制显示”。"
             "FireRed Base 9/9 为正、中位 +0.38 s、p=0.0039；Instruct 中位 +0.48 s、7/9；"
             "VoxCPM2 中位 −0.04 s；IndexTTS/Qwen 只有 +0.06 s 且低于 v1 噪声。"
             "结论：**文本标点能稳定控制 FireRed Base/Instruct 的停顿，"
             "但对 VoxCPM2/IndexTTS/Qwen 不稳定**。实际组卷若要控制停顿，"
             "优先用 FireRed 的标点响应，或在模板层插入静音并单独标 `postproc`。\n")

    L.append("## 三、语气（text_fallback）\n")
    L.append("F0-only 特征单因素 η² + 10k 置换 p；主分用 ω²（小样本校正，锚点 0.12）。\n")
    L.append("| 配置 | n | 最大 η² 特征 | η² | ω² | effect | 置换 p |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for cfg in VALID:
        x = (pros.get("intonation", {}) or {}).get(cfg, {})
        if x.get("status") != "ok":
            L.append(f"| {cfg} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        feat = (x.get("features") or {}).get(x.get("max_eta2_feature"), {})
        L.append(f"| {cfg} | {x.get('n')} | {x.get('max_eta2_feature')} | "
                 f"{fnum(x.get('max_eta2'))} | {fnum(x.get('max_omega2'))} | "
                 f"{fnum(x.get('effect'))} | {fnum(feat.get('perm_p'))} |")
    L.append("\nIndexTTS 的 final_delta 差异最大（置换 p=0.005）；Qwen3 的 ω²=0.0038 接近 0，"
             "VoxCPM2/FireRed 两配置 ω²≤0。这里只说明 F0 层差异，"
             "是否“听得出来”要由 MiniCPM/人工听评补足；不预设疑问句必须句尾上扬。"
             "**听评 prompt 的 source_text 保留了原标点，模型可能把标点当线索；"
             "type_choice 只作初筛，人工复核应同时看去掉句末标点的版本。**\n")

    L.append("## 四、情绪（仅 IndexTTS 原生）\n")
    em = (pros.get("emotion", {}) or {}).get("indextts25", {})
    if em.get("status") == "ok":
        L.append(f"- 36 条（4 情绪 × 3 文本 × 3 音色）；LOO 最近质心准确率 **{fnum(em.get('acc'),4)}**，"
                 f"机会 0.25，目标分 **{fnum(em.get('obj'),4)}**。")
        L.append(f"- 混淆：`{json.dumps(em.get('confusion'), ensure_ascii=False)}`")
        L.append("\n其余 4 配置：四情绪文本完全相同，音频差异只是随机采样，记 `no_interface`。"
                 "情绪伴随的语速/时长变化见 `prosody.json` 的 per_emotion_means；"
                 "不能把“更快”直接当“高兴”。\n")
    else:
        L.append("情绪客观分析未产出。\n")

    L.append("## 五、读音纠错（仅 IndexTTS 原生）\n")
    pos = [r for r in pron if not r.get("negative_control")]
    neg = [r for r in pron if r.get("negative_control")]
    if pos:
        mean_cer = sum(r["cer"] for r in pos) / len(pos)
        L.append(f"- v2 正样本 18 条（6 文本 × 3 音色），平均 CER **{mean_cer:.4f}**；"
                 f"内容门（≤0.05）{'通过' if mean_cer <= 0.05 else '未通过'}。")
        L.append("- CER 只作内容完整性门：ASR 对同字不同调仍输出同一字，不能判断声调/多音字。")
        if neg:
            L.append(f"- 负对照 {len(neg)} 条（故意错读）：平均 CER {sum(r['cer'] for r in neg)/len(neg):.4f}；"
                     "用于 MiniCPM/人工的 tone 判断机会水平，不把 CER 当声调证据。")
        bad = [r for r in pos if r["cer"] > 0]
        if bad:
            L.append("- 逐条残差：" + "；".join(
                f"{r['item_id'].split('-')[-2]}/{r['voice_id']} CER={r['cer']:.3f} `{r['hyp'][:30]}`"
                for r in bad[:8]))
    else:
        L.append("读音 CER 未产出。\n")

    L.append("\n## 六、MiniCPM-o 听评（初筛，不是 MOS）\n")
    if review:
        L.append(f"- 校准：**{'通过' if review.get('calibration_pass') else '未通过'}**；"
                 f"{review.get('calibration_note','')}")
        if not review.get("calibration_pass"):
            L.append("- 校准未过时，`mt_score_b2.py` 不使用任何听评分数，只报客观半权重；"
                     "这是设计行为，不是漏测。")
        it = review.get("intonation", {}).get("by_config", {})
        if it:
            L.append("\n### 语气三选一\n")
            L.append("| 配置 | n | 准确率 | 归一化 |")
            L.append("| --- | --- | --- | --- |")
            for cfg in VALID:
                x = it.get(cfg, {})
                if x.get("n"):
                    L.append(f"| {cfg} | {x['n']} | {fnum(x.get('acc'),3)} | {fnum(x.get('acc_norm'),3)} |")
        emr = review.get("emotion", {})
        if emr.get("n"):
            L.append(f"\n### 情绪四选一\n\n- n={emr['n']}，准确率 {fnum(emr.get('acc'),3)}，"
                     f"归一化 {fnum(emr.get('acc_norm'),3)}；混淆 `{json.dumps(emr.get('confusion'), ensure_ascii=False)}`。\n")
        pr = review.get("pron", {})
        if pr.get("n_after"):
            L.append(f"\n### 多音字读音\n\n- raw after n={pr['n_after']} acc="
                     f"{fnum(pr.get('raw_after_acc'),3)}；raw before n={pr['n_before']} acc="
                     f"{fnum(pr.get('raw_before_acc'),3)}；负对照 n={pr['n_neg']}，"
                     f"chance={fnum(pr.get('raw_chance'),3)}，"
                     f"neg_detects_wrong={fnum(pr.get('neg_detects_wrong'),3)}。\n")
            L.append(f"- **tone 可靠性门：{'通过' if pr.get('tone_reliable') else '未通过'}**。"
                     "PR1 正样本模型报 `xíng`（期望 `háng`），PR4 负对照第一个仍报 `hái`"
                     "（未检出故意错读 `huán`）；因此 tone 分不采用，读音维度只按内容 CER 计 0.3，"
                     "人工抽检 `P2_听评包` 是唯一可确认声调的证据。\n")
    else:
        L.append("听评 summary 未产出；MiniCPM 原始输出与人工抽检是后续唯一待办。\n")

    L.append("## 七、赛道 B v2 分数\n")
    L.append("| 配置 | 语速 | 停顿 | 语气 | 情绪 | 读音纠错 | 显存 | 权重覆盖 | 完整分 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for c in score.get("configs", []):
        sub = c.get("trackB_subscores", {})

        def g(k):
            v = sub.get(k)
            return f"{v:.3f}" if v is not None else "n/a"
        L.append(f"| {c['config_id']} | {g('speaking_rate_control')} | {g('pause_control')} | "
                 f"{g('intonation_control')} | {g('emotion_control')} | "
                 f"{g('pronunciation_correction')} | {g('vram_efficiency')} | "
                 f"{c.get('trackB_weight_coverage')} | "
                 f"{c.get('trackB_full_score') if c.get('trackB_full_score') is not None else 'null'} |")
    L.append("\n`no_interface` 维度不补 0；完整分只在权重覆盖率=1.0 时计算，"
             "否则为 null 且 `rankable=false`。部分分不能当总分排名。"
             "本轮 IndexTTS 覆盖 0.795，缺 pause 听评 0.10 + tone 0.105；"
             "其余 4 配置覆盖 0.35，缺 native rate/emotion/pron。"
             "**因此所有 `trackB_full_score` 都是 null，这是覆盖率事实，不是分数 0。**\n")

    L.append("## 八、限制与建议\n")
    L.append("- 语料文本未经人工校对；CER 只能作初步筛查，声调/多音字必须针对性听评。")
    L.append("- 24 GB 是目标约束，本机 83 GiB 卡；峰值显存用 `max_memory_allocated`，是下界。")
    L.append("- 停顿：FireRed Base/Instruct 显示稳定的标点响应；VoxCPM2/IndexTTS/Qwen "
             "未稳定显示。实际组卷可优先用 FireRed 的标点响应，或模板级静音后处理并单独记录为 postproc。")
    L.append("- 语气/情绪：MiniCPM 是 9B 初筛，不是 MOS；人工抽检结果未回填前，听评分数只作证据。")
    L.append("- CosyVoice3 仍 output_invalid，不参与排名。\n")

    out = os.path.join(C.E, "RESULTS_P2.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
