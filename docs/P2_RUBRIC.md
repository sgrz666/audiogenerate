# P2 赛道 B 评分口径（预注册 v2）

版本：P2-RUBRIC-2026-11-13
适用：`experiments/model-test` 的 5 个有效配置（voxcpm2 / indextts25 / qwen3tts_base / fireredtts3_base / fireredtts3_instruct）。CosyVoice3 仍为 `output_invalid`，不参与本轮。

本文件在评分前冻结。除“发现代码/数据错误必须更正”外，不因结果好坏调整阈值；任何更正都要在 `RESULTS_P2.md` 留痕。

## 0. 状态词

- `ok`：实际生成并测量。
- `not_tested`：本轮未测，不补分。
- `no_interface`：该配置没有对应原生接口，且没有可比的文本驱动信号。
- `gate_failed`：pilot/校准门未过，不产生分数。
- `unsupported`：同 `no_interface`，用于控制矩阵。
- 任何缺失维度不得写 0 分；`trackB_full_score` 仅在 `trackB_weight_coverage == 1.0` 时计算，否则为 `null`，并标 `rankable: false`。

## 1. 权重

| 维度 | 权重 |
|---|---:|
| speaking_rate_control | 0.25 |
| pause_control | 0.20 |
| intonation_control | 0.15 |
| emotion_control | 0.15 |
| pronunciation_correction | 0.15 |
| vram_efficiency | 0.10 |

## 2. 各维度口径

### 2.1 speaking_rate_control（沿用 v1）

- 仅统计 `control_mode == native` 的 rate 项。
- `rate_score = clip01(1 - (mean_rel_err_pct - 5) / (25 - 5))`；≤5% → 1.0，≥25% → 0。
- 同时报 `rate_n_native`、`rate_within_5pct`、`mean_rel_err_pct`。
- VoxCPM2/Qwen3/FireRed 无原生语速接口：`no_interface`，不把 text_fallback 计入 native 分。

### 2.2 pause_control（v2，text_fallback）

- 三档文本：none 去掉 `，`；comma 保留 `，`；period 把 `，` 替换为 `。`。
- 每项生成 3 个 seed（`r1/r2/r3`），同一 `(config,voice,text)` 的三档必须使用同一 seed 配对。
- 暂停检测器（冻结）：16 kHz、`librosa.effects.split(y, top_db=30, frame_length=400, hop_length=160)`；相邻非静音区间之间的间隔 ≥100 ms 记为一个 gap；记录 gap 的 start/end/duration。
- 参数验证：在 IndexTTS `P2-pause-indextts25-MALE-P1-comma.wav` 最长 voiced 段（2.44–3.62 s）中点注入 200 ms 零静音，检测到 0.18 s（误差 0.02 s）。相邻低能量段可能并入同一 gap，这是已记录限制；因此只用配对差值，不把绝对 gap 数当唯一证据。
- 每 `(config,voice,text)` 对 3 seeds 取 `span_per_char = speech_span_s / norm_chars` 的中位数；`speech_span_s` 用与语速同一把尺（`mt_common.speech_span`，20 ms 帧、保留句内停顿、去首尾静音）。
- **主效应（同文本配对）**：`delta_s = (span_per_char_comma - span_per_char_none) * norm_chars`；`effect = clip01(median(delta_s)/0.2)`。0.2 s 是项目预注册的“每句可感知附加停顿”锚点。用中位数而不是“正差值的均值”：VoxCPM2 中位数 −0.04 s、p=1.0，若取正差均值会虚高到 0.44。用 span 而不是 gap 数，因为 `effects.split(top_db=30)` 会漏浅停顿（P2b pause pilot：period r2/r3 报 0 gap，但 cpm 中位数 225.4→214.3→208.6 已显示变慢）。
- **辅助描述**：gap 检测器输出 `n_gaps/total_gap/max_gap` 与位置，用于“是否在标点处停顿、是否词内断开”的描述与听评定位，不作为主分。可报 `effects.split(top_db=20/25/30)` 的敏感性；主分只用 span。
- `listen`：MiniCPM-o/人工对“分句处是否有可感知停顿”的二选一（none→无；comma/period→有），按机会 0.5 归一化 `clip01((acc-0.5)/0.5)`。校准实测 `pauses_boundaries` 1–5 分对 long/short 都给 2 分，不能区分，因此只作次要自然度描述，不进主分。
- `pause_control = 0.5*effect + 0.5*listen`。若没有听评，`pause_control` 只报 `effect` 并把听评分量标 `not_tested`，权重覆盖按 0.10 计。
- **MiniCPM pause 听评偏置（2026-11-13 实测）**：真实样本中 VoxCPM2/FireRed Base 的 none 也被报成“有”，9/9 全部为“有”；校准 long/short 能分，但总体偏置明显。本轮 pause 听评分不用于计分，只作证据；人工抽检 `P2_听评包` 后再回填。
- `delta_period = (span_per_char_period - span_per_char_comma) * norm_chars` 只报探索性效应量与符号检验，不计分；不预设 period 一定长于 comma。
- 门：v2 的 `delta_s` 中位效应必须大于 v1 同文本重复按同一公式算出的噪声底；否则写“停顿控制未显示”，不调阈值救。

### 2.3 intonation_control（text_fallback）

- F0 提取：`librosa.pyin`，16 kHz，70–400 Hz，`frame_length=1024`，`hop_length=160`；5 帧中值滤波；相邻帧跳变 >6 ST 视为八度错误并用局部中值替换；记录 pyin 失败帧比例。
- 特征（F0-only，避免文本长度混杂）：`f0_mean`、`st_std`、`st_range`（P95-P5）、`final_delta`（末 200 ms 平均 ST − 前 300 ms 平均 ST）。duration/dB 只作描述，不进主分。
- 每配置 27 条（3 类型 × 3 文本 × 3 音色）；单因素 η²（3 组）+ 10k 次置换 p；同时报 ω²（小样本校正）。
- `effect = clip01(max_ω²_F0 / 0.12)`；0.12 是 Cohen f=0.4（大效应）对应的 ω² 锚点（项目预注册）。只用 η² 会把 Qwen 的 ω²=0.0038 抬成 0.58，故主分用 ω²，η² 只作描述。
- `listen`：MiniCPM-o/人工三选一（陈述/疑问/感叹）识别准确率，按机会 1/3 归一化：`clip01((acc-1/3)/(2/3))`。
- `intonation_control = 0.5*effect + 0.5*listen`；没有听评时只报 `effect`，覆盖按 0.075 计。
- 不预设“疑问句必须句尾上扬”；标点只作线索。若客观 F0 层没有差异，如实写“未显示”，不因文本标点不同而给分。
- 听评时 source_text 保留原标点，模型可能把标点当线索；`type_choice` 只作初筛，人工复核应同时看去掉句末标点的版本（陈述/感叹同文，疑问保留“吗”），避免把文本分类当语调判断。

### 2.4 emotion_control（仅 IndexTTS 原生）

- 仅 IndexTTS 有 `emo_vector`；其余 4 配置四情绪文本完全相同，音频差异只是采样噪声：`no_interface`。
- 特征：`f0_mean`、`st_std`、`st_range`、`db_mean`、`db_std`、duration；标准化后做 leave-one-out 最近质心 4 类。`final_delta` 是语气特征，先验排除；把它加进来会把 acc 从 0.583 拉到 0.472，作为敏感性备注，不作为主口径。
- `obj = clip01((LOO_acc - 0.25) / 0.75)`；报混淆矩阵与每类均值。
- `listen`：MiniCPM-o/人工四选一（平静/高兴/悲伤/生气）准确率，按机会 0.25 归一化。
- `emotion_control = 0.5*obj + 0.5*listen`；没有听评时只报 `obj`，覆盖按 0.075 计。
- 已知混杂：情绪伴随语速/时长变化；报告时同时给 duration/cpm，不把“更快”直接当“高兴”。

### 2.5 pronunciation_correction（仅 IndexTTS 原生）

- 提示格式：`<字|PINYIN+数字声调>`，如 `银<行|HANG2>`；读音必须在 `repos/index-tts/checkpoints/pinyin.vocab` 内（注意 `JVE2` 不是 `JUE2`）。
- `content_ok = 1 if mean CER <= 0.05 else 0`；CER 用 SenseVoiceSmall，`use_itn=False`，`t2s=True`，去标点；CER 只作内容完整性门，不能判断声调/多音字。
- `before_acc`：P1 同文本无提示音频的目标读音二选一准确率（5 对 × 3 音色；PR1 单列）。
- `after_acc`：P2 v2 hint 音频的目标读音二选一准确率。
- `chance`：由 6 条故意错读负对照实测的准确率（不是 0.5 拍脑袋）。
- `after_acc_norm = clip01((after_acc - chance) / (1 - chance))`。
- `improvement_norm = clip01((after_acc - before_acc) / max(1 - before_acc, 0.01))`。
- `pronunciation_correction = 0.3*content_ok + 0.4*after_acc_norm + 0.3*improvement_norm`。
- 门：pilot 2 条（PR2/PR4）必须无逐字读字母痕迹且 CER≤0.05；不过则 `gate_failed`，不跑 18 条。
- **tone 可靠性门（2026-11-13 实测）**：MiniCPM 必须在故意错读负对照中检出 ≥2/3 的错读（`neg_detects_wrong >= 0.6667`），其 `after_acc/before_acc/chance` 才进入分数；否则只报内容 CER（`0.3*content_ok`，覆盖按 0.045 计），并写明“tone 未证实，人工抽检待做”。
- 本轮实测：PR1 正样本模型报 `xíng`（期望 `háng`），负对照 PR4 报 `hai/huan/huan`（第一个仍未检出 HUAN）；`neg_detects_wrong=3/6=0.5`，故 tone 分不采用。内容 CER 0.0074 仍说明“不再逐字读字母”的格式修复有效。
- 其他配置没有读音提示接口：`no_interface`；只报内容 CER，不计入纠错分。

### 2.6 vram_efficiency

- `vram_efficiency = 1.0 if fits_24gb else 0.0`；`fits_24gb = max_peak_vram_gib <= 24.0`。
- 保留原始 `mean_rtf`、`max_peak_vram_gib`；注明 `torch.cuda.max_memory_allocated` 是下界（NVML 坏，无法测进程总占用）。
- 本机是 83 GiB 卡，24 GB 是目标约束；该分只表示部署门，不表示“越小越好”。

## 3. 听评协议

- 使用 `skills/语音评价Skill_20261109/speech-synthesis-audio-review/SKILL.md` 的 8 维 JSON；MiniCPM-o 为初筛，不是 MOS。
- 校准集 10–12 条：正常、逐字读字母、错声调、截断/削波、语速极端；要求 ≥80% 与人工标签一致，且对植入缺陷不得给 `usable`；否则 MiniCPM 只作 triage，分数以人工为准。
- 人工抽检 20–30 条，分层覆盖五个维度；盲听包不透明 ID，answer_key 分离。
- 任何听评分数都必须带 `n`、覆盖来源（MiniCPM/人工）与一致率；不能把 MiniCPM 的 5 分制当人工 MOS。

## 4. 输出

- `results/score_trackB_v2.json`：每配置 subscores、`trackB_weight_coverage`、`trackB_full_score`、`rankable`、`not_tested_reason`。
- `results/P2b/<cid>.jsonl`：v2 生成记录（含 seed/control_mode/out_wav）。
- `results/P2b/eval/pause_pairs.json`：配对效果表。
- `results/P2b/eval/prosody.json`：语气/情绪客观结果。
- `results/P2b/review/`：MiniCPM 原始/解析结果与人工抽检。
- `RESULTS_P2.md`、更新后的 `RESULTS.md` / `STATUS.md`。
