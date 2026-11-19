# P2 控制实验收口报告（v2）

本报告由 `scripts/mt_report_p2.py` 从实测 JSON 生成；未测/不支持/门失败分别记录，不补 0。评分口径见 `docs/P2_RUBRIC.md`。

## 一、这轮修了什么

- **停顿 v1 作废**：`build_p2.py` 的 `PAUSE_SETTINGS` 标点从未拼进文本，none/comma/period 传入同一句话；VoxCPM2/FireRed 三档音频字节相同。v2 用 `texts/p2b.jsonl` 的三种文本变体重跑 405 条（5 配置 × 3 音色 × 3 设置 × 3 文本 × 3 seeds）。
- **IndexTTS 读音 v1 作废**：旧提示是 `<行|xíng>`，源码要求 `<行|HANG2>`（拼音+数字声调）；ASR 回转写听到 “x i n g”“太昌涨”“脚猪位”。v2 改为词表内数字声调，并修掉 `_pinyin_text` 首次出现替换导致的嵌套标注；PR1 由 `行|xíng;行|háng` 改为 `行|HANG2;长|CHANG2`。
- **情绪 text_fallback 无信号**：除 IndexTTS 外，四情绪文本完全相同，音频差异只是采样噪声，记 `no_interface`，不给 0 分。
- **语气全部 text_fallback**：标点文本不同，有信号，但不是原生接口；结果标 `control_mode=text_fallback`。

## 二、停顿 v2（文本标点三档）

主效应：同文本配对 `delta_s=(span_per_char_comma-span_per_char_none)*nch`，3 seeds 取中位数，`effect=clip01(median(delta_s)/0.2)`。gap 检测只作位置/自然度描述（`librosa.effects.split(top_db=30)` 会漏浅停顿）。

| 配置 | n 三元组 | comma−none 中位 s | 均值 s | 正差/总数 | 符号检验 p | effect | v1 同文本噪声 s | 门 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| voxcpm2 | 9 | -0.040 | 0.096 | 4/9 | 1.000 | 0.000 | 0.000 | 未通过 |
| indextts25 | 9 | 0.060 | 0.104 | 6/9 | 0.508 | 0.300 | 0.120 | 未通过 |
| qwen3tts_base | 9 | 0.060 | 0.149 | 7/9 | 0.180 | 0.300 | 0.200 | 未通过 |
| fireredtts3_base | 9 | 0.380 | 0.478 | 9/9 | 0.004 | 1.000 | 0.000 | 通过 |
| fireredtts3_instruct | 9 | 0.480 | 0.382 | 7/9 | 0.180 | 1.000 | 0.000 | 通过 |

**读法**：中位数为正且大于 v1 同文本噪声才算“停顿控制显示”。FireRed Base 9/9 为正、中位 +0.38 s、p=0.0039；Instruct 中位 +0.48 s、7/9；VoxCPM2 中位 −0.04 s；IndexTTS/Qwen 只有 +0.06 s 且低于 v1 噪声。结论：**文本标点能稳定控制 FireRed Base/Instruct 的停顿，但对 VoxCPM2/IndexTTS/Qwen 不稳定**。实际组卷若要控制停顿，优先用 FireRed 的标点响应，或在模板层插入静音并单独标 `postproc`。

## 三、语气（text_fallback）

F0-only 特征单因素 η² + 10k 置换 p；主分用 ω²（小样本校正，锚点 0.12）。

| 配置 | n | 最大 η² 特征 | η² | ω² | effect | 置换 p |
| --- | --- | --- | --- | --- | --- | --- |
| voxcpm2 | 27 | st_std | 0.059 | -0.019 | 0.000 | 0.531 |
| indextts25 | 27 | final_delta | 0.348 | 0.286 | 1.000 | 0.005 |
| qwen3tts_base | 27 | st_range | 0.081 | 0.004 | 0.032 | 0.366 |
| fireredtts3_base | 27 | st_std | 0.014 | -0.065 | 0.000 | 0.844 |
| fireredtts3_instruct | 27 | f0_mean | 0.024 | -0.055 | 0.000 | 0.744 |

IndexTTS 的 final_delta 差异最大（置换 p=0.005）；Qwen3 的 ω²=0.0038 接近 0，VoxCPM2/FireRed 两配置 ω²≤0。这里只说明 F0 层差异，是否“听得出来”要由 MiniCPM/人工听评补足；不预设疑问句必须句尾上扬。**听评 prompt 的 source_text 保留了原标点，模型可能把标点当线索；type_choice 只作初筛，人工复核应同时看去掉句末标点的版本。**

## 四、情绪（仅 IndexTTS 原生）

- 36 条（4 情绪 × 3 文本 × 3 音色）；LOO 最近质心准确率 **0.5833**，机会 0.25，目标分 **0.4444**。
- 混淆：`{"calm->sad": 1, "calm->angry": 1, "calm->calm": 5, "happy->angry": 2, "happy->happy": 7, "sad->sad": 4, "angry->sad": 2, "angry->calm": 1, "angry->angry": 5, "sad->happy": 5, "angry->happy": 1, "calm->happy": 2}`

其余 4 配置：四情绪文本完全相同，音频差异只是随机采样，记 `no_interface`。情绪伴随的语速/时长变化见 `prosody.json` 的 per_emotion_means；不能把“更快”直接当“高兴”。

## 五、读音纠错（仅 IndexTTS 原生）

- v2 正样本 18 条（6 文本 × 3 音色），平均 CER **0.0074**；内容门（≤0.05）通过。
- CER 只作内容完整性门：ASR 对同字不同调仍输出同一字，不能判断声调/多音字。
- 负对照 6 条（故意错读）：平均 CER 0.0357；用于 MiniCPM/人工的 tone 判断机会水平，不把 CER 当声调证据。
- 逐条残差：MALE/MALE CER=0.067 `你得把作业写完才能出去玩的开心`；FEMALE/FEMALE CER=0.067 `你得把作业写完才能出去玩的开心`

## 六、MiniCPM-o 听评（初筛，不是 MOS）

- 校准：**通过**；n_hard=8 agreement=1.00 defect_marked_usable=False soft=4; gate=pass

### 语气三选一

| 配置 | n | 准确率 | 归一化 |
| --- | --- | --- | --- |
| voxcpm2 | 9 | 0.778 | 0.667 |
| indextts25 | 9 | 0.667 | 0.500 |
| qwen3tts_base | 9 | 1.000 | 1.000 |
| fireredtts3_base | 9 | 0.667 | 0.500 |
| fireredtts3_instruct | 9 | 0.889 | 0.833 |

### 情绪四选一

- n=24，准确率 0.833，归一化 0.778；混淆 `{"angry->无法判断": 4, "calm->calm": 6, "happy->happy": 6, "sad->sad": 6, "angry->angry": 2}`。


### 多音字读音

- raw after n=12 acc=0.833；raw before n=10 acc=1.000；负对照 n=6，chance=0.500，neg_detects_wrong=0.500。

- **tone 可靠性门：未通过**。PR1 正样本模型报 `xíng`（期望 `háng`），PR4 负对照第一个仍报 `hái`（未检出故意错读 `huán`）；因此 tone 分不采用，读音维度只按内容 CER 计 0.3，人工抽检 `P2_听评包` 是唯一可确认声调的证据。

## 七、赛道 B v2 分数

| 配置 | 语速 | 停顿 | 语气 | 情绪 | 读音纠错 | 显存 | 权重覆盖 | 完整分 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| voxcpm2 | n/a | 0.000 | 0.333 | n/a | n/a | 1.000 | 0.35 | null |
| indextts25 | 1.000 | 0.300 | 0.750 | 0.611 | 0.300 | 1.000 | 0.795 | null |
| qwen3tts_base | n/a | 0.300 | 0.516 | n/a | n/a | 1.000 | 0.35 | null |
| fireredtts3_base | n/a | 1.000 | 0.250 | n/a | n/a | 1.000 | 0.35 | null |
| fireredtts3_instruct | n/a | 1.000 | 0.417 | n/a | n/a | 1.000 | 0.35 | null |

`no_interface` 维度不补 0；完整分只在权重覆盖率=1.0 时计算，否则为 null 且 `rankable=false`。部分分不能当总分排名。本轮 IndexTTS 覆盖 0.795，缺 pause 听评 0.10 + tone 0.105；其余 4 配置覆盖 0.35，缺 native rate/emotion/pron。**因此所有 `trackB_full_score` 都是 null，这是覆盖率事实，不是分数 0。**

## 八、限制与建议

- 语料文本未经人工校对；CER 只能作初步筛查，声调/多音字必须针对性听评。
- 24 GB 是目标约束，本机 83 GiB 卡；峰值显存用 `max_memory_allocated`，是下界。
- 停顿：FireRed Base/Instruct 显示稳定的标点响应；VoxCPM2/IndexTTS/Qwen 未稳定显示。实际组卷可优先用 FireRed 的标点响应，或模板级静音后处理并单独记录为 postproc。
- 语气/情绪：MiniCPM 是 9B 初筛，不是 MOS；人工抽检结果未回填前，听评分数只作证据。
- CosyVoice3 仍 output_invalid，不参与排名。

