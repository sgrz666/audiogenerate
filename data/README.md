# 中文听力考试切片数据集（HSK 1–6）

面向汉语听力考试语音合成项目的切片语料说明。数据来自原始考试录音，按题目/说话轮次切分，逐条配有文本、时间戳、说话人候选标签和复核状态。

- 版本日期：2026-11-10
- 规模：7 个分包、36 套材料、1410 题、7046 条切片、9.04 小时、2.87 GB
- 音频：44.1 kHz、16-bit、单声道 WAV；每条配同名 UTF-8 TXT
- 分发：7 个独立 ZIP（每个小于 500 MB），包内自带离线试听页与 `checksums.sha256`
- **本仓库只包含数据集介绍**：7 个分包的 README 和 `summary.json` / `volumes.json`，**不含音频、转写全文、标注表、试听页**。原始数据包请向项目维护者获取。

## 数据来源与范围

材料为 HSK 一级至六级的 2018 年真题 1–5 与考试大纲样卷，共 36 套：

| 分包 | 套题 | 套数 | 题数 | 切片 | 时长 | 音频大小 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 01 一级 | HSK1-1…5、HSK1-outline | 6 | 120 | 526 | 1582.72 s（26.4 min） | 139,618,656 B |
| 02 二级 | HSK2-1…5、HSK2-outline | 6 | 210 | 1366 | 4490.30 s（74.8 min） | 396,104,196 B |
| 03 三级 | HSK3-1…5、HSK3-outline | 6 | 240 | 1906 | 7438.68 s（124.0 min） | 656,175,148 B |
| 04 四级 | HSK4-1…5、HSK4-outline | 6 | 270 | 1156 | 5137.97 s（85.6 min） | 453,226,126 B |
| 05 五级 | HSK5-1…5、HSK5-outline | 6 | 270 | 1180 | 5623.28 s（93.7 min） | 496,025,408 B |
| 06 六级（一） | HSK6-1…3 | 3 | 150 | 467 | 4142.33 s（69.0 min） | 365,374,222 B |
| 07 六级（二） | HSK6-4…5、HSK6-outline | 3 | 150 | 445 | 4143.41 s（69.1 min） | 365,468,240 B |
| **合计** |  | **36** | **1410** | **7046** | **32558.68 s（9.04 h）** | **2,871,991,996 B（2.87 GB）** |

36 套材料的源归档来自同一批原始材料，各分包 `summary.json` 记录的源归档 sha256 均为 `4bba0058dc968425615f71f34b4006c2139f1f00c7e38fa06b390be9c0cf8f7f`。逐套的来源文件名、来源 sha256、题数、切片数和时长在 `packages/<分包名>/metadata/volumes.json` 中。

## 原始分包结构

每个 ZIP 完整解压后可直接双击根目录的 `index.html` 离线试听，无需合并分卷：

```
<分包>/
├── README.md                     # 该包说明（本仓库已收录）
├── index.html                    # 离线试听页：按套题/题号/说话人筛选、搜索、播放
├── data/<套题编号>/
│   ├── <exam>_Q<题号>_U<轮次>.wav     # 44.1 kHz / 16-bit / 单声道
│   ├── <exam>_Q<题号>_U<轮次>.txt     # 同名文本，UTF-8 无 BOM
│   └── 标注.tsv                       # 该套完整标注，UTF-8 with BOM
├── metadata/
│   ├── samples.tsv                    # 全包标注（UTF-8 BOM）
│   ├── samples.jsonl                  # 结构化版本，保留数组与空值
│   ├── questions.jsonl                # 题目、轮次、共用材料关联
│   ├── volumes.json                   # 逐套来源与统计
│   └── summary.json                   # 全包统计与状态（本仓库已收录）
├── evaluation/                        # 语音评价 Skill、输入示例与输出 schema
└── checksums.sha256                   # 包内所有内容文件的 SHA-256
```

本仓库的 `packages/<分包名>/` 只保留 `README.md` 和 `metadata/summary.json`、`metadata/volumes.json`；其余文件均在数据 ZIP 中。

## 标注字段

`标注.tsv` / `samples.jsonl` 每行一条切片，主要字段如下（完整 40 列以数据包内表头为准）：

| 组 | 字段 | 说明 |
| --- | --- | --- |
| 标识 | `sample_id`、`exam_key`、`level`、`source_kind` | `sample_id` 形如 `HSK1-4_Q15_U04`；`source_kind` 为“2018真题”或大纲样卷 |
| 定位 | `question_no`、`utterance_no`、`part`、`repeat_index` | 题号、题内轮次、部分、重复播放序号；缺失时为 `null`/空 |
| 文件 | `audio_file`、`text_file`、`audio_path`、`text_path` | 前两者相对套题目录，后两者相对分包根目录 |
| 文本 | `text`、`text_review_status`、`text_sources` | 自动转写文本及其复核状态、来源 |
| 音频 | `duration_s`、`sample_rate_hz`、`channels`、`sample_width_bits`、`audio_bytes`、`audio_sha256` | 实测音频属性与哈希 |
| 时间 | `source_start_s`、`source_end_s`、`speech_start_in_clip_s`、`speech_end_in_clip_s` | 原始整卷时间；当前片段内的语音起止 |
| 说话人 | `speaker_scope`、`speaker_key`、`role`、`global_speaker_id`、`speaker_label` | 身份按套题限定；`role` 为题号/提示/对话/提问等朗读功能 |
| 说话人状态 | `speaker_assignment_status`、`speaker_assignment_source`、`speaker_similarity`、`speaker_margin` | 自动/待核/已确认；相似度与判决余量 |
| 复核 | `flags`、`review_note`、`status`、`punctuation_source`、`punctuation_review_note` | 保留原标注中的标记、备注和标点来源 |

### 状态与质量分布

说话人指派（7046 条）：

| 状态 | 条数 |
| --- | ---: |
| `automatic`（自动推定） | 3345 |
| `needs_review`（待核） | 3698 |
| `user_confirmed`（已有确认记录） | 3 |

文本复核（7046 条）：

| 状态 | 条数 |
| --- | ---: |
| `automatic_unproofread`（自动转写，未校对） | 6903 |
| `needs_review`（待复核） | 79 |
| `unreliable_placeholder`（不可靠占位） | 64 |

说话人标签分布（候选标签，7046 条）：

| 标签 | 条数 |
| --- | ---: |
| 旁白 | 2861 |
| 男播音 | 1894 |
| 女播音 | 1739 |
| 说话人待核 | 277 |
| 播音员4（待确认） | 275 |

## 使用前必须注意

1. **标签不是金标准。** 约一半切片仍为 `needs_review`；`automatic` 也只表示自动推定。同名“播音员4”不能跨卷合并为同一真人，`global_speaker_id` 是原标注编号而非已确认身份。组织训练集时先按 `speaker_scope`/`speaker_key` 核对。
2. **文本需校对。** 6903 条为自动转写未校对，64 条是不可靠占位。训练/评测前应做回转写或人工抽检，不能直接当作标准答案。
3. **重复与缺口。** 各包保留 manifest 中全部轮次，包括题号、提示、重复朗读和待核片段，未额外清洗或去重。四级包的 `repeat_index_missing` 为 183，其余包为 0；统计重复播放时需注意。
4. **音频参数不统一由本仓库保证。** 抽样实测为 44.1 kHz / 16-bit / 单声道；批量使用时以每行 `sample_rate_hz`、`channels`、`sample_width_bits` 为准。
5. **切分口径。** `source_start_s`/`source_end_s` 是原始整卷时间，`speech_start_in_clip_s`/`speech_end_in_clip_s` 是片段内位置；首尾留白随片段保留，未按任务书的语速计算口径再次裁剪。

## 分发、校验与许可

- 数据包为 7 个独立 ZIP，每个小于 500 MB；包内 `checksums.sha256` 覆盖所有内容文件，可用于解压后的完整性核对。本仓库不包含这些哈希清单（音频不在仓库中，无法复算）。
- 各包附同一份语音评价 Skill：把“实际音频 + 原文 + 目标要求”交给支持音频理解的模型，输出结构化评价。Skill 源码另见仓库根目录 [`skills/语音评价Skill_20261109/`](../skills/语音评价Skill_20261109/)。
- 切片来自原始考试录音，音频与文本的权利归原权利人。本仓库只做数据集介绍，不再分发音频或原文；使用前请自行确认授权范围，不要用于商业再分发或再许可。
- 如需获取数据包，请联系项目维护者，并说明用途；获取后请先核对 `checksums.sha256`。
