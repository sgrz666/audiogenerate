# 中文听力切片数据 · 四级

本包包含 6 套完整材料、270 题、1156 条 WAV 及对应同名 TXT。
套题：HSK4-1, HSK4-2, HSK4-3, HSK4-4, HSK4-5, HSK4-outline。数字后缀 1–5 为2018真题，outline 为考试大纲样卷。

## 使用方法

这是独立 ZIP，先完整解压，再双击根目录的 `index.html`，即可离线试听；不需要其他包，也无需合并分卷。
页面支持按套题和题号查看、搜索文稿、按候选说话人与朗读角色筛选、播放切片，以及查看待核说明。
请保留 `index.html` 与 `data` 文件夹的相对位置。页面内的题组/材料关联仅跳转到原标注关联题，不自动合并前题音频。
音频与对应文本按套题放在 `data/套题编号/`；每套的 `标注.tsv` 包含文本、角色、说话人标签及复核状态，可用表格软件打开。
`audio_file`、`text_file` 是与该套标注表同目录的文件名；其余路径字段均相对于本包根目录。

```
index.html                    双击打开的离线试听入口
data/套题编号/*.wav、*.txt     逐条配对的原样音频和文本
data/套题编号/标注.tsv        该套完整标注，含说话人标签
metadata/samples.tsv          本包全部标注（UTF-8 BOM）
metadata/samples.jsonl        同一数据的结构化版本，保留数组与空值
metadata/questions.jsonl      题目、轮次和共用材料关联
metadata/volumes.json         本包各套来源与统计
metadata/summary.json         本包统计与状态
evaluation/                  评价 Skill、输入示例、输出格式、使用说明
checksums.sha256              本包所有内容文件的 SHA-256
```

## 标签说明

- `speaker_label` 为中文候选标签，`role` 为题号、提示、对话、提问等功能。
- `speaker_scope`、`speaker_key` 按套题限定候选身份；`global_speaker_id` 沿用原标注，并非已经确认的跨卷真人身份。同名“播音员4”不能直接跨卷合并。
- `speaker_assignment_status`：automatic 为自动推定，needs_review 为待核，user_confirmed 为已有确认记录。原有置信度、flags 和复核备注全部保留。
- `text_review_status` 保留自动转写/待复核状态，未将文本或身份改成已人工确认。
- `source_start_s/source_end_s` 为原始整卷时间，`speech_start_in_clip_s/speech_end_in_clip_s` 为当前片段内位置。缺失的题型和遍次仍为 null/表格空值。

## 数据范围与评价 Skill

本次仅重新分包，没有修改音频、文本或说话人标签。题号、提示、重复朗读、首尾余量和待核片段沿用原交付版本，没有另做清洗或去重。
未收录缓存、模型、旧版、备份、整卷原始录音或整题 WAV。

这些切片来自原始考试录音；评价 Skill 用于后续待评音频，将实际音频、原文、目标要求与 Skill 一起交给支持音频理解的模型。示例音频文件名是占位示例。本包没有实际模型评价报告。
