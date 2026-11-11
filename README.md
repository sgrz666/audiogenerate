# 汉语听力考试语音合成（audiogenerate）

大学生创新训练项目。目标是在三个固定播音音色下，研究汉语听力考试音频的**语速、停顿、语气和基础情绪控制**，改善多音字、数字读法与播报自然度，并形成可复现的训练、推理和组卷流程。

项目以开源语音合成模型为基础：先跑零样本克隆基线，横向比较候选基座，再选两个做 LoRA 适配；最终交付三个音色的适配参数、数据清单与标注说明、训练/推理脚本、组卷模板、简易页面和对比音频。

任务书见 [`task/任务书-汉语听力考试语音合成-v8.docx`](task/任务书-汉语听力考试语音合成-v8.docx)（纯文本可用 `python tools/docx_text.py` 提取）。

## 当前进度（2026-11-11）

- **候选基座已就位**：5 个 TTS 基座 + 2 个必需配套仓库，主清单 90 个文件、IndexTTS 辅助件 7 个文件，均按 HF API 清单完成 size + sha256 / git-blob sha1 校验，覆盖率 100%，0 缺失、0 不匹配（权重合计约 46 GiB，磁盘含缓存约 47 GiB）。校验门做过反向探针，篡改/删文件都会报警。详见 [`model/README.md`](model/README.md) 与 [`tools/manifests/`](tools/manifests/)。
- **数据已盘点**：7 个分包、36 套材料、7046 条切片、约 9.04 小时、2.87 GB；音频为 44.1 kHz / 16-bit / 单声道 WAV，逐条配同名 TXT 与标注表。详见 [`data/README.md`](data/README.md)。
- **实验尚未开始**：本机 `nvidia-smi` 报 `Failed to initialize NVML: Driver/library version mismatch`（内核 NVRM 580.173.02，用户态库 580.178.04），共享节点上还有其他 GPU 作业，不能自行重载模块或重启，需等管理员处理。在此之前零样本克隆基线无法运行。
- **仓库不含**：模型权重、数据音频/转写原文、下载凭据与临时目录。模型只保留清单与说明，数据只保留数据集介绍和分包统计。

## 目录结构

```
.
├── README.md                     # 本文件
├── data/
│   ├── README.md                 # 数据集介绍：来源、规模、字段、标注状态、使用限制
│   └── packages/                 # 7 个分包的 README + summary.json / volumes.json
│       └── <分包名>/metadata/
├── model/
│   └── README.md                 # 5 个候选基座 + 2 个配套件的版本、大小、坑位说明
├── task/
│   └── 任务书-汉语听力考试语音合成-v8.docx
├── tools/
│   ├── fetch_models.py           # 建清单 / 下载 / 逐文件校验 HF 模型
│   ├── fetch_aux.py              # IndexTTS-2.5 辅助模型预取与校验
│   ├── negctl.py                 # 校验门的反向探针（注入篡改/缺失）
│   ├── hf_probe.py / hf_tree.py  # HF API 探测与仓库树查看
│   ├── docx_text.py              # 仅用标准库提取 docx 文本
│   ├── bd_pull_parallel.py       # 百度网盘分包下载器（分块/并发/续传/双校验）
│   ├── manifests/                # 逐文件 size + sha256 清单与校验报告
│   ├── logs/                     # 原始下载日志
│   └── README.md                 # 各脚本的用法、校验口径与环境备注
├── skills/
│   └── 语音评价Skill_20261109/   # 按实际音频 + 原文 + 要求输出结构化评价
├── experiments/                  # 预留：基线/微调/对比实验，目前为空
└── .gitignore
```

## 候选基座一览

| 目录（未入库） | 上游仓库 | 大小 | 项目中的角色 |
| --- | --- | --- | --- |
| `IndexTTS-2.5/` | IndexTeam/IndexTTS-2.5 | 5.11 GiB + 2.77 GiB 辅助 | 语速系数、拼音纠错、8 维情绪接口 |
| `VoxCPM2/` | openbmb/VoxCPM2 | 4.62 GiB | 任务书指定主基座，三个说话人 LoRA |
| `Fun-CosyVoice3-0.5B-2512/` | FunAudioLLM/Fun-CosyVoice3-0.5B-2512 | 9.08 GiB | 中文发音、多音字、数字读法 |
| `Qwen3-TTS-12Hz-1.7B-Base/` | Qwen/Qwen3-TTS-12Hz-1.7B-Base | 4.23 GiB | 零样本克隆、单说话人微调 |
| `FireRedTTS3/` | FireRedTeam/FireRedTTS3 | 19.35 GiB | 语音编辑与表现力（Base + Instruct） |
| `CosyVoice-ttsfrd/` | FunAudioLLM/CosyVoice-ttsfrd | 0.32 GiB | CosyVoice3 文本前端配套 |
| `Qwen3-TTS-Tokenizer-12Hz/` | Qwen/Qwen3-TTS-Tokenizer-12Hz | 0.64 GiB | Qwen3-TTS 独立 tokenizer 配套 |

权重目录名与上游仓库名一致，完整 revision、逐文件大小与哈希见 [`tools/manifests/`](tools/manifests/)。各模型的加载注意事项（如 IndexTTS 的平铺 `semantic_codec_model.safetensors`、CosyVoice 成对权重的口径选择）写在 [`model/README.md`](model/README.md)。

## 数据集速览

| 分包 | 套数 | 题数 | 切片 | 时长 | 音频大小 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 一级（1-5 + 大纲样卷） | 6 | 120 | 526 | 26.4 min | 139.6 MB |
| 二级（1-5 + 大纲样卷） | 6 | 210 | 1366 | 74.8 min | 396.1 MB |
| 三级（1-5 + 大纲样卷） | 6 | 240 | 1906 | 124.0 min | 656.2 MB |
| 四级（1-5 + 大纲样卷） | 6 | 270 | 1156 | 85.6 min | 453.2 MB |
| 五级（1-5 + 大纲样卷） | 6 | 270 | 1180 | 93.7 min | 496.0 MB |
| 六级（真题 1-3） | 3 | 150 | 467 | 69.0 min | 365.4 MB |
| 六级（真题 4-5 + 大纲样卷） | 3 | 150 | 445 | 69.1 min | 365.5 MB |
| **合计** | **36** | **1410** | **7046** | **9.04 h** | **2.87 GB** |

数据来自原始考试录音的切片，保留自动转写与候选说话人状态，**没有**把待核标签当成人工确认事实。字段、标注质量分布、已知缺口与使用限制见 [`data/README.md`](data/README.md)。

## 复现方式

### 1. 取模型权重并校验

需要 Python 3、`huggingface_hub`、`hf_transfer`，以及约 47 GiB 磁盘。脚本里的 `ROOT` / `MANIFEST_DIR` 是项目当时的绝对路径，换机器时先改这两处。

```bash
python tools/fetch_models.py manifest   # 从 HF API 生成逐文件 size + sha256 清单
python tools/fetch_models.py fetch      # 下载 5 个基座 + 2 个配套仓库
python tools/fetch_models.py verify     # 逐文件校验，非 100% 覆盖率即失败
python tools/fetch_aux.py all           # IndexTTS-2.5 辅助模型：清单 + 下载 + 校验
```

校验原则：清单来自 HF API，与下载器相互独立；LFS 文件比 sha256，非 LFS 文件比 git blob sha1。校验门的反向探针：

```bash
python tools/negctl.py
```

它会在符号链接副本上分别注入 LFS sha256 篡改、普通文件 sha1 篡改、删文件三种缺陷，要求校验门报警并以非零码退出，同时要求未篡改副本通过。

### 2. 数据

音频分包不随仓库分发（7 个独立 ZIP，每个小于 500 MB，便于传输）。本仓库保留数据集介绍、7 个分包 README 与分包统计，便于核对字段和申请数据。数据包内自带 `checksums.sha256` 和离线试听页 `index.html`；各包的源归档 sha256 一致，为 `4bba0058dc968425615f71f34b4006c2139f1f00c7e38fa06b390be9c0cf8f7f`。

### 3. 任务书文本

```bash
python tools/docx_text.py "task/任务书-汉语听力考试语音合成-v8.docx"
```

### 4. 数据分发包下载（可选）

`tools/bd_pull_parallel.py` 是下载分包用的百度网盘分块下载器，读取 `/tmp/bd_files.json`（文件名、大小）和 `/tmp/bdjar.txt`（Netscape cookie jar），每块 64 KB、6 秒截止、多线程、断点续传，并按文件大小与 MD5/CRC 校验后原子落位。它不是模型或训练流程的一部分，仓库中不包含任何 cookie、分享链接或凭据。

## 已知限制

- 数据标签是“自动初标 + 待核”状态，不是人工金标准。说话人同名不代表同一真人，跨卷不能直接合并；文本为自动转写，训练前需要复核。统计见 [`data/README.md`](data/README.md)。
- 数据切片来自原始考试录音，音频与原文的权利归原权利人。本仓库只提供介绍，不再分发音频；使用前请自行确认授权范围，不要用于商业再分发。
- 模型权重的许可跟随上游仓库，下载前请阅读各自 LICENSE。
- 当前节点 GPU 驱动/用户态库版本不一致，推理与训练均未开始；这是环境问题，不是模型或数据问题。

## 下一步

- [ ] 修复/等待 GPU 节点 NVML 版本不一致问题
- [ ] 零样本克隆基线：固定测试文本、参考音频与推理参数
- [ ] 三个固定说话人的 LoRA 适配（VoxCPM2 主线）
- [ ] 语速/停顿、语气、四情绪控制的有效范围与稳定性实验
- [ ] 整套文稿解析、模板组卷、局部重生成与导出页面
- [ ] 对比音频与测试报告
