# model-test —— 五模型零样本横向评测

本目录是任务书第五节"先建零样本克隆基线、再选基座做 LoRA"的执行现场。
**当前状态：脚手架与测试集已就绪，推理尚未产出任何结果。** 下面如实区分已验证与未验证。

## 一、本机 GPU 实况（与方案假设不同，已实测）

| 项目 | 方案假设 | 实测 |
| --- | --- | --- |
| 卡型 | RTX 4090 | **NVIDIA RTX 6000D** |
| 显存 | 24 GB | **83.05 GiB** |
| 计算能力 | — | sm_120（Blackwell），156 SM |
| CUDA | — | **可用**，torch 2.8.0+cu128，matmul 实测通过 |
| nvidia-smi | 可用 | **不可用**（NVML 版本不匹配） |

`nvidia-smi` 报 `Driver/library version mismatch`：内核模块 580.173.02 vs 用户态库 580.178.04。
**坏的只是 NVML 查询通路，CUDA 驱动 API 正常**，所以推理不受阻。
此前 `model/README.md` 里"任何模型都跑不起来"的结论是错的，已就地更正。

由此产生两条口径约束：

1. **峰值显存不能用 nvidia-smi 测**，改用 `torch.cuda.max_memory_allocated()`，
   每条记录写 `vram_method` 字段说明来源。
2. 24 GB 是**目标约束**不是本机上限。评测照常跑，但每条结果都算 `fits_24gb`，
   最后单独回答"这个配置能不能塞进 4090"，不把 83 GiB 上的宽松结果当成 24 GB 结论。

共享节点，`nvidia` 模块仍有其他作业引用，**不重载模块、不重启**。

## 二、测试集（已构建，但文本未经人工校对）

`scripts/build_testset.py` 生成，随机种子 20261111，可复现。

### 参考音色 refs/

三个身份锚点取自全库**仅有**的 3 条 `speaker_assignment_status == user_confirmed` 记录（均在 HSK4-1）：
MALE `HSK4-1_Q01_U03`、FEMALE `HSK4-1_Q09_U03`、NARRATOR `HSK4-1_Q13_U04`。
锚点本身太短（2.46 s / 2.51 s），不适合直接做克隆参考，因此参考音频改从
**同一 `global_speaker_id`、且 `speaker_assignment_source == confirmed_reference_voice_comparison`**
（即与锚点比对过）的切片里挑 5–12 s、相似度最高的：

| 音色 | 参考切片 | 时长 | 与锚点相似度 | 候选数 |
| --- | --- | --- | --- | --- |
| MALE | HSK5-1_Q17_U03 | 7.179 s | 0.8348 | 481 |
| FEMALE | HSK3-1_Q40_U09 | 7.128 s | 0.8340 | 331 |
| NARRATOR | HSK4-2_Q44_U01 | 5.120 s | 0.7675 | 53 |

**注意**：这三条参考跨了不同试卷（HSK5-1 / HSK3-1 / HSK4-2）。
"跨试卷是否同一真人"仍未人工确认，`refs.json` 里每条都带 `identity_caveat`。
音色相似度分数只能说明"与该锚点嵌入接近"，不等于身份已确认。

### 文本 texts/

- `p0.csv`：10 条**原创**冒烟文本（陈述/疑问/感叹/数字/多音字/长句），不依赖语料，避免被脏标注卡住。
- `p1.csv`：60 条 = 48 条 HSK 分级真题文本（每级 8 条，跨试卷分散、去重）+ 12 条**原创**发音专项
  （多音字 行/长/重/得/还/都/发/觉、数字 年份电话/分数/价格小数、四声对比）。
- `all.jsonl`：合并，供 runner 读取。

生成量与方案一致：P0 = 10×3×6 = **180**，P1 = 60×3×6 = **1080**。

**必须知道的局限**：全库 7046 条文本中 6903 条 `automatic_unproofread`、79 条 `needs_review`、
64 条 `unreliable_placeholder`，**没有任何一条已人工校对**。
已排除全部 placeholder，但选中的语料文本仍逐条保留真实 `review_status` 并标 `needs_human_proofread=true`。
**在人工校对前，CER 只能作为初步筛查，不能当作发音准确性的最终结论。**

## 三、串行与资源控制（按你的唯一要求实现）

`scripts/mt_serial.py` 是编排器，`scripts/mt_run.py` 是单配置工作进程。

- **一个配置一个独立进程，严格串行**：`subprocess.run` 等子进程退出后才起下一个，
  进程退出即释放全部显存与宿主内存，中间再留 3 s 让 OS 回收。绝不并行加载两个模型。
- 进程内另有 `del + empty_cache + ipc_collect + gc`，`torch.set_num_threads(8)` 限制 CPU 线程。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 降低碎片。
- 每条记录都带 `peak_vram_gib` / `reserved_vram_gib` / `rss_gib`，显存和内存都可事后审计。
- 每个配置独立 venv（`envs/<name>`，`--system-site-packages` 复用 torch），
  避免五套依赖互相污染。
- **断点续跑**：runner 启动时读已有结果，`status==ok` 的 (text_id, voice) 直接跳过。

## 四、记录字段与状态口径

结果落 `results/<phase>/<config_id>.jsonl`，一行一条，含：
配置/家族/变体、text_id、voice_id、level、category、原文、拼音提示、文本来源与校对状态、
参考切片 id、输出 wav 路径、采样率、总时长、发声起止与跨度、首尾静音、
规范字数/汉字数、字/分钟、生成墙钟、RTF、峰值显存、RSS、seed、时间戳。

状态四分，**不混为一谈**：`ok` / `failed`（运行报错，带 error_type 与 traceback）/
`unsupported`（接口不支持）/ `not_tested`（本阶段未跑，例如 env 缺失）。

语速口径按任务书：`chars_per_min = 规范字数 ÷ 发声跨度 × 60`，
发声跨度用 20 ms 帧 RMS、相对阈值 0.02 削掉首尾静音，**保留句内停顿**；
方法写进每条记录的 `span_method`，避免口径含糊。

## 五、指标 scripts/mt_metrics.py

- **音色一致性**：CAM++，权重直接用已在盘上的
  `model/IndexTTS-2.5/hf_cache/campplus_cn_common.bin`，特征配方照抄
  `indextts/infer_v2_5.py:640`（16 k、80 维 kaldi fbank、dither=0、减均值），
  与语料说话人标签的产生方式一致，余弦相似度。
- **CER**：需要 ASR 模型。**尚未下载**，未接入时该列记 `not_tested`，不填猜测值。
  文本归一化去标点、可选繁转简（缺 zhconv 时不做并记录）。
- 汇总输出 `results/metrics_<phase>.json`，含成功率、均值、`max_peak_vram_gib`、`fits_24gb`。
- 自然度/语气/情绪走人工盲听，用 `skills/语音评价Skill_20261109`，自动指标不替代人耳。

## 六、已实测的结果与踩过的坑

### P0 首批结果（10 文本 × 3 音色，每配置 30 条）

| 配置 | 成功 | 音色相似度 | CER | 字/分钟 | RTF | 峰值显存 | 塞得进 24GB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VoxCPM2 | 30/30 | 0.7176 | 0.0 | 244.5 | 0.635 | 5.58 GiB | 是 |
| Qwen3-TTS-1.7B-Base | 30/30 | 0.7123 | 0.0 | 249.4 | 1.550 | 4.35 GiB | 是 |
| IndexTTS-2.5 | 修复中 | — | — | — | — | — | — |
| CosyVoice3 | 修复中 | — | — | — | — | — | — |
| FireRedTTS3-Base | 修复中 | — | — | — | — | — | — |
| FireRedTTS3-Instruct | 修复中 | — | — | — | — | — | — |

**P0 的 CER=0 没有区分度**（天花板效应）：10 条冒烟文本又短又干净，两个模型都全对。
真正能拉开差距的是 P1 的长句、多音字与数字专项。不要把 P0 的 0 当成"发音完美"的结论。

### 坑 1：CER 口径——ITN 会把正确朗读判成错

第一次算 CER 得到 **0.1042**，看着像发音很差。逐条查最差样本发现全是**测量口径问题，不是合成错误**：

| 文本 | ASR 回写（ITN 开） | 判 CER |
| --- | --- | --- |
| 他的电话号码是**一三八零九零一二三四五**。 | 他的电话号码是**13809012345**。 | 0.611 |
| 会议改到**三月五号**下午**两点半**。 | 会议改到**3月5号**下午**2点半**。 | 0.231 |
| 请你把书翻到第**十五**页。 | 请你把书翻到第**15**页。 | 0.200 |

音频读得完全正确，是 SenseVoice 的 ITN 把汉字数字改写成阿拉伯数字，与汉字形式的参考文本对不上。
**改成 `use_itn=False` 后 CER 归零**。参考文本是汉字形式，就必须关掉 ITN 对齐口径。
这也顺带成了 CER 管路的正向对照：同一批音频，口径错时 0.611、口径对时 0.0，
证明这个指标不是恒等于 0 的摆设。

### 坑 2：适配器初版四个签名全错（已按源码逐一改正）

第一版适配器是照 README 写的，对着源码核验后发现四处都错：

- `CosyVoice3.__init__` **没有 `load_jit` 参数**（只有 load_trt/load_vllm/fp16）。
- IndexTTS 语速参数是 **`duration_factor`，不是 `speed`**；传 `speed` 会落进
  `**generation_kwargs` 被静默忽略——典型的"参数传了但没生效"。
- Qwen3-TTS 用自己的 **`qwen_tts.Qwen3TTSModel`**，不是 transformers `AutoModel`。
- FireRedTTS3 在 **`fireredtts3.core`**（没有 api.py），构造参数是 `pretrained_model_dir`，
  语种标签是 `"Chinese"` 而不是 `"zh"`；且 `use_fasttext` 默认 True 会去找没下载的 lid.176，
  必须显式关掉并传语种。

### 坑 3：环境依赖（每个都对应一条真实报错）

| 配置 | 报错 | 处理 |
| --- | --- | --- |
| IndexTTS | `requires a different Python: 3.12.14 not in '<3.12,>=3.10'` | 改用 mamma 的 Python 3.11.18 重建 venv |
| IndexTTS | `operator torchvision::nms does not exist` | venv 里 torch 2.8.0 配了 torchvision 0.22.1（属 2.7.1），装 0.23.0 对齐 |
| CosyVoice | `deepspeed==0.15.1` 需要 CUDA_HOME/nvcc | DeepSpeed 是**训练**依赖，推理不需要，从 requirements 剔除 |
| CosyVoice | `No module named 'hyperpyyaml'` | 补装 HyperPyYAML==1.2.3 |
| FireRedTTS3 | `flash-attn` 需要 nvcc | 剔除；见下方代码改动声明 |
| FireRedTTS3 | `FlashAttention2 has been toggled on ... not installed` | 见下方代码改动声明 |

另外：aliyun 镜像 pip 实测只有 200–400 KB/s，直接 curl 测同一文件却有 16 MB/s，
清华源 32 MB/s。已给**这几个实验 venv 各自**写 `pip.conf` 指向清华源（不动全局配置），
后续安装从几十分钟降到几分钟。

### 代码改动声明（不是静默修改）

FireRedTTS3 把 `attn_implementation='flash_attention_2'` **硬编码**在
`fireredtts3/redae/redae.py:37,122` 与 `fireredtts3/llm/fireredtts3_base.py:49`。
本机没有 nvcc，装不了 flash-attn，因此把这三处字面量改成 `"sdpa"`，
原文件备份为 `*.orig_flash`。**这是注意力后端的改动**，已写进每条推理结果的
`attn_backend` 字段，比较时按同一后端口径看待，不当作模型原生配置。
同理 Qwen3-TTS 适配器在检测不到 flash_attn 时回退 sdpa，并记录 `attn` 字段。

## 七、当前进度与续跑方法

已完成并验证：

- [x] GPU/CUDA 实测，更正错误结论
- [x] 五个代码仓库 clone 到 `repos/`
- [x] 参考音色 3 条 + P0 10 条 + P1 60 条文本，计数与方案一致
- [x] 串行编排器、单配置 runner、6 个适配器（签名已逐一核对源码）、指标脚本
- [x] ASR（SenseVoiceSmall）已下载并接入，CER 口径已修正并做过正向对照
- [x] **VoxCPM2 P0 30/30、Qwen3-TTS-Base P0 30/30**，含相似度/CER/语速/RTF/峰值显存

进行中 / 未完成：

- [ ] IndexTTS-2.5 / CosyVoice3 / FireRedTTS3-Base / FireRedTTS3-Instruct 的 P0
      （依赖问题已定位并修复，正在重跑）
- [ ] **P1 未开始**（1080 条），P2/P3 未开始
- [ ] 人工盲听未做；自然度/语气/情绪尚无评分
- [ ] 赛道 A/B 的加权总分未算（需先固定归一化方式，且人工维度缺位）
- [ ] 语料文本仍全部未经人工校对，CER 只能作初步筛查

续跑命令（在远端）：

```bash
E=/data/shangguanrz/projectsss/speakgenerate/experiments/model-test
# 单配置冒烟（先 3 条验证通路）
PYTHONPATH=$E/scripts $E/envs/voxcpm/bin/python $E/scripts/mt_run.py \
    --config voxcpm2 --phase P0 --limit 3
# P0 全量，严格串行
$HOME/miniconda3/envs/deepgen/bin/python $E/scripts/mt_serial.py --phase P0
# P0 通过后再 P1
$HOME/miniconda3/envs/deepgen/bin/python $E/scripts/mt_serial.py --phase P1
# 指标（先只算音色相似度）
PYTHONPATH=$E/scripts $E/envs/voxcpm/bin/python $E/scripts/mt_metrics.py \
    --phase P0 --speaker-only
```

## 八、选型口径（尚未有数据支撑）

方案建议保留 VoxCPM2 作主线、其余四个竞争第二个名额。
**这是待验证的假设，不是结论**——目前一条音频都还没生成。
FireRedTTS3 的 Base 与 Instruct 分开记为两个配置，但家族仍按五个统计；
Instruct 的编辑收益不得计入 Base 的零样本能力。