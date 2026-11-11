# model —— 五个候选 TTS 基座的权重

这里放的是任务书第五节要用的候选基座权重，供"先跑零样本克隆基线、横向比较后再选 2 个做 LoRA"这一步使用。
只有权重，没有代码；各模型的 GitHub 仓库还需要另外 clone 到别处（不要 clone 进本目录，免得和权重混在一起）。

目录名一律用官方仓库名，方便和 `../tools/manifests/` 里的清单一一对上。
上层 `../data/` 是汉语听力真题音频，`../task/` 是任务书，`../tools/` 是下载与校验脚本。

## 里面有什么

| 目录 | 上游仓库 | revision（前 12 位） | 大小 |
| --- | --- | --- | --- |
| `IndexTTS-2.5/` | IndexTeam/IndexTTS-2.5 | c39ce5ba9815 | 5.11 GiB + 2.77 GiB 辅助 |
| `VoxCPM2/` | openbmb/VoxCPM2 | 7936080b4116 | 4.62 GiB |
| `Fun-CosyVoice3-0.5B-2512/` | FunAudioLLM/Fun-CosyVoice3-0.5B-2512 | 29e01c4e8d00 | 9.08 GiB |
| `Qwen3-TTS-12Hz-1.7B-Base/` | Qwen/Qwen3-TTS-12Hz-1.7B-Base | fd4b25438912 | 4.23 GiB |
| `FireRedTTS3/` | FireRedTeam/FireRedTTS3 | dcf1bdcd1b8b | 19.35 GiB |
| `CosyVoice-ttsfrd/` | FunAudioLLM/CosyVoice-ttsfrd | 8c0f9244a4f7 | 0.32 GiB |
| `Qwen3-TTS-Tokenizer-12Hz/` | Qwen/Qwen3-TTS-Tokenizer-12Hz | 7dd38ad4e9ba | 0.64 GiB |

后两个不是"第六、第七个模型"，是上面两个模型的必需配套件，理由见下。
完整 revision、逐文件 size 与 sha256 在 `../tools/manifests/*.json`，校验结果在 `verify_report.json`。

### IndexTTS-2.5/

任务书里语速、拼音纠错、情绪控制这三条最对口的一个，README 明确写了支持语速系数和
`<词|读音>` 形式的拼音/CMU/Kana 提示。主仓库是 gpt.pth、s2mel.pth、codec.pth、
qwen0.6bemo4-merge/（情绪用的 Qwen0.6B）、feat1/feat2.pt（说话人与情绪矩阵）。

`hf_cache/` 是我按 `indextts/utils/model_download.py` 里 `ensure_models_available()` 的实际期望路径
预置的四个辅助模型：`w2v-bert-2.0/`、`semantic_codec_model.safetensors`（来自 amphion/MaskGCT）、
`campplus_cn_common.bin`（来自 funasr/campplus）、`bigvgan/`（nvidia/bigvgan_v2_22khz_80band_256x）。
官方 README 说这些"首次运行时自动下载"，提前放好是为了首跑不依赖网络。
注意 `semantic_codec_model.safetensors` 是**平铺文件名**，不是 `semantic_codec/` 子目录——代码按平铺名找。

坑：
- `config.yaml` 里写 `bpe_model: bpe.model`，但仓库里实际是 `multilingual_zh_ja_yue_char_del.tiktoken`，
  没有 `bpe.model` 这个文件。代码走 `get_tokenizer()`，跑通前别按 config 字面去找 bpe.model。
- 情绪接口是 **8 维向量**，顺序 `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`，
  和任务书的"平静/高兴/悲伤/生气"四情绪不是同一套。要做映射，不能直接按四情绪的下标取值。
- README 标称推理约 6 GB 显存，但那是官方说法，24GB 单卡上的实际峰值仍需按任务书要求实测。

### VoxCPM2/

任务书第五节指定的主基座，LoRA 适配三个固定说话人就落在它上面。
权重是单个 `model.safetensors`（4.27 GiB）加 `audiovae.pth`，另有 `tokenization_voxcpm2.py`
（自定义 tokenizer 代码，加载时要 `trust_remote_code`）。结构最简单，没有额外配套件。

### Fun-CosyVoice3-0.5B-2512/

中文发音、多音字、数字读法这条线的主力。注意仓库里有**成对的权重**：
`llm.pt` 与 `llm.rl.pt`（RL 版）、`speech_tokenizer_v3.onnx` 与 `speech_tokenizer_v3.batch.onnx`、
以及 `flow.pt` 与 `flow.decoder.estimator.fp32.onnx`。
做基线时必须固定用哪一份并写进记录，否则横向对比会混入口径差异，把推理配置变化误判成微调收益
（任务书第五节第 4 条专门点了这一点）。`CosyVoice-BlankEN/` 是自带的文本前端基座，
`cosyvoice3.yaml` 里 `qwen_pretrain_path` 为空即指它，不需要另外下 Qwen。

`../CosyVoice-ttsfrd/` 是它 README 的 Model download 一节要求的第二个仓库（文本前端资源），
按官方布局与模型目录平级放置。少了它文本规范化会缺资源。

### Qwen3-TTS-12Hz-1.7B-Base/

零样本克隆与单说话人微调那条线。`speech_tokenizer/` 子目录已含 tokenizer 权重。

`../Qwen3-TTS-Tokenizer-12Hz/` 是同一份权重的独立仓库副本——两者 `model.safetensors` 的
sha256 完全相同（`836b7b357f5ea43e…`）。之所以还单独下一份，是因为官方 README 把 tokenizer
列为独立下载项、qwen-tts 包按仓库名加载；留着是为了兼容那条代码路径。
**这是有意的重复，不是下载出错**；若确认代码只走 Base 内的子目录，可以删掉独立目录省 0.64 GiB。

### FireRedTTS3/

语音编辑与表现力那条线，按你的建议 Base 和 Instruct 都要测，所以两份都下了：
`fireredtts3_base/`（7.90 GiB）、`fireredtts3_instruct/`（7.90 GiB）、`redae/`（3.52 GiB 音频自编码器）、
`campp/`（说话人嵌入）、`text_tokenizer/`。自包含，没有外部辅助权重。

坑：
- 全量 19.35 GiB 是五个里最大的，Base 与 Instruct 是两份独立 8 GB 权重，别当成一份。
  24GB 单卡能否同时驻留 base+redae 需实测，这正是任务书要求记录的显存峰值。
- README 提到可选的 FastText `lid.176` 语种识别模型，**没有下载**：我们的文本是中文，
  按 README 建议直接显式传语言标签即可，不需要自动语种检测。
- Base 依赖显式语言标签才有最佳表现，批量合成时记得逐句带上。

## 校验与溯源

`../tools/fetch_models.py` 先从 HF API 取逐文件 size + sha256 落成清单，再下载，最后按清单校验：
LFS 文件比 sha256，非 LFS 文件比 git blob sha1。清单来源与下载器相互独立，所以校验不是自证。

最近一次全量校验：**7 个仓库 90 个文件，覆盖率 100%，0 缺失 0 不匹配，PASS**。
辅助模型单独校验 7/7 PASS（`verify_report_aux.json`）。

校验门做过反向探针（`../tools/negctl.py`）：在符号链接副本上分别注入 LFS sha256 篡改、
普通文件 sha1 篡改、删文件三种缺陷，门都如实报警并以非零码退出；未篡改的副本 22/22 通过。
所以"PASS"不是空转绿灯。

各仓库内的 `.cache/huggingface/` 是下载元数据，**不要删**（删了会触发重新校验或重下），
也不要提交进 git。

## 已知的推理阻塞点（与下载无关）

本机 `nvidia-smi` 报 `Failed to initialize NVML: Driver/library version mismatch`：
内核模块 NVRM 是 **580.173.02**，用户态 `libnvidia-ml.so.580.178.04` 是 **580.178.04**，两者不一致。
属于驱动升级后未重载内核模块。`nvidia` 模块引用计数 289，说明这台共享节点上有其他作业正在用 GPU，
**不能自行卸载/重载模块或重启**，需要管理员在维护窗口处理。

权重下载与校验不受影响，已全部完成；但在该问题解决前，任何模型都跑不起来，
零样本克隆基线实验要等 GPU 可用之后才能开始。