# 控制接口核查与 P2 设计（赛道 B）

本文件的接口结论**全部对着仓库源码核过**，不是照 README 猜的。
赛道 B 的关键事实是：**六个配置的原生控制能力差别极大**，不能假设都能做语速/情绪控制。

## 一、原生控制接口矩阵（已核验）

| 配置 | 语速 | 情绪 | 语气/指令 | 读音纠错 | 依据 |
| --- | --- | --- | --- | --- | --- |
| IndexTTS-2.5 | `duration_factor` | `emo_vector`（8 维） | 文本驱动 | `<词\|读音>` 拼音/CMU/Kana | `infer_v2_5.py:506` infer 签名 |
| CosyVoice3 | `speed=` | 指令文本 | `inference_instruct2(tts_text, instruct_text, prompt_wav, speed)` | 文本前端 | `cosyvoice.py:177` |
| FireRedTTS3-Base | 无 | 无 | 无 | 无 | `core.py:238` generate 只有推理/前端参数 |
| FireRedTTS3-Instruct | 无 | 指令（见下） | `generate_voice_design(instruction=…)` | 无 | `core.py:340/407` |
| VoxCPM2 | **无** | **无** | **无** | 无 | `core.py:207` _generate 无 speed/emo/instruct 参数 |
| Qwen3-TTS-1.7B-Base | **无** | **无** | **无** | 无 | `qwen3_tts_model.py:470` generate_voice_clone 无 instruct |

两个必须写清的边界：

1. **FireRedTTS3-Instruct 的 `generate_tts` 是 ICL 零样本克隆，签名里没有 instruct 参数。**
   它的指令能力在 `generate_voice_design(instruction=…)`（按描述**设计**音色）
   与 `generate_semantic_edit` / `generate_acoustic_edit`（语音编辑）。
   也就是说"克隆某个参考音色 + 用指令改情绪"这条路，在 Instruct 上**不是现成接口**；
   不能把 voice_design 的表现当成"克隆音色下的情绪控制"。P2 要分开记。
2. **Qwen3-TTS 的指令能力在 CustomVoice / VoiceDesign 变体上**，
   我们下载的是 **Base**，`generate_voice_clone` 没有 instruct。
   不能把别的变体的能力算到 Base 头上。

**选型含义**：任务书要研究语速/停顿/语气/情绪控制，
原生接口最对口的是 **IndexTTS-2.5 与 CosyVoice3**；
VoxCPM2 与 Qwen3-TTS-Base 基本只能靠文本驱动或后处理，
FireRedTTS3-Instruct 强在语音编辑而非克隆态指令控制。
这与"VoxCPM2 作主线、其余竞争第二名额"的预设并不冲突，
但**第二名额若看重控制能力，证据明显偏向 IndexTTS-2.5 / CosyVoice3**。

## 二、P2 项目集（918 条，与方案计数一致）

| 子实验 | 组合 | 条数 |
| --- | --- | --- |
| 语速 | 6 配置 × 3 音色 × 5 档 × 3 文本 | 270 |
| 停顿 | 6 配置 × 3 音色 × 3 设置 × 3 文本 | 162 |
| 语气 | 6 配置 × 3 音色 × 3 句型 × 3 文本 | 162 |
| 情绪 | 6 配置 × 3 音色 × 4 情绪 × 3 文本 | 216 |
| 读音纠错 | 6 配置 × 3 音色 × 6 文本 | 108 |
| 合计 | | **918** |

不支持原生接口的配置**照跑但记 `unsupported`**（用文本驱动兜底并标注），
不静默跳过——方案要求"未测试/接口不支持/运行失败分别记录"。

### 语速档位与口径

目标语速以**字/分钟**给出，档位 160 / 200 / 240 / 280 / 320。
- IndexTTS：`duration_factor` 是时长倍率，与字/分钟成反比，需先测基线再换算，记录换算式。
- CosyVoice3：`speed` 是速率倍率，同样先测基线再换算。
- 其余配置：无原生语速参数，记 unsupported；如需后处理变速，**单独标注为后处理**，
  不混入原生控制成绩（方案明确"原生接口及后处理"要分开）。

实际语速按任务书口径：规范化朗读字数 ÷ 朗读跨度 × 60，
朗读跨度用 20 ms 帧 RMS 削首尾静音、保留句内停顿（与 P0/P1 同一把尺）。
相对偏差 = |实际 − 目标| / 目标 × 100%，验收目标 ≤5%。

### 情绪四类的映射

任务书要平静/高兴/悲伤/生气。IndexTTS 的 8 维顺序是
`[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`，
映射为 平静→calm(idx7)、高兴→happy(idx0)、悲伤→sad(idx2)、生气→angry(idx1)，
**不是按四情绪下标直接取值**。CosyVoice3 / FireRedTTS3 用中文指令文本描述情绪。

### 读音纠错

用 P1 的 12 条发音专项里挑 6 条多音字文本，
IndexTTS 传 `<词\|拼音>` 形式，CosyVoice3 走文本前端，
其余配置无读音提示接口 → 记 unsupported，仅做文本改写对照。

## 三、执行约束

P2 仍走 `mt_serial.py` 的串行机制：一个配置一个进程、跑完释放显存。
按方案" P2 可根据 P1 结果先筛选表现较好的配置"，
**默认只对 P1 通过的配置跑 P2**，避免在跑不通的配置上浪费算力。