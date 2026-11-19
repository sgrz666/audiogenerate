# 评测流水线状态与续跑说明（停车快照）

本文件记录流水线在 Agent 轮次耗尽时的确切状态。所有脚本以 `setsid nohup` 启动，
**会持续自主运行**，不依赖 Agent 在线。全程串行、一配置一进程、跑完释放显存。

## 一、已验证完成

- 5 个模型权重 + 配套件已下载并逐文件 sha256/sha1 校验（见 `../../model/README.md`）。
- GPU 实测：RTX 6000D 83 GiB，sm_120，CUDA 可用；NVML 坏（nvidia-smi 不可用），
  峰值显存用 `torch.cuda.max_memory_allocated()`。**不是 24 GB 卡**，24 GB 是目标约束，
  每条结果带 `fits_24gb` 标记。
- 测试集：3 参考音色（锚点取自仅有的 3 条 user_confirmed）+ P0 10 文本 + P1 60 文本
  （48 HSK 分级 + 12 发音专项）。**语料文本全部未经人工校对**，参考音色跨试卷、身份未确认。
- 6 个适配器签名已对源码核验；P0 已 6 配置跑通（CosyVoice3/Instruct 经环境修复后通过）。
- 修复过的真实缺陷：CER 关 ITN 对齐参考文本形态；结果日志按 item 去重（ok 优先）；
  FireRedTTS3-Instruct 上游 unpack bug（`core.py:384` 解包 3 值但后端返回 2 值）已最小修复并备份。

## 二、进行中（自主运行）

链路：`P1(4配置) → round3(CosyVoice3/Instruct 的 P0+P1) → finalize(指标/评分/报告)
→ run_p2p3(P2 控制 918 条 + P3 稳定性) → finalize2(赛道B评分+报告)`

停车时：P1 的 VoxCPM2 / IndexTTS-2.5 / Qwen3-TTS-Base 各 180/180 已完成；
FireRedTTS3-Base P1 约 160/180。CosyVoice3 与 Instruct 的 P0/P1 尚未开始（排在 round3）。

## 三、如何查看进度 / 结果

```bash
E=/data/shangguanrz/projectsss/speakgenerate/experiments/model-test
# 各阶段日志
tail -f $E/logs/p1_driver.log $E/logs/round3_driver.log $E/logs/finalize.log \
        $E/logs/p2p3_driver.log $E/logs/finalize2.log
# 音频计数
for p in P0 P1 P2 P3; do for d in $E/audio/$p/*/; do \
  printf "%s/%s " $p $(basename $d); find $d -name "*.wav"|wc -l; done; done
# 最终报告与评分
cat $E/RESULTS.md
cat $E/results/score_P1.json $E/results/score_trackB.json
```

完成标志：`finalize.log` 出现 `FINALIZE DONE`（P0/P1 收口）；
`finalize2.log` 出现 `FINALIZE2 DONE`（含 P2/P3 的全链路收口）。

## 四、尚未完成（不含糊）

- P1 的 FireRedTTS3-Base 收尾 + CosyVoice3/Instruct 的 P0/P1。
- P1 指标/评分/报告（finalize 自动产出）。
- P2 控制实验（918 条）与 P3 稳定性（run_p2p3 自动跑）。
- **人工匿名盲听**：`mt_blind.py` 会生成盲听包（不透明编号 + 密封 answer_key），
  但**主观评分必须由人完成**，脚本不产生主观分。
- **最终选型结论**：需人工盲听 + P2 控制结果补齐后才下。目前只有客观部分分
  （赛道 A 覆盖 65% 权重），不足以定论。

## 五、若需中断后恢复

各 runner 都支持断点续跑（按 item 去重、ok 跳过）。重跑某阶段直接再执行对应脚本即可，
不会重复已成功的项。切勿并行起两个 GPU 进程——串行是硬约束。
## 六、补充修复（停车后）

- **FireRedTTS3-Instruct**：上游 unpack bug 修复后 P0 已 30/30 通过，P1 自动放行并在跑。
- **CosyVoice3**：真正阻塞是 `third_party/Matcha-TTS` git submodule 为空
  （clone 时未带 --recurse-submodules），导致 `No module named matcha`。
  已补 clone Matcha-TTS，`from cosyvoice.cli.cosyvoice import CosyVoice3` 全链路导入通过。
  其 P0/P1/P2/P3 由 `cosyvoice_retry.sh` 在主链路 FINALIZE2 之后串行补跑，
  完成后重算指标/评分/盲听包/报告。完成标志：`cosyretry_driver.log` 出现 `COSYRETRY DONE`。

## 七、CosyVoice3 修复链（最终跑通）

CosyVoice3 连续踩了四个坑，逐一对源码修复后才跑通：
1. `third_party/Matcha-TTS` git submodule 为空 → 补 clone。
2. `No module named pkg_resources`：venv 里 setuptools 83 已移除 pkg_resources → 装 `setuptools<81`。
3. `Invalid file: tensor`：此版本 `inference_zero_shot` 的 prompt_wav 要**文件路径**
   （frontend 内部自己 load_wav），不能传预加载 tensor → 改传路径。
4. `<|endofprompt|> not detected`：CosyVoice3 要求 prompt_text 末尾带 `<|endofprompt|>`
   （llm.py:479 断言 token 151646）→ prompt_text 追加该 token；instruct2 用
   "You are a helpful assistant. <指令><|endofprompt|>" 格式。

修复后 P0 27/30 成功。余 3 条失败是**模型短文本边界缺陷**
（`Kernel size(4) > input size(3)`，conv 输入过短），非适配器问题，如实记录。

## 八、CosyVoice3 调试记录（截至本轮，仍 output_invalid）

已尝试并排除的修复（均仍产 garbage，回转写与原文无关）：
1. 补 clone 空 submodule Matcha-TTS —— 解决 import，但仍 garbage。
2. setuptools<81 恢复 pkg_resources —— 解决加载，但仍 garbage。
3. prompt_wav 传文件路径而非 tensor —— 解决 "Invalid file"，但仍 garbage。
4. prompt_text 追加 <|endofprompt|> —— 通过 llm.py 断言，但仍 garbage。
5. text_frontend=False（官方复现建议）—— 仍 garbage。

结论：裸 CosyVoice3 类的 zero_shot 路径在此 build 下生成内容错误，非上述表层问题。
AutoModel 仅是同一类的薄封装，非差异点。需更深层调试（疑似 llm.pt 加载/参考音频条件化/
speech_tokenizer 版本），**CosyVoice3 维持 output_invalid，不参与排名**。
其余 5 配置（VoxCPM2/IndexTTS-2.5/Qwen3-TTS-Base/FireRedTTS3-Base/Instruct）结果有效。

### 补充诊断：参考音频已排除
三个参考切片经 SenseVoice 回转写均与 ref_text 一致、RMS 0.06–0.07、时长 5–7s，
**参考音频有效，非 garbage 根因**。CosyVoice3 在输入有效的前提下仍产 garbage，
确认为该 build zero_shot 路径的模型/集成缺陷（疑似 llm.pt / speech_tokenizer 版本），
需 deeper debugging。CosyVoice3 维持 output_invalid。

## 九、P2 v2 收口（2026-11-13）

**修复了两个 v1 失效项**：
1. 停顿三档文本（`build_p2.py` 的 `mark` 从未拼进文本）→ `texts/p2b.jsonl` 三种文本变体，405 条（5 配置 × 3 音色 × 3 设置 × 3 文本 × 3 seeds）。
2. IndexTTS 读音提示 `<行|xíng>` → `<行|HANG2>`（数字声调），修掉 `_pinyin_text` 首次出现替换导致的嵌套；PR1 改为 `行|HANG2;长|CHANG2`。pilot PR2/PR4 CER 0.000，18 条正样本 + 6 条故意错读负对照全部生成。

**P2 v2 客观结果**：
- 停顿（同文本 speech-span 中位增量，3 seeds）：FireRed Base **9/9 正、+0.38 s、p=0.0039**；Instruct **+0.48 s、7/9**；VoxCPM2 −0.04 s；IndexTTS/Qwen +0.06 s 且低于 v1 同文本噪声（0.12/0.20 s）→ 文本标点对 FireRed 两配置稳定，对其余三配置不稳定。
- 语气（F0-only η²/ω²）：IndexTTS ω²=0.286（effect 1.0）；Qwen3 0.0038（0.032）；VoxCPM2/FireRed 两配置 ω²≤0。
- 情绪（仅 IndexTTS 原生）：LOO 最近质心 21/36=0.583，obj 0.444；混淆主要在 sad↔happy、calm↔angry。
- 读音：正样本 18 条平均 CER **0.0074**（内容门通过，格式修复不再逐字读字母）；但 MiniCPM tone 门未过（负对照只检出 3/6，PR1 正样本报 xíng 而非 háng），tone 分不采用，读音维度只按内容计 0.3。

**MiniCPM-o 4.5 听评**：ModelScope 18.67 GiB 下载 + manifest 校验 53/54 通过（仅 `.gitattributes` 大小差异，非权重）；`envs/minicpmo` 用 transformers 4.51.0。校准 8 个硬检查全对（agreement 1.00）；削波/断音/两档停顿为 soft（模型对音质无感）。正式听评 142 条：语气/情绪/读音；停顿二选一在 none 上全报“有”，偏置明显，**pause 听评不用于计分**。

**分数**（`results/score_trackB_v2.json`）：IndexTTS 权重覆盖 **0.795**（缺 pause 听评 0.10 + tone 0.105）；其余 4 配置 **0.35**（缺 native rate/emotion/pron）。`trackB_full_score` 全部为 null、`rankable=false`；这是覆盖率事实，不是 0 分。

**待办**：本地 `P2_听评包/index.html`（30 条匿名）人工抽检，回填 pause 与 tone 后重算；`RESULTS_P2.md` 为分维度报告，`docs/P2_RUBRIC.md` 为预注册口径。
