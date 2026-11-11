# tools —— 模型下载、校验与辅助脚本

这些脚本服务于“候选基座下载与校验”和“数据分包下载”。所有脚本都在项目根目录执行，内部的 `ROOT` / `MANIFEST_DIR` / `DEST` 是当时机器上的绝对路径，换环境时先改。

## 脚本一览

| 文件 | 作用 |
| --- | --- |
| `fetch_models.py` | `manifest` 从 HF API 生成逐文件 size + sha256 清单；`fetch` 下载 5 个基座 + 2 个配套仓库；`verify` 逐文件校验，覆盖率不足 100% 即失败 |
| `fetch_aux.py` | 预取 IndexTTS-2.5 的 4 个辅助模型到代码期望的 `hf_cache/` 路径，并单独生成/校验 `IndexTTS-2.5-aux.json` |
| `negctl.py` | 校验门的反向探针：在符号链接副本上注入 LFS sha256 篡改、普通文件 sha1 篡改、删文件三种缺陷，要求全部报警并以非零码退出 |
| `hf_probe.py` | 探测 HF 仓库是否存在、取 revision 等元信息 |
| `hf_tree.py` | 打印 HF 仓库文件树与大小 |
| `docx_text.py` | 仅用标准库从 `.docx` 提取纯文本，用于读任务书 |
| `bd_pull_parallel.py` | 百度网盘分包下载器：64 KB 分块、并发、6 秒截止、断点续传，按大小与 MD5/CRC 双校验后原子落位 |

## 模型下载与校验

依赖 Python 3、`huggingface_hub`、`hf_transfer`，约 47 GiB 磁盘。命令：

```bash
python tools/fetch_models.py manifest   # 生成 tools/manifests/*.json
python tools/fetch_models.py fetch      # 下载到 model/<仓库名>/
python tools/fetch_models.py verify     # 逐文件校验，输出 verify_report.json
python tools/fetch_aux.py all           # IndexTTS 辅助模型：manifest + fetch + verify
python tools/negctl.py                  # 反向探针（需要真实模型已下载）
```

`--root` 可覆盖校验时的模型根目录，`--repos` 可限定仓库，`--workers` 控制并发。下载会交替尝试 `huggingface.co` 与 `hf-mirror.com`，支持断点续传。

校验口径：

- 清单来自 HF API，与下载器相互独立，避免“自己验自己”。
- LFS 文件比 `sha256`，非 LFS 文件比 git blob `sha1`，同时核对文件大小。
- 报告包含 `n_files`、`n_ok`、`coverage`、`bad`；`bad` 非空或覆盖不足即非零退出。
- `negctl.py` 是反向探针：先在未篡改副本上跑正控，再分别注入 SHA-256 篡改、SHA-1 篡改、删文件三种缺陷，要求门报警且非零退出。`PASS` 不是空转绿灯。

当前结果（`tools/manifests/`）：

- 主清单：7 个仓库、90 个文件、覆盖率 100%、0 缺失、0 不匹配。
- IndexTTS 辅助件：7 个文件、7/7 通过（`verify_report_aux.json`）。
- 反向探针：三种缺陷均按预期报警，未篡改副本 22/22 通过。

## 数据分包下载

`bd_pull_parallel.py` 是数据 ZIP 的下载工具，不属于模型或训练流程：

- 输入：`/tmp/bd_files.json`（文件名、字节数）与 `/tmp/bdjar.txt`（Netscape cookie jar）。
- 输出：`data/` 下的分包 ZIP，先写 `.part`，完成后按大小与 MD5/CRC 校验再原子改名。
- 参数：`BD_WORKERS`（默认 4）、`BD_DEADLINE`（默认 6 秒）两个环境变量。
- 仓库中不包含任何 cookie、分享链接或账号信息；使用前需自行准备有权限的下载凭据。

## 日志

`logs/` 是 2026-11-11 下载基座时的原始日志，包括一次 IndexTTS 辅助件首跑缺文件（6/7 FAIL）——该文件随后补齐，最终结果以 `manifests/verify_report_aux.json` 的 7/7 为准。日志保留是为了说明下载和校验过程，不是最终状态。

## 环境备注

- 模型下载完成于 `node01`，当时 `nvidia-smi` 因内核模块与用户态库版本不一致而不可用，所以本轮只做了权重下载与校验，未做任何推理。
- `fetch_models.py`、`fetch_aux.py`、`negctl.py` 内的路径写死在项目目录；迁移到新机器时请改成自己的 `model/` 与 `tools/manifests/`。
