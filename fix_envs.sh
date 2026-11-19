#!/usr/bin/env bash
# Fix the three envs that failed, based on the actual install errors:
#   indextts   : "requires a different Python: 3.12.14 not in '<3.12,>=3.10'"
#                -> rebuild from the mamma env (Python 3.11.18, torch 2.7.1+cu128)
#   cosyvoice  : deepspeed==0.15.1 needs CUDA_HOME/nvcc to build. DeepSpeed is a
#                TRAINING dependency, not needed for zero-shot inference -> skip.
#   fireredtts : flash-attn needs nvcc. It is optional; the adapter falls back to
#                sdpa -> skip.
# Nothing here is a workaround for a model defect; these are build-time-only deps.
set -u
E=/data/shangguanrz/projectsss/speakgenerate/experiments/model-test
LOG=$E/logs
PY311=$HOME/miniconda3/envs/mamma/bin/python
IDX="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "=== FIX ENVS start $(date +%H:%M:%S) ==="
$PY311 -V

# ---- indextts: rebuild on Python 3.11 ----
echo "--- indextts (py3.11) ---"
rm -rf "$E/envs/indextts"
$PY311 -m venv --system-site-packages "$E/envs/indextts" || exit 1
printf "[global]\nindex-url = %s\ntrusted-host = pypi.tuna.tsinghua.edu.cn\n" "$IDX" > "$E/envs/indextts/pip.conf"
"$E/envs/indextts/bin/pip" install -e "$E/repos/index-tts" > "$LOG/fix_indextts.log" 2>&1
echo "indextts pip exit=$?"
tail -2 "$LOG/fix_indextts.log"

# ---- cosyvoice: requirements minus deepspeed ----
echo "--- cosyvoice (skip deepspeed) ---"
grep -v -i -E "^\s*(deepspeed|flash[-_]attn)" "$E/repos/CosyVoice/requirements.txt" > "$E/cosyvoice_req_infer.txt"
echo "skipped lines:"; grep -i -E "^\s*(deepspeed|flash[-_]attn)" "$E/repos/CosyVoice/requirements.txt"
"$E/envs/cosyvoice/bin/pip" install -r "$E/cosyvoice_req_infer.txt" > "$LOG/fix_cosyvoice.log" 2>&1
echo "cosyvoice pip exit=$?"
tail -2 "$LOG/fix_cosyvoice.log"

# ---- fireredtts: requirements minus flash-attn ----
echo "--- fireredtts (skip flash-attn) ---"
grep -v -i -E "^\s*(flash[-_]attn|deepspeed)" "$E/repos/FireRedTTS3/requirements.txt" > "$E/fireredtts_req_infer.txt"
echo "skipped lines:"; grep -i -E "^\s*(flash[-_]attn|deepspeed)" "$E/repos/FireRedTTS3/requirements.txt"
"$E/envs/fireredtts/bin/pip" install -r "$E/fireredtts_req_infer.txt" > "$LOG/fix_fireredtts.log" 2>&1
echo "fireredtts pip exit=$?"
tail -2 "$LOG/fix_fireredtts.log"

echo "=== IMPORT CHECK $(date +%H:%M:%S) ==="
for pair in "voxcpm:voxcpm" "indextts:indextts" "cosyvoice:cosyvoice" "qwen3tts:qwen_tts" "fireredtts:fireredtts3"; do
  n=${pair%%:*}; m=${pair##*:}
  if "$E/envs/$n/bin/python" -c "import $m" >/dev/null 2>&1; then echo "  $n OK"; else
    echo "  $n FAIL:"; "$E/envs/$n/bin/python" -c "import $m" 2>&1 | tail -2 | sed 's/^/      /'
  fi
done
echo "=== FIX ENVS DONE $(date +%H:%M:%S) ==="