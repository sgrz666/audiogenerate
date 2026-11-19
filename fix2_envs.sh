#!/usr/bin/env bash
# Second-round fixes, each tied to an observed P0 error:
#   indextts25 : "operator torchvision::nms does not exist"
#                -> venv has torch 2.8.0 but torchvision 0.22.1 (pairs with 2.7.1).
#                   Install torchvision 0.23.0 to match torch 2.8.
#   cosyvoice3 : "No module named 'hyperpyyaml'"
#                -> HyperPyYAML==1.2.3 is in requirements but did not get installed.
#   fireredtts3: "FlashAttention2 has been toggled on ... not installed"
#                -> flash_attention_2 is HARDCODED in redae.py:37,122 and
#                   llm/fireredtts3_base.py:49. flash-attn needs nvcc (absent).
#                   Patch those literals to "sdpa". This is an attention-backend
#                   change, recorded in the results so it is not a silent edit.
set -u
E=/data/shangguanrz/projectsss/speakgenerate/experiments/model-test
LOG=$E/logs
R=$E/repos/FireRedTTS3/fireredtts3

echo "=== FIX2 start $(date +%H:%M:%S) ==="

echo "--- indextts: torchvision 0.23.0 ---"
"$E/envs/indextts/bin/pip" install "torchvision==0.23.0" > "$LOG/fix2_indextts.log" 2>&1
echo "exit=$?"; tail -2 "$LOG/fix2_indextts.log"

echo "--- cosyvoice: HyperPyYAML ---"
"$E/envs/cosyvoice/bin/pip" install "HyperPyYAML==1.2.3" > "$LOG/fix2_cosyvoice.log" 2>&1
echo "exit=$?"; tail -2 "$LOG/fix2_cosyvoice.log"

echo "--- fireredtts: patch flash_attention_2 -> sdpa ---"
for f in "$R/redae/redae.py" "$R/llm/fireredtts3_base.py" "$R/llm/fireredtts3_instruct.py"; do
  [ -f "$f" ] || continue
  cp -n "$f" "$f.orig_flash"
  sed -i "s/flash_attention_2/sdpa/g" "$f"
done
echo "remaining flash_attention_2 occurrences (must be 0):"
grep -rn "flash_attention_2" "$R" --include=*.py | wc -l
echo "sdpa occurrences now:"
grep -rn "attn_implementation.*sdpa\|\"attn_implementation\": \"sdpa\"" "$R" --include=*.py | head -5
# also check the instruct backend file name actually present
ls "$R/llm/"

echo "=== FIX2 DONE $(date +%H:%M:%S) ==="