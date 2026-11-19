#!/usr/bin/env bash
# Wait for fix_envs.sh, then re-run P0 for the configs that failed on env issues,
# then recompute metrics for all 6. Strictly serial (mt_serial spawns one process
# per config and waits for exit).
set -u
E=/data/shangguanrz/projectsss/speakgenerate/experiments/model-test
BASE=$HOME/miniconda3/envs/deepgen/bin/python
LOG=$E/logs

echo "=== P0 RETRY waiting for fix_envs $(date +%H:%M:%S) ==="
for i in $(seq 1 240); do
  if grep -q "FIX ENVS DONE" "$LOG/fix_envs.log" 2>/dev/null; then echo "fix_envs finished"; break; fi
  sleep 30
done

echo "=== import status ==="
grep -A8 "IMPORT CHECK" "$LOG/fix_envs.log" | tail -8

echo "=== P0 RETRY SERIAL $(date +%H:%M:%S) ==="
cd "$E/scripts" || exit 1
PYTHONPATH=$E/scripts $BASE mt_serial.py --phase P0 \
    --configs indextts25,cosyvoice3,fireredtts3_base,fireredtts3_instruct \
    --timeout 3600 2>&1 | tee "$LOG/p0_retry.log"

echo "=== P0 METRICS ALL $(date +%H:%M:%S) ==="
PYTHONPATH=$E/scripts $E/envs/voxcpm/bin/python mt_metrics.py --phase P0 \
    --asr-dir "$E/asr/SenseVoiceSmall" 2>&1 | grep -vE "it/s|rtf_avg|load_data" | tail -12

echo "=== P0 RETRY DONE $(date +%H:%M:%S) ==="