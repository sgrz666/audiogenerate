#!/usr/bin/env bash
# Run fix2, then re-run P0 for the 4 remaining configs serially, then metrics.
set -u
E=/data/shangguanrz/projectsss/speakgenerate/experiments/model-test
BASE=$HOME/miniconda3/envs/deepgen/bin/python
LOG=$E/logs

echo "=== ROUND2 start $(date +%H:%M:%S) ==="
bash "$E/scripts/fix2_envs.sh" 2>&1 | tee "$LOG/fix2_run.log"

echo "=== P0 SERIAL (4 configs) $(date +%H:%M:%S) ==="
cd "$E/scripts" || exit 1
PYTHONPATH=$E/scripts $BASE mt_serial.py --phase P0 \
    --configs indextts25,cosyvoice3,fireredtts3_base,fireredtts3_instruct \
    --timeout 3600 2>&1 | tee "$LOG/p0_round2.log"

echo "=== P0 METRICS ALL $(date +%H:%M:%S) ==="
PYTHONPATH=$E/scripts $E/envs/voxcpm/bin/python mt_metrics.py --phase P0 \
    --asr-dir "$E/asr/SenseVoiceSmall" 2>&1 | grep -vE "it/s|rtf_avg|load_data" | tail -12

echo "=== ROUND2 DONE $(date +%H:%M:%S) ==="