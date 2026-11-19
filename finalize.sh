#!/usr/bin/env bash
# After all generation finishes: recompute metrics, scores and the report for P0/P1.
set -u
E=/data/shangguanrz/projectsss/speakgenerate/experiments/model-test
BASE=$HOME/miniconda3/envs/deepgen/bin/python
LOG=$E/logs

echo "=== FINALIZE waiting for round3 $(date +%H:%M:%S) ==="
for i in $(seq 1 480); do
  if grep -q "ROUND3 DONE" "$LOG/round3_driver.log" 2>/dev/null; then echo "round3 finished"; break; fi
  sleep 30
done

cd "$E/scripts" || exit 1
for PH in P0 P1; do
  echo "=== $PH metrics $(date +%H:%M:%S) ==="
  PYTHONPATH=$E/scripts $E/envs/voxcpm/bin/python mt_metrics.py --phase $PH \
      --asr-dir "$E/asr/SenseVoiceSmall" 2>&1 | grep -vE "it/s|rtf_avg|load_data" | tail -8
  echo "=== $PH score ==="
  PYTHONPATH=$E/scripts $BASE mt_score.py --phase $PH 2>&1 | tail -8
done

echo "=== report $(date +%H:%M:%S) ==="
PYTHONPATH=$E/scripts $BASE mt_report.py 2>&1 | tail -3
echo "=== FINALIZE DONE $(date +%H:%M:%S) ==="