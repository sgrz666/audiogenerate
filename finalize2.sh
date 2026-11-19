#!/usr/bin/env bash
# After P2/P3: compute Track B scores and regenerate the report.
set -u
E=/data/shangguanrz/projectsss/speakgenerate/experiments/model-test
BASE=$HOME/miniconda3/envs/deepgen/bin/python
LOG=$E/logs
echo "=== FINALIZE2 waiting for P2P3 $(date +%H:%M:%S) ==="
for i in $(seq 1 720); do
  if grep -q "P2P3 DONE" "$LOG/p2p3_driver.log" 2>/dev/null; then echo "p2p3 done"; break; fi
  sleep 30
done
cd "$E/scripts" || exit 1
echo "=== trackB score $(date +%H:%M:%S) ==="
PYTHONPATH=$E/scripts $BASE mt_score_b.py 2>&1 | tail -12
echo "=== report $(date +%H:%M:%S) ==="
PYTHONPATH=$E/scripts $BASE mt_report.py 2>&1 | tail -3
echo "=== FINALIZE2 DONE $(date +%H:%M:%S) ==="