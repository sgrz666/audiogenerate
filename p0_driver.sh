#!/usr/bin/env bash
# Wait for env builds, then run P0 for all 6 configs strictly serially, then metrics.
# Separate file on purpose: build_envs.sh is still being read by bash, so it must
# not be edited in place.
set -u
E=/data/shangguanrz/projectsss/speakgenerate/experiments/model-test
BASE=$HOME/miniconda3/envs/deepgen/bin/python
LOG=$E/logs

echo "=== P0 DRIVER waiting for env builds $(date +%H:%M:%S) ==="
for i in $(seq 1 240); do   # up to 120 min
  if grep -q "ALL ENV BUILDS DONE" "$LOG/build_envs.log" 2>/dev/null; then
    echo "env builds finished"; break
  fi
  sleep 30
done

echo "=== env import status $(date +%H:%M:%S) ==="
for pair in "voxcpm:voxcpm" "indextts:indextts" "cosyvoice:cosyvoice" "qwen3tts:qwen_tts" "fireredtts:fireredtts3"; do
  n=${pair%%:*}; m=${pair##*:}
  if $E/envs/$n/bin/python -c "import $m" >/dev/null 2>&1; then
    echo "  $n OK"
  else
    echo "  $n IMPORT_FAIL"
  fi
done

echo "=== P0 SERIAL RUN $(date +%H:%M:%S) ==="
cd "$E/scripts" || exit 1
PYTHONPATH=$E/scripts $BASE mt_serial.py --phase P0 --timeout 3600 2>&1 | tee "$LOG/p0_serial.log"

echo "=== P0 METRICS $(date +%H:%M:%S) ==="
PYTHONPATH=$E/scripts $E/envs/voxcpm/bin/python mt_metrics.py --phase P0 \
    --asr-dir "$E/asr/SenseVoiceSmall" 2>&1 | grep -vE "it/s|rtf_avg|load_data" | tail -20

echo "=== P0 DRIVER DONE $(date +%H:%M:%S) ==="