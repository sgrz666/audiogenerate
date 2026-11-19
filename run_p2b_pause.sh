#!/usr/bin/env bash
# Run the P2b pause set (405 items, 5 configs) strictly serially, one config
# per process.  Does not touch the v1 P2 results.
set -u
E=/data/shangguanrz/projectsss/speakgenerate/experiments/model-test
PY=$HOME/miniconda3/envs/deepgen/bin/python
LOG=$E/logs
CFGS="voxcpm2 indextts25 qwen3tts_base fireredtts3_base fireredtts3_instruct"

echo "=== P2b PAUSE START $(date +%H:%M:%S) ==="
for c in $CFGS; do
  envname=$($PY -c "import sys;sys.path.insert(0,'$E/scripts');import mt_common as C;print(C.cfg_by_id('$c')['env'])")
  echo "--- $c (env $envname) $(date +%H:%M:%S) ---"
  PYTHONPATH=$E/scripts "$E/envs/$envname/bin/python" "$E/scripts/mt_run_p2.py" \
      --config "$c" --texts-file texts/p2b.jsonl --tag P2b --sub pause \
      > "$LOG/P2b_pause_$c.log" 2>&1
  echo "P2b pause $c exit=$? $(date +%H:%M:%S)"
  tail -2 "$LOG/P2b_pause_$c.log"
  sleep 3
done
echo "=== P2b PAUSE DONE $(date +%H:%M:%S) ==="
