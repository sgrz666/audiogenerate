#!/usr/bin/env bash
# After P0/P1 finalize: run P2 (control, Track B) then P3 (stability) for the
# configs that passed P1, strictly serial, then recompute scores and the report.
set -u
E=/data/shangguanrz/projectsss/speakgenerate/experiments/model-test
BASE=$HOME/miniconda3/envs/deepgen/bin/python
LOG=$E/logs

echo "=== P2P3 waiting for finalize $(date +%H:%M:%S) ==="
for i in $(seq 1 480); do
  if grep -q "FINALIZE DONE" "$LOG/finalize.log" 2>/dev/null; then echo "finalize done"; break; fi
  sleep 30
done

# configs admitted by P1 (30/30 ok)
PASS=$(PYTHONPATH=$E/scripts $BASE - <<'PY'
import json, os, sys
E="/data/shangguanrz/projectsss/speakgenerate/experiments/model-test"
sys.path.insert(0, os.path.join(E,"scripts"))
import mt_common as C
out=[]
for c in C.CONFIGS:
    p=C.result_path(c["config_id"],"P1")
    if not os.path.isfile(p): continue
    best={}
    for l in open(p,encoding="utf-8"):
        if not l.strip(): continue
        r=json.loads(l); k=(r.get("text_id"),r.get("voice_id"))
        if r.get("status")=="ok": best[k]="ok"
        elif k not in best: best[k]=r.get("status")
    if sum(1 for v in best.values() if v=="ok")==180: out.append(c["config_id"])
print(",".join(out))
PY
)
echo "P1-passed configs: $PASS"
if [ -z "$PASS" ]; then echo "no config passed P1; abort"; echo "=== P2P3 DONE ==="; exit 0; fi

cd "$E/scripts" || exit 1
IFS=',' read -ra CFGS <<< "$PASS"

echo "=== P2 SERIAL $(date +%H:%M:%S) ==="
for c in "${CFGS[@]}"; do
  envname=$($BASE -c "import sys;sys.path.insert(0,'$E/scripts');import mt_common as C;print(C.cfg_by_id('$c')['env'])")
  echo "--- P2 $c (env $envname) ---"
  PYTHONPATH=$E/scripts "$E/envs/$envname/bin/python" mt_run_p2.py --config "$c" \
      > "$LOG/P2_$c.log" 2>&1
  echo "P2 $c exit=$? $(date +%H:%M:%S)"
  tail -2 "$LOG/P2_$c.log"
  sleep 3
done

echo "=== P3 SERIAL $(date +%H:%M:%S) ==="
for c in "${CFGS[@]}"; do
  envname=$($BASE -c "import sys;sys.path.insert(0,'$E/scripts');import mt_common as C;print(C.cfg_by_id('$c')['env'])")
  echo "--- P3 $c ---"
  PYTHONPATH=$E/scripts "$E/envs/$envname/bin/python" mt_run_p3.py --config "$c" \
      --reps 3 --texts 6 > "$LOG/P3_$c.log" 2>&1
  echo "P3 $c exit=$? $(date +%H:%M:%S)"
  tail -2 "$LOG/P3_$c.log"
  sleep 3
done

echo "=== blind package $(date +%H:%M:%S) ==="
PYTHONPATH=$E/scripts $BASE mt_blind.py --phase P1 2>&1 | tail -3

echo "=== P2P3 DONE $(date +%H:%M:%S) ==="