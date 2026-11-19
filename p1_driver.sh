#!/usr/bin/env bash
# P1 driver. Per the plan, P1 runs only after P0 passes.
# Gate: a config is admitted to P1 only if its P0 has 30/30 ok. Configs that
# failed P0 are left as not_tested for P1 rather than burning compute.
set -u
E=/data/shangguanrz/projectsss/speakgenerate/experiments/model-test
BASE=$HOME/miniconda3/envs/deepgen/bin/python
LOG=$E/logs

echo "=== P1 DRIVER waiting for round2 $(date +%H:%M:%S) ==="
for i in $(seq 1 240); do
  if grep -q "ROUND2 DONE" "$LOG/p0_round2_driver.log" 2>/dev/null; then echo "round2 finished"; break; fi
  sleep 30
done

echo "=== P0 GATE $(date +%H:%M:%S) ==="
PASS=$(PYTHONPATH=$E/scripts $BASE - <<'PY'
import json, os
E = "/data/shangguanrz/projectsss/speakgenerate/experiments/model-test"
import sys
sys.path.insert(0, os.path.join(E, "scripts"))
import mt_common as C
ok_cfgs = []
for c in C.CONFIGS:
    p = C.result_path(c["config_id"], "P0")
    if not os.path.isfile(p):
        print(f"  {c['config_id']:22s} NO_P0_RESULT", file=sys.stderr); continue
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    # dedupe by (text_id, voice_id): keep ok if any attempt succeeded
    best = {}
    for r in rows:
        k = (r.get("text_id"), r.get("voice_id"))
        if r.get("status") == "ok":
            best[k] = "ok"
        elif k not in best:
            best[k] = r.get("status")
    n_ok = sum(1 for v in best.values() if v == "ok")
    admit = n_ok == 30
    print(f"  {c['config_id']:22s} P0 ok={n_ok}/30 -> {'ADMIT' if admit else 'HOLD'}", file=sys.stderr)
    if admit:
        ok_cfgs.append(c["config_id"])
print(",".join(ok_cfgs))
PY
)
echo "admitted configs: $PASS"
if [ -z "$PASS" ]; then
  echo "no config passed P0; P1 not started"; echo "=== P1 DRIVER DONE ==="; exit 0
fi

echo "=== P1 SERIAL $(date +%H:%M:%S) ==="
cd "$E/scripts" || exit 1
PYTHONPATH=$E/scripts $BASE mt_serial.py --phase P1 --configs "$PASS" --timeout 7200 2>&1 | tee "$LOG/p1_serial.log"

echo "=== P1 METRICS $(date +%H:%M:%S) ==="
PYTHONPATH=$E/scripts $E/envs/voxcpm/bin/python mt_metrics.py --phase P1 \
    --asr-dir "$E/asr/SenseVoiceSmall" 2>&1 | grep -vE "it/s|rtf_avg|load_data" | tail -12

echo "=== P1 DRIVER DONE $(date +%H:%M:%S) ==="