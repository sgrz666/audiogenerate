#!/usr/bin/env bash
# CosyVoice3 was blocked by an empty Matcha-TTS git submodule (now cloned).
# Run its full P0/P1/P2/P3 AFTER the main pipeline finishes, to keep the GPU serial.
set -u
E=/data/shangguanrz/projectsss/speakgenerate/experiments/model-test
BASE=$HOME/miniconda3/envs/deepgen/bin/python
LOG=$E/logs
PY=$E/envs/cosyvoice/bin/python

echo "=== COSYRETRY waiting for FINALIZE2 $(date +%H:%M:%S) ==="
for i in $(seq 1 720); do
  if grep -q "FINALIZE2 DONE" "$LOG/finalize2.log" 2>/dev/null; then echo "main pipeline done"; break; fi
  sleep 30
done

cd "$E/scripts" || exit 1
export PYTHONPATH=$E/scripts
for PH in P0 P1; do
  echo "=== cosyvoice3 $PH $(date +%H:%M:%S) ==="
  $PY mt_run.py --config cosyvoice3 --phase $PH > "$LOG/cosyretry_$PH.log" 2>&1
  echo "cosyvoice3 $PH exit=$? $(date +%H:%M:%S)"; tail -2 "$LOG/cosyretry_$PH.log"
  sleep 3
done
# P2/P3 only if P1 passed
PASS=$($BASE - <<'PY'
import json,os,sys
E="/data/shangguanrz/projectsss/speakgenerate/experiments/model-test"
p=os.path.join(E,"results","P1","cosyvoice3.jsonl")
best={}
if os.path.isfile(p):
    for l in open(p,encoding="utf-8"):
        if not l.strip():continue
        r=json.loads(l);k=(r.get("text_id"),r.get("voice_id"))
        if r.get("status")=="ok":best[k]="ok"
        elif k not in best:best[k]=r.get("status")
print("yes" if sum(1 for v in best.values() if v=="ok")==180 else "no")
PY
)
echo "cosyvoice3 P1 passed: $PASS"
if [ "$PASS" = "yes" ]; then
  $PY mt_run_p2.py --config cosyvoice3 > "$LOG/cosyretry_P2.log" 2>&1; echo "P2 exit=$?"
  sleep 3
  $PY mt_run_p3.py --config cosyvoice3 --reps 3 --texts 6 > "$LOG/cosyretry_P3.log" 2>&1; echo "P3 exit=$?"
fi

echo "=== recompute metrics/scores/report $(date +%H:%M:%S) ==="
for PH in P0 P1; do
  $E/envs/voxcpm/bin/python mt_metrics.py --phase $PH --asr-dir "$E/asr/SenseVoiceSmall" 2>&1 | grep -vE "it/s|rtf_avg|load_data" | tail -7
  $BASE mt_score.py --phase $PH 2>&1 | tail -7
done
$BASE mt_score_b.py 2>&1 | tail -7
$BASE mt_blind.py --phase P1 2>&1 | tail -2
$BASE mt_report.py 2>&1 | tail -2
echo "=== COSYRETRY DONE $(date +%H:%M:%S) ==="