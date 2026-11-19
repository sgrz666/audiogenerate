#!/usr/bin/env bash
# Round 3: finish the two remaining configs, strictly after P1 finishes so the GPU
# is never shared.
#   fireredtts3_instruct: UPSTREAM BUG. core.py:384 does `seg, seg_sr, _ =` but the
#     backend generate_tts (llm/fireredtts3_instruct.py:392) returns only
#     (gen_audio, gen_audio_sr). The third value was discarded anyway, so dropping
#     it from the unpack is behaviour-preserving. Patched + documented.
#   cosyvoice3: deps installed line-by-line by fix4 (grpcio pin blocked the rest).
set -u
E=/data/shangguanrz/projectsss/speakgenerate/experiments/model-test
BASE=$HOME/miniconda3/envs/deepgen/bin/python
LOG=$E/logs
CORE=$E/repos/FireRedTTS3/fireredtts3/core.py

echo "=== ROUND3 waiting for P1 $(date +%H:%M:%S) ==="
for i in $(seq 1 480); do
  if grep -q "P1 DRIVER DONE" "$LOG/p1_driver.log" 2>/dev/null; then echo "P1 finished"; break; fi
  sleep 30
done

echo "=== patch instruct unpack bug ==="
cp -n "$CORE" "$CORE.orig_unpack"
sed -i 's/seg, seg_sr, _ = super().generate_tts(/seg, seg_sr = super().generate_tts(/' "$CORE"
echo "remaining 3-value unpacks (must be 0):"
grep -c "seg, seg_sr, _ = super().generate_tts(" "$CORE" || true
grep -n "seg, seg_sr = super().generate_tts(" "$CORE"

echo "=== wait for cosyvoice deps (fix4) ==="
for i in $(seq 1 120); do
  if grep -q "FIX4 DONE" "$LOG/fix4_cosyvoice.log" 2>/dev/null; then break; fi
  sleep 15
done
tail -3 "$LOG/fix4_cosyvoice.log"

echo "=== P0 for remaining 2 configs $(date +%H:%M:%S) ==="
cd "$E/scripts" || exit 1
PYTHONPATH=$E/scripts $BASE mt_serial.py --phase P0 \
    --configs cosyvoice3,fireredtts3_instruct --timeout 3600 2>&1 | tee "$LOG/p0_round3.log"

echo "=== P0 METRICS ALL $(date +%H:%M:%S) ==="
PYTHONPATH=$E/scripts $E/envs/voxcpm/bin/python mt_metrics.py --phase P0 \
    --asr-dir "$E/asr/SenseVoiceSmall" 2>&1 | grep -vE "it/s|rtf_avg|load_data" | tail -10

echo "=== P1 for newly-passing configs $(date +%H:%M:%S) ==="
PASS=$(PYTHONPATH=$E/scripts $BASE - <<'PY'
import json, os, sys
E = "/data/shangguanrz/projectsss/speakgenerate/experiments/model-test"
sys.path.insert(0, os.path.join(E, "scripts"))
import mt_common as C
out = []
for cid in ["cosyvoice3", "fireredtts3_instruct"]:
    p = C.result_path(cid, "P0")
    if not os.path.isfile(p):
        continue
    best = {}
    for l in open(p, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        k = (r.get("text_id"), r.get("voice_id"))
        if r.get("status") == "ok":
            best[k] = "ok"
        elif k not in best:
            best[k] = r.get("status")
    if sum(1 for v in best.values() if v == "ok") == 30:
        out.append(cid)
print(",".join(out))
PY
)
echo "newly admitted: $PASS"
if [ -n "$PASS" ]; then
  PYTHONPATH=$E/scripts $BASE mt_serial.py --phase P1 --configs "$PASS" --timeout 7200 2>&1 | tee "$LOG/p1_round3.log"
  PYTHONPATH=$E/scripts $E/envs/voxcpm/bin/python mt_metrics.py --phase P1 \
      --asr-dir "$E/asr/SenseVoiceSmall" 2>&1 | grep -vE "it/s|rtf_avg|load_data" | tail -10
fi

echo "=== ROUND3 DONE $(date +%H:%M:%S) ==="