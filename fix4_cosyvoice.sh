#!/usr/bin/env bash
# CosyVoice deps: the pinned grpcio==1.57.0 has no py3.12 wheel and aborts the
# whole `pip install -r`, so everything after it (incl. openai-whisper) never
# installs. Install line-by-line instead, skipping known build-blockers, then
# add the unpinned replacements. CPU/network only.
set -u
E=/data/shangguanrz/projectsss/speakgenerate/experiments/model-test
PIP=$E/envs/cosyvoice/bin/pip
LOG=$E/logs/fix4_cosyvoice.log
: > "$LOG"

echo "=== per-line install $(date +%H:%M:%S) ===" | tee -a "$LOG"
while IFS= read -r line; do
  # strip comments/whitespace; skip blanks and known build-blockers
  pkg=$(echo "$line" | sed 's/#.*//' | tr -d '[:space:]')
  [ -z "$pkg" ] && continue
  case "$pkg" in
    deepspeed*|flash-attn*|flash_attn*|grpcio*) echo "SKIP $pkg" | tee -a "$LOG"; continue;;
  esac
  if $PIP install "$pkg" >> "$LOG" 2>&1; then
    echo "OK   $pkg" | tee -a "$LOG"
  else
    echo "FAIL $pkg" | tee -a "$LOG"
  fi
done < "$E/repos/CosyVoice/requirements.txt"

echo "=== unpinned replacements $(date +%H:%M:%S) ===" | tee -a "$LOG"
$PIP install grpcio openai-whisper >> "$LOG" 2>&1 && echo "OK grpcio+whisper" | tee -a "$LOG" || echo "FAIL grpcio+whisper" | tee -a "$LOG"

echo "=== import check ===" | tee -a "$LOG"
PYTHONPATH=$E/repos/CosyVoice $E/envs/cosyvoice/bin/python -c \
  "from cosyvoice.cli.cosyvoice import CosyVoice3; print('CosyVoice3 IMPORT OK')" 2>&1 | tail -3 | tee -a "$LOG"
echo "=== FIX4 DONE $(date +%H:%M:%S) ===" | tee -a "$LOG"