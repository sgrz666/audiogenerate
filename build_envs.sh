#!/usr/bin/env bash
# Sequentially build the remaining model envs. CPU/network only, no GPU use.
# Each env is isolated (--system-site-packages reuses torch 2.8 from deepgen).
set -u
E=/data/shangguanrz/projectsss/speakgenerate/experiments/model-test
BASE=$HOME/miniconda3/envs/deepgen/bin/python
LOG=$E/logs

build () {
  local name="$1"; local repo="$2"; local mod="$3"; shift 3
  local py="$E/envs/$name/bin/python"
  echo "=== [$name] start $(date +%H:%M:%S) ==="
  if [ -x "$py" ] && "$py" -c "import $mod" >/dev/null 2>&1; then
    echo "[$name] already importable, skip"; return 0
  fi
  if [ ! -x "$py" ]; then
    "$BASE" -m venv --system-site-packages "$E/envs/$name" || { echo "[$name] venv FAILED"; return 1; }
  fi
  "$E/envs/$name/bin/pip" install "$@" > "$LOG/install_$name.log" 2>&1
  local rc=$?
  echo "[$name] pip exit=$rc $(date +%H:%M:%S)"
  tail -2 "$LOG/install_$name.log"
  return $rc
}

# index-tts: editable install of the repo
build indextts index-tts indextts -e "$E/repos/index-tts"
# CosyVoice: requirements then the repo itself
build cosyvoice CosyVoice cosyvoice -r "$E/repos/CosyVoice/requirements.txt"
# Qwen3-TTS: repo editable (falls back to requirements if no pyproject)
if [ -f "$E/repos/Qwen3-TTS/pyproject.toml" ] || [ -f "$E/repos/Qwen3-TTS/setup.py" ]; then
  build qwen3tts Qwen3-TTS qwen3_tts -e "$E/repos/Qwen3-TTS"
else
  build qwen3tts Qwen3-TTS transformers -r "$E/repos/Qwen3-TTS/requirements.txt"
fi
# FireRedTTS3
build fireredtts FireRedTTS3 fireredtts3 -r "$E/repos/FireRedTTS3/requirements.txt"

echo "=== ALL ENV BUILDS DONE $(date +%H:%M:%S) ==="
for n in voxcpm indextts cosyvoice qwen3tts fireredtts; do
  if [ -x "$E/envs/$n/bin/python" ]; then echo "env $n present"; else echo "env $n MISSING"; fi
done