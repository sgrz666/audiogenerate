"""Shared utilities for the 5-model / 6-config zero-shot TTS comparison.

Design constraints from the task owner: strictly serial execution, and watch both
host RAM and GPU VRAM. Therefore each config runs in its OWN process; process exit
is the cleanup guarantee. Within a process we also del + empty_cache + gc.

VRAM note: this host's NVML is broken (driver 580.173.02 vs lib 580.178.04) so
nvidia-smi cannot be used. Peak VRAM is measured with torch.cuda.max_memory_allocated,
which works because the CUDA driver API itself is functional.
"""
import gc
import json
import os
import re
import time

import numpy as np

E = "/data/shangguanrz/projectsss/speakgenerate/experiments/model-test"
MODEL = "/data/shangguanrz/projectsss/speakgenerate/model"

CONFIGS = [
    {"config_id": "voxcpm2", "family": "VoxCPM2", "variant": "base",
     "env": "voxcpm", "model_dir": MODEL + "/VoxCPM2"},
    {"config_id": "indextts25", "family": "IndexTTS-2.5", "variant": "base",
     "env": "indextts", "model_dir": MODEL + "/IndexTTS-2.5"},
    {"config_id": "cosyvoice3", "family": "CosyVoice3-0.5B", "variant": "base",
     "env": "cosyvoice", "model_dir": MODEL + "/Fun-CosyVoice3-0.5B-2512"},
    {"config_id": "qwen3tts_base", "family": "Qwen3-TTS-1.7B", "variant": "Base",
     "env": "qwen3tts", "model_dir": MODEL + "/Qwen3-TTS-12Hz-1.7B-Base"},
    {"config_id": "fireredtts3_base", "family": "FireRedTTS3", "variant": "Base",
     "env": "fireredtts", "model_dir": MODEL + "/FireRedTTS3"},
    {"config_id": "fireredtts3_instruct", "family": "FireRedTTS3", "variant": "Instruct",
     "env": "fireredtts", "model_dir": MODEL + "/FireRedTTS3"},
]
VOICES = ["MALE", "FEMALE", "NARRATOR"]


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def cfg_by_id(cid):
    for c in CONFIGS:
        if c["config_id"] == cid:
            return c
    raise SystemExit(f"unknown config {cid}")


def load_texts(phase):
    p = os.path.join(E, "texts", "all.jsonl")
    out = []
    with open(p, encoding="utf-8") as f:
        for ln in f:
            d = json.loads(ln)
            if d["phase"] == phase:
                out.append(d)
    return out


def load_refs():
    with open(os.path.join(E, "refs", "refs.json"), encoding="utf-8") as f:
        return {r["voice_id"]: r for r in json.load(f)}


# ---------- wav io (scipy fallback so we do not depend on soundfile) ----------
def write_wav(path, wav, sr):
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    try:
        import soundfile as sf
        sf.write(path, wav, sr, subtype="PCM_16")
    except ImportError:
        from scipy.io import wavfile
        wavfile.write(path, sr, (wav * 32767).astype(np.int16))
    return path


def read_wav(path):
    try:
        import soundfile as sf
        w, sr = sf.read(path, dtype="float32", always_2d=False)
    except ImportError:
        from scipy.io import wavfile
        sr, w = wavfile.read(path)
        if w.dtype == np.int16:
            w = w.astype(np.float32) / 32768.0
        elif w.dtype == np.int32:
            w = w.astype(np.float32) / 2147483648.0
        else:
            w = w.astype(np.float32)
    if w.ndim > 1:
        w = w.mean(axis=1)
    return w.astype(np.float32), int(sr)


def speech_span(wav, sr, frame_ms=20, rel_thresh=0.02, abs_floor=1e-4):
    """Energy-based leading/trailing silence trim.

    Returns (start_s, end_s) of the voiced span. Internal pauses are KEPT, matching
    the task book's speaking-rate definition (exclude head/tail silence only).
    Method is recorded in every result row so the measurement basis is auditable.
    """
    n = max(1, int(sr * frame_ms / 1000))
    if wav.size < n:
        return 0.0, float(wav.size) / sr
    nf = wav.size // n
    frames = wav[: nf * n].reshape(nf, n)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    thr = max(float(rms.max()) * rel_thresh, abs_floor)
    idx = np.where(rms > thr)[0]
    if idx.size == 0:
        return 0.0, float(wav.size) / sr
    return float(idx[0] * n) / sr, float((idx[-1] + 1) * n) / sr


def han_count(s):
    return len(re.findall(r"[\u4e00-\u9fff]", s))


def norm_chars(s):
    """Characters counted for speaking rate: Han + digits + latin letters."""
    return len(re.findall(r"[\u4e00-\u9fff0-9A-Za-z]", s))


# ---------- resource helpers ----------
def rss_gib():
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for ln in f:
                if ln.startswith("VmRSS:"):
                    return round(int(ln.split()[1]) / 1024 ** 2, 3)
    except OSError:
        pass
    return None


def vram_reset():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def vram_peak_gib():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
    except Exception:
        pass
    return None


def vram_reserved_gib():
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.memory_reserved() / 1024 ** 3, 3)
    except Exception:
        pass
    return None


def release(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def out_path(config_id, phase, voice, text_id):
    d = os.path.join(E, "audio", phase, config_id, voice)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, text_id + ".wav")


def result_path(config_id, phase):
    d = os.path.join(E, "results", phase)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, config_id + ".jsonl")