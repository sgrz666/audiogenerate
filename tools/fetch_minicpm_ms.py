#!/usr/bin/env python3
"""Download openbmb/MiniCPM-o-4_5 from ModelScope (fast in China) into model/.

ModelScope was measured ~4-10x faster than huggingface.co / hf-mirror for this
repo. Files are identical to the HF release, so the HF manifest sha256 is used
afterwards to verify integrity (fetch_models.py verify).
"""
import os
import sys
import time
import threading

DEST = "/data/shangguanrz/projectsss/speakgenerate/model/MiniCPM-o-4_5"
MODEL_ID = "OpenBMB/MiniCPM-o-4_5"
EXPECT_BYTES = int(18.67 * 1024**3)


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def dir_bytes(path):
    t = 0
    for base, _, names in os.walk(path):
        for n in names:
            try:
                t += os.path.getsize(os.path.join(base, n))
            except OSError:
                pass
    return t


def main():
    from modelscope import snapshot_download
    os.makedirs(DEST, exist_ok=True)
    stop = threading.Event()

    def mon():
        t0 = time.time(); b0 = dir_bytes(DEST)
        while not stop.wait(30):
            b = dir_bytes(DEST)
            sp = (b - b0) / max(time.time() - t0, 1e-6)
            eta = (EXPECT_BYTES - b) / sp / 60 if sp > 1 else -1
            log(f"  {b/1024**3:.2f}/18.67 GiB ({b/EXPECT_BYTES*100:.1f}%) "
                f"{sp/1024**2:.1f} MiB/s ETA {eta:.1f} min")

    th = threading.Thread(target=mon, daemon=True); th.start()
    log(f"[start] ModelScope {MODEL_ID} -> {DEST}")
    for attempt in range(1, 7):
        try:
            snapshot_download(MODEL_ID, local_dir=DEST)
            log("[done] snapshot_download returned")
            break
        except Exception as e:
            log(f"  attempt {attempt} failed: {type(e).__name__}: {str(e)[:250]}")
            time.sleep(min(15 * attempt, 60))
    stop.set(); th.join(timeout=5)
    got = dir_bytes(DEST)
    log(f"[final] {got/1024**3:.2f} GiB on disk (expect ~18.67)")
    return 0


if __name__ == "__main__":
    sys.exit(main())