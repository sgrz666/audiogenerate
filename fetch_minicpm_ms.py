#!/usr/bin/env python3
"""Fetch MiniCPM-o-4_5 from ModelScope into a fresh dir and verify against the
HF manifest. Resumable: aria2 control files + size checks; safe to rerun.

Usage (remote):
  setsid nohup python3 fetch_minicpm_ms.py > logs/fetch_minicpm_ms.log 2>&1 &
"""
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = "/data/shangguanrz/projectsss/speakgenerate"
E = os.path.join(ROOT, "experiments", "model-test")
MANIFEST = os.path.join(ROOT, "tools", "manifests", "MiniCPM-o-4_5.json")
DEST = os.path.join(ROOT, "model", "MiniCPM-o-4_5_ms")
REPO = "OpenBMB/MiniCPM-o-4_5"
BASE = f"https://modelscope.cn/models/{REPO}/resolve/master"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    files = manifest["files"]
    os.makedirs(DEST, exist_ok=True)
    todo = []
    for f in files:
        out = os.path.join(DEST, f["path"])
        if os.path.isfile(out) and os.path.getsize(out) == f["size"]:
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        todo.append(f)
    log(f"manifest files={len(files)} todo={len(todo)} dest={DEST}")
    if not todo:
        log("nothing to download")
    else:
        aria_in = os.path.join(E, "logs", "minicpm_aria.txt")
        os.makedirs(os.path.dirname(aria_in), exist_ok=True)
        with open(aria_in, "w", encoding="utf-8") as fo:
            for f in todo:
                out = os.path.join(DEST, f["path"])
                fo.write(f"{BASE}/{f['path']}\n")
                fo.write(f"  dir={os.path.dirname(out)}\n")
                fo.write(f"  out={os.path.basename(out)}\n")
                fo.write("  continue=true\n")
                fo.write(f"  max-tries=0\n")
                fo.write(f"  retry-wait=10\n")
        cmd = ["aria2c", "-i", aria_in, "-j", "4", "-x", "8", "-s", "8",
               "-k", "1M", "--file-allocation=none",
               "--console-log-level=warn", "--summary-interval=60",
               "--max-tries=0", "--retry-wait=10", "--auto-file-renaming=false"]
        log("aria2 " + " ".join(cmd))
        r = subprocess.run(cmd)
        log(f"aria2 exit={r.returncode}")

    # size + sha256 verification
    report = {"checked": 0, "size_bad": [], "sha_bad": [], "missing": []}
    for f in files:
        out = os.path.join(DEST, f["path"])
        if not os.path.isfile(out):
            report["missing"].append(f["path"])
            continue
        if os.path.getsize(out) != f["size"]:
            report["size_bad"].append({"path": f["path"],
                                       "want": f["size"],
                                       "got": os.path.getsize(out)})
            continue
        if f.get("lfs_sha256"):
            got = sha256(out)
            if got != f["lfs_sha256"]:
                report["sha_bad"].append({"path": f["path"], "want": f["lfs_sha256"],
                                          "got": got})
                continue
        report["checked"] += 1
    report["total"] = len(files)
    report["ok"] = (not report["missing"] and not report["size_bad"]
                    and not report["sha_bad"])
    out = os.path.join(E, "results", "v1", "minicpm_ms_verify.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("verify " + json.dumps({k: v for k, v in report.items() if k != "sha_bad"},
                               ensure_ascii=False))
    if report["sha_bad"]:
        log("sha_bad " + json.dumps(report["sha_bad"], ensure_ascii=False)[:2000])
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
