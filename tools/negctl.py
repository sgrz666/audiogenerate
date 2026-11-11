#!/usr/bin/env python3
"""Reverse probe for the verify gate.

Builds a symlink farm of an already-downloaded repo, injects ONE defect at a
time, and asserts the gate reports exactly that defect and exits non-zero.
Covers both hash paths (LFS sha256 and plain git-blob sha1) plus a missing file.
Also runs a positive control on the untouched repo.

Never modifies the real model files: defects live in a scratch farm only.
"""
import json
import os
import shutil
import subprocess
import sys

TOOLS = "/data/shangguanrz/projectsss/speakgenerate/tools"
ROOT = "/data/shangguanrz/projectsss/speakgenerate/model"
SCRATCH = "/data/shangguanrz/projectsss/speakgenerate/.tmp/verify_negctl"
NAME = "IndexTTS-2.5"
PY = sys.executable


def log(m):
    print(m, flush=True)


def build_farm(man, dest):
    if os.path.exists(dest):
        shutil.rmtree(dest)
    for f in man["files"]:
        src = os.path.join(ROOT, NAME, f["path"])
        dst = os.path.join(dest, f["path"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.symlink(src, dst)


def run_verify(root):
    p = subprocess.run(
        [PY, os.path.join(TOOLS, "fetch_models.py"), "verify",
         "--repos", NAME, "--root", root, "--workers", "8"],
        capture_output=True, text=True)
    rep = os.path.join(TOOLS, "manifests", "verify_report.json")
    data = json.load(open(rep, encoding="utf-8"))[NAME]
    return p.returncode, data


def corrupt(path):
    """Replace a symlink with a real copy whose last byte is flipped."""
    real = os.path.realpath(path)
    with open(real, "rb") as f:
        blob = bytearray(f.read())
    blob[-1] ^= 0xFF
    os.remove(path)
    with open(path, "wb") as f:
        f.write(blob)


def main():
    man = json.load(open(os.path.join(TOOLS, "manifests", NAME + ".json"),
                         encoding="utf-8"))
    files = man["files"]
    lfs_small = min((f for f in files if f.get("lfs_sha256")),
                    key=lambda f: f["size"])
    plain_small = min((f for f in files if not f.get("lfs_sha256")),
                      key=lambda f: f["size"])
    log(f"LFS probe target   : {lfs_small['path']} ({lfs_small['size']} B)")
    log(f"plain probe target : {plain_small['path']} ({plain_small['size']} B)")

    os.makedirs(SCRATCH, exist_ok=True)
    dest = os.path.join(SCRATCH, NAME)
    failures = []

    # positive control: untouched farm must pass at 100% coverage
    build_farm(man, dest)
    rc, d = run_verify(SCRATCH)
    log(f"[positive] rc={rc} coverage={d['coverage']*100:.1f}% "
        f"ok={d['n_ok']}/{d['n_files']}")
    if rc != 0 or d["coverage"] != 1.0 or d["n_files"] == 0:
        failures.append("positive control did not pass cleanly")

    cases = [
        ("LFS sha256 corruption", lfs_small["path"], "HASH_MISMATCH", "sha256"),
        ("plain sha1 corruption", plain_small["path"], "HASH_MISMATCH", "sha1"),
        ("missing file", plain_small["path"], "MISSING", None),
    ]
    for label, rel, want_status, want_kind in cases:
        build_farm(man, dest)
        target = os.path.join(dest, rel)
        if want_status == "MISSING":
            os.remove(target)
        else:
            corrupt(target)
        rc, d = run_verify(SCRATCH)
        bad = d["bad"]
        ok = (rc != 0 and len(bad) == 1 and bad[0]["status"] == want_status
              and bad[0]["path"] == rel and d["coverage"] < 1.0
              and (want_kind is None or bad[0].get("hash_kind") == want_kind))
        log(f"[{label}] rc={rc} bad={len(bad)} "
            f"status={bad[0]['status'] if bad else None} "
            f"kind={bad[0].get('hash_kind') if bad else None} "
            f"coverage={d['coverage']*100:.1f}% -> {'CAUGHT' if ok else 'MISSED'}")
        if not ok:
            failures.append(label)

    shutil.rmtree(SCRATCH, ignore_errors=True)
    log("scratch removed")
    if failures:
        log("NEGCTL FAIL: " + "; ".join(failures))
        return 1
    log("NEGCTL PASS: gate catches sha256, sha1 and missing-file defects")
    return 0


if __name__ == "__main__":
    sys.exit(main())