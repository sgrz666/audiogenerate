#!/usr/bin/env python3
"""Pre-fetch IndexTTS-2.5 auxiliary models into the exact paths the code expects.

Spec taken from indextts/utils/model_download.py (ensure_models_available) and
indextts/infer_v2_5.py, not guessed:
  hf_cache/w2v-bert-2.0/                     <- facebook/w2v-bert-2.0 (subset)
  hf_cache/semantic_codec_model.safetensors  <- amphion/MaskGCT semantic_codec/model.safetensors
  hf_cache/campplus_cn_common.bin            <- funasr/campplus campplus_cn_common.bin
  hf_cache/bigvgan/{config.json,bigvgan_generator.pt}
                                             <- nvidia/bigvgan_v2_22khz_80band_256x

w2v-bert-2.0 ships the same weights in several formats; we take json/safetensors/
txt/model only, skipping tf/flax/bin duplicates. Everything fetched is hash-checked
against the HF API manifest.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import fetch_models as fm

DEST = os.path.join(fm.ROOT, "IndexTTS-2.5", "hf_cache")
MANPATH = os.path.join(fm.MANIFEST_DIR, "IndexTTS-2.5-aux.json")
W2V_SUFFIXES = (".json", ".safetensors", ".txt", ".model")

# (repo, local_relpath, remote_path or None for whole-repo subset)
JOBS = [
    ("facebook/w2v-bert-2.0", "w2v-bert-2.0", None),
    ("amphion/MaskGCT", "semantic_codec_model.safetensors",
     "semantic_codec/model.safetensors"),
    ("funasr/campplus", "campplus_cn_common.bin", "campplus_cn_common.bin"),
    ("nvidia/bigvgan_v2_22khz_80band_256x", "bigvgan/config.json", "config.json"),
    ("nvidia/bigvgan_v2_22khz_80band_256x", "bigvgan/bigvgan_generator.pt",
     "bigvgan_generator.pt"),
]


def build_manifest():
    entries, trees = [], {}
    for repo, local_rel, remote in JOBS:
        if repo not in trees:
            trees[repo] = {r["path"]: r for r in fm.repo_tree(repo)
                           if r.get("type") == "file"}
        tree = trees[repo]
        if remote is None:
            picks = [p for p in tree if p.endswith(W2V_SUFFIXES)]
        else:
            if remote not in tree:
                raise SystemExit(f"{repo}: remote file not found: {remote}")
            picks = [remote]
        for p in picks:
            r = tree[p]
            lfs = r.get("lfs") or {}
            rel = os.path.join(local_rel, p) if remote is None else local_rel
            entries.append({
                "repo": repo, "remote": p, "local": rel.replace("\\", "/"),
                "size": lfs.get("size") or r.get("size") or 0,
                "blob_oid": r.get("oid"), "lfs_sha256": lfs.get("oid"),
            })
    entries.sort(key=lambda e: e["local"])
    man = {"dest": DEST, "files": entries, "n_files": len(entries),
           "total_bytes": sum(e["size"] for e in entries)}
    os.makedirs(fm.MANIFEST_DIR, exist_ok=True)
    tmp = MANPATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    os.replace(tmp, MANPATH)
    fm.log(f"[aux manifest] {len(entries)} files "
           f"{man['total_bytes']/1024**3:.2f} GiB -> {MANPATH}")
    return man


def fetch(man):
    try:
        import hf_transfer  # noqa
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    except ImportError:
        pass
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from huggingface_hub import snapshot_download, hf_hub_download

    # whole-repo subset first
    for repo, local_rel, remote in JOBS:
        if remote is not None:
            continue
        target = os.path.join(DEST, local_rel)
        os.makedirs(target, exist_ok=True)
        pats = ["*" + s for s in W2V_SUFFIXES]
        for attempt in range(1, 7):
            try:
                fm.log(f"[aux] snapshot {repo} -> {target}")
                snapshot_download(repo_id=repo, local_dir=target,
                                  allow_patterns=pats, max_workers=8)
                break
            except Exception as e:
                fm.log(f"  attempt {attempt}: {type(e).__name__}: {str(e)[:200]}")
                time.sleep(min(15 * attempt, 60))
    # single files
    for e in man["files"]:
        if e["repo"] == "facebook/w2v-bert-2.0":
            continue
        target = os.path.join(DEST, e["local"])
        os.makedirs(os.path.dirname(target), exist_ok=True)
        for attempt in range(1, 7):
            try:
                # hf_hub_download preserves the remote subpath, so land it in a
                # staging dir then move to the exact flat name the code expects.
                stage = os.path.join(DEST, ".stage")
                os.makedirs(stage, exist_ok=True)
                hf_hub_download(repo_id=e["repo"], filename=e["remote"],
                                local_dir=stage)
                got = os.path.join(stage, e["remote"])
                if os.path.abspath(got) != os.path.abspath(target):
                    os.replace(got, target)
                # prune empty dirs left behind by the move
                d = os.path.dirname(got)
                while d.startswith(stage) and d != stage:
                    if os.path.isdir(d) and not os.listdir(d):
                        os.rmdir(d)
                    d = os.path.dirname(d)
                break
            except Exception as ex:
                fm.log(f"  attempt {attempt}: {type(ex).__name__}: {str(ex)[:200]}")
                time.sleep(min(15 * attempt, 60))
    return 0


def verify(man):
    def check(e):
        p = os.path.join(DEST, e["local"])
        r = {"path": e["local"], "expect_size": e["size"]}
        if not os.path.isfile(p):
            r["status"] = "MISSING"
            return r
        n = os.path.getsize(p)
        r["actual_size"] = n
        if n != e["size"]:
            r["status"] = "SIZE_MISMATCH"
            return r
        want = e.get("lfs_sha256")
        kind, digest = fm.hash_file(p, want, e["size"])
        exp = want if want else e.get("blob_oid")
        r["hash_kind"] = kind
        r["status"] = ("OK" if exp and digest == exp
                       else ("NO_EXPECTED_HASH" if not exp else "HASH_MISMATCH"))
        return r

    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(check, man["files"]))
    n = len(res)
    ok = sum(1 for r in res if r["status"] == "OK")
    bad = [r for r in res if r["status"] != "OK"]
    for r in bad:
        fm.log(f"    BAD {r['status']} {r['path']}")
    fm.log(f"[aux verify] {ok}/{n} OK coverage={ok/n*100 if n else 0:.1f}%")
    rp = os.path.join(fm.MANIFEST_DIR, "verify_report_aux.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump({"n_files": n, "n_ok": ok, "bad": bad, "files": res},
                  f, ensure_ascii=False, indent=1)
    good = (n > 0 and ok == n and not bad)
    fm.log("[aux verify] PASS" if good else "[aux verify] FAIL")
    return 0 if good else 1


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("manifest", "all"):
        man = build_manifest()
    else:
        man = json.load(open(MANPATH, encoding="utf-8"))
    if cmd in ("fetch", "all"):
        fetch(man)
    if cmd in ("verify", "all"):
        return verify(man)
    return 0


if __name__ == "__main__":
    sys.exit(main())