#!/usr/bin/env python3
"""Fetch the 5 TTS model repos into speakgenerate/model and verify them.

Subcommands
  manifest  build per-repo expected-file manifests from the HF API (independent
            of the downloader, so verification is not a tautology)
  fetch     snapshot_download each repo with resume + retry + rate/ETA logging
  verify    check EVERY manifest file: size, plus sha256 for LFS files or git
            blob sha1 for plain files. Reports coverage; fails unless 100%.

Only writes under model/<name> and tools/manifests. Never touches data/.
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://huggingface.co"
HOSTS = ["https://huggingface.co", "https://hf-mirror.com"]
ROOT = "/data/shangguanrz/projectsss/speakgenerate/model"
MANIFEST_DIR = "/data/shangguanrz/projectsss/speakgenerate/tools/manifests"

REPOS = [
    ("IndexTeam/IndexTTS-2.5", "IndexTTS-2.5"),
    ("openbmb/VoxCPM2", "VoxCPM2"),
    ("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "Fun-CosyVoice3-0.5B-2512"),
    ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "Qwen3-TTS-12Hz-1.7B-Base"),
    ("FireRedTeam/FireRedTTS3", "FireRedTTS3"),
    # required companion for CosyVoice3 text frontend (per its README), not a 6th TTS model
    ("FunAudioLLM/CosyVoice-ttsfrd", "CosyVoice-ttsfrd"),
    # Qwen3-TTS README documents the tokenizer as a separate download; the qwen-tts
    # package loads it by repo name. Base/speech_tokenizer has the same sha256.
    ("Qwen/Qwen3-TTS-Tokenizer-12Hz", "Qwen3-TTS-Tokenizer-12Hz"),
]


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def api_get(url, tries=8):
    """GET a HF API path, retrying and alternating official/mirror hosts."""
    path = url.split("://", 1)[1].split("/", 1)[1]
    last = None
    for attempt in range(tries):
        host = HOSTS[attempt % len(HOSTS)]
        try:
            req = urllib.request.Request(host + "/" + path,
                                         headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8")), r.headers.get("Link", "")
        except Exception as e:
            last = e
            log(f"  [api retry {attempt+1}/{tries}] {host} {type(e).__name__}")
            time.sleep(min(3 * (attempt + 1), 20))
    raise last


def repo_revision(repo):
    data, _ = api_get(f"{API}/api/models/{repo}")
    return data.get("sha")


def repo_tree(repo):
    """Recursive file tree, following Link pagination."""
    out, cursor = [], None
    while True:
        url = f"{API}/api/models/{repo}/tree/main?recursive=true"
        if cursor:
            url += "&cursor=" + urllib.parse.quote(cursor, safe="")
        rows, link = api_get(url)
        out.extend(rows)
        if 'rel="next"' in link and "cursor=" in link:
            cursor = link.split("cursor=")[1].split(">")[0].split("&")[0]
        else:
            break
    return out


def cmd_manifest(args):
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    selected = select(args)
    grand = 0
    for repo, name in selected:
        rev = repo_revision(repo)
        rows = repo_tree(repo)
        files = []
        for r in rows:
            if r.get("type") != "file":
                continue
            lfs = r.get("lfs") or {}
            size = lfs.get("size") or r.get("size") or 0
            files.append({
                "path": r["path"],
                "size": size,
                "blob_oid": r.get("oid"),
                "lfs_sha256": lfs.get("oid"),
            })
        files.sort(key=lambda x: x["path"])
        total = sum(f["size"] for f in files)
        man = {"repo": repo, "name": name, "revision": rev,
               "files": files, "n_files": len(files), "total_bytes": total}
        path = os.path.join(MANIFEST_DIR, name + ".json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(man, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
        grand += total
        log(f"[manifest] {name} rev={rev[:12]} files={len(files)} "
            f"{total/1024**3:.2f} GiB -> {path}")
    log(f"[manifest] TOTAL {grand/1024**3:.2f} GiB")
    return 0


def load_manifest(name):
    with open(os.path.join(MANIFEST_DIR, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def dir_bytes(path):
    """Real bytes on disk, including .cache/*.incomplete so progress is honest."""
    total = 0
    for base, dirs, names in os.walk(path):
        for n in names:
            try:
                total += os.path.getsize(os.path.join(base, n))
            except OSError:
                pass
    return total


def cmd_fetch(args):
    # prefer accelerated backends when present, else plain HTTP
    try:
        import hf_transfer  # noqa
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        log("[fetch] hf_transfer enabled")
    except ImportError:
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from huggingface_hub import snapshot_download
    import huggingface_hub
    log(f"[fetch] huggingface_hub {huggingface_hub.__version__}")

    selected = select(args)
    failed = []
    for repo, name in selected:
        man = load_manifest(name)
        dest = os.path.join(ROOT, name)
        os.makedirs(dest, exist_ok=True)
        expect = man["total_bytes"]
        log(f"[start] {repo} -> {dest} expect {expect/1024**3:.2f} GiB "
            f"({man['n_files']} files) rev={man['revision'][:12]}")

        stop = threading.Event()

        def monitor():
            t0 = time.time()
            b0 = dir_bytes(dest)
            while not stop.wait(30):
                b = dir_bytes(dest)
                dt = max(time.time() - t0, 1e-6)
                sp = (b - b0) / dt
                eta = (expect - b) / sp / 60 if sp > 1 else -1
                log(f"  [{name}] {b/1024**3:.2f}/{expect/1024**3:.2f} GiB "
                    f"({b/expect*100:.1f}%) {sp/1024**2:.1f} MiB/s "
                    f"ETA {eta:.1f} min")

        th = threading.Thread(target=monitor, daemon=True)
        th.start()
        ok = False
        for attempt in range(1, 9):
            try:
                snapshot_download(repo_id=repo, revision=man["revision"],
                                  local_dir=dest, max_workers=args.workers)
                # snapshot_download can return WITHOUT error while incomplete
                # ("Returning existing local_dir ... cannot be accessed"), so
                # treat a byte shortfall as a failure and keep retrying.
                got = dir_bytes(dest)
                if got < expect:
                    raise RuntimeError(
                        f"incomplete after return: {got}/{expect} bytes")
                ok = True
                break
            except Exception as e:
                log(f"  [{name}] attempt {attempt} failed: "
                    f"{type(e).__name__}: {str(e)[:300]}")
                if attempt == 2:
                    from huggingface_hub import constants
                    constants.ENDPOINT = "https://hf-mirror.com"
                    log(f"  [{name}] switching endpoint -> {constants.ENDPOINT}")
                time.sleep(min(15 * attempt, 90))
        stop.set()
        th.join(timeout=5)
        got = dir_bytes(dest)
        if ok and got >= expect:
            log(f"[done] {name} {got/1024**3:.2f} GiB on disk")
        else:
            log(f"[FAIL] {name} ok={ok} disk={got/1024**3:.2f} GiB "
                f"expect={expect/1024**3:.2f} GiB")
            failed.append(name)
    if failed:
        log(f"[fetch] FAILED: {', '.join(failed)}")
        return 1
    log("[fetch] all repos downloaded")
    return 0


def hash_file(path, want_sha256, size):
    """Return (kind, digest). LFS -> sha256 of content; plain -> git blob sha1."""
    if want_sha256:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(blk)
        return "sha256", h.hexdigest()
    h = hashlib.sha1()
    h.update(b"blob %d\0" % size)
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(blk)
    return "sha1", h.hexdigest()


def cmd_verify(args):
    selected = select(args)
    root = args.root or ROOT
    report, all_ok = {}, True
    for repo, name in selected:
        man = load_manifest(name)
        dest = os.path.join(root, name)
        files = man["files"]
        results = []

        def check(f):
            p = os.path.join(dest, f["path"])
            r = {"path": f["path"], "expect_size": f["size"]}
            if not os.path.isfile(p):
                r["status"] = "MISSING"
                return r
            actual = os.path.getsize(p)
            r["actual_size"] = actual
            if actual != f["size"]:
                r["status"] = "SIZE_MISMATCH"
                return r
            want256 = f.get("lfs_sha256")
            try:
                kind, digest = hash_file(p, want256, f["size"])
            except Exception as e:
                r["status"] = "HASH_ERROR:" + type(e).__name__
                return r
            expect = want256 if want256 else f.get("blob_oid")
            r["hash_kind"] = kind
            r["expect_hash"] = expect
            r["actual_hash"] = digest
            if not expect:
                r["status"] = "NO_EXPECTED_HASH"
            elif digest == expect:
                r["status"] = "OK"
            else:
                r["status"] = "HASH_MISMATCH"
            return r

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(check, files))

        n = len(files)
        ok = sum(1 for r in results if r["status"] == "OK")
        bad = [r for r in results if r["status"] != "OK"]
        coverage = ok / n if n else 0.0
        nbytes = sum(r.get("actual_size", 0) for r in results)
        log(f"[verify] {name}: {ok}/{n} OK coverage={coverage*100:.1f}% "
            f"{nbytes/1024**3:.2f} GiB")
        for r in bad:
            log(f"    BAD {r['status']} {r['path']} "
                f"expect={r.get('expect_size')} actual={r.get('actual_size')}")
        report[name] = {"repo": repo, "revision": man["revision"],
                        "n_files": n, "n_ok": ok, "coverage": coverage,
                        "bytes": nbytes, "bad": bad}
        # gate: full coverage AND zero bad AND non-degenerate file count
        if coverage != 1.0 or bad or n == 0:
            all_ok = False
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    rp = os.path.join(MANIFEST_DIR, "verify_report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    log(f"[verify] report -> {rp}")
    log("[verify] PASS" if all_ok else "[verify] FAIL")
    return 0 if all_ok else 1


def select(args):
    if not args.repos:
        return REPOS
    keys = [k.strip() for k in args.repos.split(",") if k.strip()]
    out = []
    for k in keys:
        hit = [r for r in REPOS if k in (r[0], r[1])]
        if not hit:
            raise SystemExit(f"unknown repo selector: {k}")
        out.extend(hit)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["manifest", "fetch", "verify"])
    ap.add_argument("--repos", default="", help="comma separated names/ids")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--root", default="", help="override model root (verify)")
    args = ap.parse_args()
    return {"manifest": cmd_manifest, "fetch": cmd_fetch,
            "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())