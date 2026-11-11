"""Fetch recursive file tree + sizes for candidate repos (read-only)."""
import json
import sys
import urllib.request

BASE = "https://huggingface.co"
REPOS = [
    "IndexTeam/IndexTTS-2.5",
    "openbmb/VoxCPM2",
    "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "FireRedTeam/FireRedTTS3",
]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def tree(repo):
    out, cursor = [], None
    while True:
        url = f"{BASE}/api/models/{repo}/tree/main?recursive=true"
        if cursor:
            url += "&cursor=" + urllib.parse.quote(cursor)
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=60) as r:
            rows = json.loads(r.read().decode("utf-8"))
            link = r.headers.get("Link", "")
        out.extend(rows)
        if 'rel="next"' in link:
            cursor = link.split("cursor=")[1].split(">")[0]
        else:
            break
    return out


def main():
    import urllib.parse  # noqa
    grand = 0
    for repo in REPOS:
        print(f"=== {repo} ===", flush=True)
        try:
            rows = tree(repo)
        except Exception as e:
            print(f"  [ERR] {type(e).__name__}: {e}", flush=True)
            continue
        files = [r for r in rows if r.get("type") == "file"]
        total = 0
        for f in sorted(files, key=lambda x: -(x.get("size") or 0)):
            sz = f.get("size") or 0
            lfs = f.get("lfs") or {}
            if lfs.get("size"):
                sz = lfs["size"]
            total += sz
            print(f"  {sz/1024**2:10.2f} MB  {f['path']}"
                  + (f"  sha256={lfs['oid'][:16]}" if lfs.get("oid") else ""))
        print(f"  -- {len(files)} files, {total/1024**3:.2f} GiB", flush=True)
        grand += total
        print(flush=True)
    print(f"GRAND TOTAL {grand/1024**3:.2f} GiB")


if __name__ == "__main__":
    import urllib.parse
    sys.exit(main())