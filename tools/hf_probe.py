"""Query HF API to resolve real repo ids for the 5 requested TTS families.

Read-only: only hits /api/models endpoints, prints candidates.
"""
import json
import sys
import urllib.request

BASE = "https://huggingface.co"
TERMS = ["IndexTTS", "VoxCPM", "CosyVoice", "Qwen3-TTS", "FireRedTTS"]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode("utf-8"))


def search(term):
    url = f"{BASE}/api/models?search={term}&limit=40&sort=downloads&direction=-1"
    try:
        rows = get(url)
    except Exception as e:
        print(f"  [ERR] {type(e).__name__}: {e}")
        return
    for m in rows:
        print(f"  {m.get('id'):<58} dl={m.get('downloads', 0):<8} "
              f"mod={str(m.get('lastModified', ''))[:10]}")


def main():
    for t in TERMS:
        print(f"=== search: {t} ===", flush=True)
        search(t)
        print(flush=True)


if __name__ == "__main__":
    sys.exit(main())