#!/usr/bin/env python3
"""Build an anonymized P2 listening package (audio + HTML form + answer key).

Spec JSON: list of
  {"blind_id": "...", "audio_path": "...", "source_text": "...",
   "target": "...", "kind": "intonation|emotion|pause|pron|...",
   "dimensions": ["intonation","naturalness"]}

The HTML contains no model names; answer_key.json maps blind_id -> audio_path.
"""
import argparse
import json
import os
import shutil
import sys

E = "/data/shangguanrz/projectsss/speakgenerate/experiments/model-test"
DIMS = ["content_pronunciation", "speaking_rate", "pauses_boundaries",
        "intonation", "emotion", "naturalness", "audio_quality",
        "voice_consistency"]
DIM_CN = {"content_pronunciation": "内容/发音", "speaking_rate": "语速",
          "pauses_boundaries": "停顿/边界", "intonation": "语气",
          "emotion": "情绪", "naturalness": "自然度",
          "audio_quality": "音质", "voice_consistency": "音色一致性"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out-dir", default=os.path.join(E, "blind", "P2"))
    args = ap.parse_args()
    spec = json.load(open(args.spec, encoding="utf-8"))
    out = args.out_dir
    adir = os.path.join(out, "audio")
    os.makedirs(adir, exist_ok=True)
    key = {}
    rows = []
    for i, s in enumerate(spec, 1):
        # Always use an opaque sequential ID; spec IDs like
        # "pause_indextts25_period" would leak model names into the HTML.
        orig = s.get("blind_id") or f"item_{i}"
        bid = f"p2_{i:03d}"
        if bid in key:
            raise SystemExit(f"duplicate blind_id {bid}")
        src = s["audio_path"]
        if not os.path.isfile(src):
            raise SystemExit(f"missing {src}")
        dst = os.path.join(adir, bid + ".wav")
        shutil.copyfile(src, dst)
        key[bid] = {"spec_id": orig, "audio_path": src,
                    "source_text": s.get("source_text", ""),
                    "target": s.get("target", ""), "kind": s.get("kind", ""),
                    "model": s.get("model", "")}
        rows.append({"blind_id": bid, "source_text": s.get("source_text", ""),
                     "target": s.get("target", ""), "kind": s.get("kind", ""),
                     "dimensions": s.get("dimensions", DIMS)})
    data = json.dumps(rows, ensure_ascii=False)
    dims = json.dumps(DIMS, ensure_ascii=False)
    dimcn = json.dumps(DIM_CN, ensure_ascii=False)
    html = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>P2 匿名听评</title>
<style>
body{font-family:-apple-system,"Segoe UI","Noto Sans SC",sans-serif;max-width:980px;margin:24px auto;padding:0 16px;background:#f6f7f9;color:#1d2129}
h1{font-size:22px;margin:0 0 6px} .sub{color:#6b7280;font-size:13px;margin-bottom:16px}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;margin:14px 0;box-shadow:0 1px 2px rgba(0,0,0,.03)}
.meta{font-size:12px;color:#6b7280;margin:2px 0 8px;word-break:break-all}
.src{font-size:15px;line-height:1.6;margin:6px 0}
.target{display:inline-block;background:#eef2ff;color:#3730a3;border-radius:999px;padding:2px 10px;font-size:12px;margin:4px 0}
audio{width:100%;margin:6px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin-top:8px}
label{font-size:12px;color:#374151;display:flex;flex-direction:column;gap:3px}
select,textarea,button{font:inherit;border:1px solid #d1d5db;border-radius:8px;padding:6px 8px;background:#fff}
textarea{width:100%;min-height:44px}
.bar{position:sticky;top:0;background:#f6f7f9;padding:10px 0;display:flex;gap:10px;align-items:center;z-index:5}
button{background:#2563eb;color:#fff;border-color:#2563eb;cursor:pointer;padding:9px 16px}
button.sec{background:#fff;color:#2563eb}
.small{font-size:12px;color:#6b7280}
</style></head><body>
<h1>P2 匿名听评</h1>
<div class="sub">音频与模型名已匿名。每个片段按需评分；1=严重问题，5=在本次范围内未发现明显问题。不确定选“未评”。</div>
<div class="bar"><button onclick="exportJSON()">导出 reviews.json</button>
<button class="sec" onclick="fillNA()">未评项填 NA</button>
<span class="small" id="count"></span></div>
<div id="cards"></div>
<script>
const ROWS = __DATA__;
const DIMS = __DIMS__;
const DIMCN = __DIMCN__;
const cards = document.getElementById('cards');
function sel(name,id){let o='<option value="">未评</option>';for(let i=1;i<=5;i++)o+=`<option value="${i}">${i}</option>`;o+='<option value="NA">NA</option>';return `<label>${name}<select data-id="${id}" data-field="${name}">${o}</select></label>`}
ROWS.forEach((r,idx)=>{
  const div=document.createElement('div'); div.className='card';
  div.innerHTML=`<div class="meta">${r.blind_id} · ${r.kind}</div>
  <div class="src">${r.source_text}</div>
  ${r.target?`<div class="target">目标：${r.target}</div>`:''}
  <audio controls preload="none" src="audio/${r.blind_id}.wav"></audio>
  <div class="grid">${DIMS.map(d=>sel(DIMCN[d],r.blind_id)).join('')}
  <label>结论<select data-id="${r.blind_id}" data-field="decision">
  <option value="">未评</option><option>usable</option><option>revise</option><option>human_review</option><option>unavailable</option>
  </select></label></div>
  <label style="margin-top:8px">问题/备注<textarea data-id="${r.blind_id}" data-field="notes"></textarea></label>`;
  cards.appendChild(div);
});
function collect(){
  const out={}; ROWS.forEach(r=>{out[r.blind_id]={blind_id:r.blind_id,source_text:r.source_text,target:r.target,kind:r.kind,scores:{},decision:'',notes:''}});
  document.querySelectorAll('select[data-id]').forEach(el=>{
    const id=el.dataset.id, f=el.dataset.field, v=el.value;
    if(f==='decision') out[id].decision=v;
    else if(v==='NA') out[id].scores[f]='NA';
    else if(v!=='') out[id].scores[f]=Number(v);
  });
  document.querySelectorAll('textarea[data-id]').forEach(el=>{out[el.dataset.id].notes=el.value});
  return out;
}
function update(){const c=collect();const n=Object.values(c).filter(x=>x.decision||Object.keys(x.scores).length).length;document.getElementById('count').textContent=`已评 ${n}/${ROWS.length}`}
document.addEventListener('change',update);document.addEventListener('input',update);
function fillNA(){document.querySelectorAll('select[data-id]').forEach(el=>{if(el.dataset.field!=='decision'&&!el.value)el.value='NA'});update()}
function exportJSON(){
  const blob=new Blob([JSON.stringify({schema_version:'p2-human-1.0',reviews:Object.values(collect())},null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='reviews.json';a.click();
}
</script></body></html>"""
    html = (html.replace("__DATA__", data).replace("__DIMS__", dims)
            .replace("__DIMCN__", dimcn))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    json.dump(key, open(os.path.join(out, "answer_key.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"wrote {out} n={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
