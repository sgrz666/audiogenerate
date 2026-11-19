#!/usr/bin/env python3
"""Strictly serial orchestrator: one config at a time, one process per config.

Never runs two models concurrently. Waits for each child to exit (which releases
its VRAM and host RAM) before starting the next, and records the gap.

Usage: mt_serial.py --phase P0 [--configs a,b,c] [--limit N]
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--configs", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    cfgs = C.CONFIGS
    if args.configs:
        want = [x.strip() for x in args.configs.split(",") if x.strip()]
        cfgs = [c for c in C.CONFIGS if c["config_id"] in want]

    logdir = os.path.join(C.E, "logs")
    os.makedirs(logdir, exist_ok=True)
    summary = []
    C.log(f"=== SERIAL {args.phase}: {len(cfgs)} configs, one at a time ===")
    for c in cfgs:
        cid = c["config_id"]
        py = os.path.join(C.E, "envs", c["env"], "bin", "python")
        rec = {"config_id": cid, "phase": args.phase, "env": c["env"]}
        if not os.path.isfile(py):
            rec.update(status="not_tested", reason=f"env missing: {py}")
            C.log(f"[{cid}] SKIP not_tested ({rec['reason']})")
            summary.append(rec)
            continue
        lf = os.path.join(logdir, f"{args.phase}_{cid}.log")
        cmd = [py, os.path.join(C.E, "scripts", "mt_run.py"),
               "--config", cid, "--phase", args.phase]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(C.E, "scripts")
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        C.log(f"[{cid}] START -> {lf}")
        t0 = time.time()
        with open(lf, "w", encoding="utf-8") as f:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                               env=env, timeout=args.timeout)
        wall = round(time.time() - t0, 1)
        rec.update(exit_code=p.returncode, wall_s=wall, log=lf)
        # count statuses from the result file
        rp = C.result_path(cid, args.phase)
        counts = {}
        if os.path.isfile(rp):
            with open(rp, encoding="utf-8") as f:
                for ln in f:
                    try:
                        d = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    counts[d.get("status", "?")] = counts.get(d.get("status", "?"), 0) + 1
        rec["counts"] = counts
        C.log(f"[{cid}] EXIT={p.returncode} wall={wall}s counts={counts}")
        summary.append(rec)
        time.sleep(3)  # let the OS reclaim before the next model

    sp = os.path.join(C.E, "results", f"serial_{args.phase}.json")
    os.makedirs(os.path.dirname(sp), exist_ok=True)
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    C.log(f"=== SERIAL {args.phase} COMPLETE -> {sp} ===")
    for r in summary:
        C.log(f"  {r['config_id']:22s} {r.get('status', 'run')} "
              f"exit={r.get('exit_code')} counts={r.get('counts')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())