"""Score camera paths without ground truth, using Drex's own per-keyframe focus (ball / face).

Metrics (live-play keyframes only, i.e. shot wide/medium):
  ball_out   share of keyframes where the focus point is outside the final crop
  ball_edge  share where it sits in the outer 10% of the crop on either side
  accel      median |camera acceleration| in % of frame width per s^2 (lower = calmer)
  reversals  direction changes per minute within shots (pan left->right->left = jitter)

python3 scripts/eval_path.py --name clasico [--old]   # --old scores saved plan_*.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reframe import pipeline as pl  # noqa: E402


def score(s0, events, s3, plans):
    W, fps = s0["W"], s0["fps"]
    out = edge = n = 0
    acc, rev, dur = [], 0, 0.0
    for ei, plan in plans.items():
        c0 = plan["clip"][0]
        cx, h = np.array(plan["cx"]), np.array(plan["h"])
        for k in s3[ei]:
            if k["shot"] not in ("wide", "medium") or not k.get("focus_kf"):
                continue
            i = int(round((k["t"] - c0) * fps))
            if i >= len(cx):
                continue
            fx = k["focus_kf"]["x"] * W / k["kf_W"]
            w = h[i] * 9 / 16
            d = abs(fx - cx[i])
            n += 1
            out += d > w / 2
            edge += 0.4 * w < d <= w / 2
        cuts = plan["cuts"]
        b = [0] + [int(round((c - c0) * fps)) for c in cuts] + [len(cx)]
        for a_, b_ in zip(b[:-1], b[1:]):
            seg = cx[a_:b_]
            if len(seg) < 5:
                continue
            v = np.diff(seg) * fps
            acc.extend(np.abs(np.diff(v) * fps) / W * 100)
            # count only meaningful swings: direction changes between extrema >= 5% of width apart
            ext = [seg[0]]
            for x in seg[1:]:
                if len(ext) >= 2 and np.sign(x - ext[-1]) == np.sign(ext[-1] - ext[-2]):
                    ext[-1] = x
                elif abs(x - ext[-1]) >= 0.05 * W or len(ext) == 1:
                    ext.append(x)
            rev += max(0, len(ext) - 2)
            dur += len(seg) / fps
    return {"ball_out": out / n, "ball_edge": edge / n, "accel": float(np.median(acc)),
            "reversals_per_min": rev / dur * 60, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="clasico")
    ap.add_argument("--old", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    run = ROOT / "runs" / a.name
    s0 = json.loads((run / "s0.json").read_text())
    events = json.loads((run / "events.json").read_text())
    s3 = json.loads((run / "s3.json").read_text())
    if a.old:
        plans = {str(i): json.loads((run / f"plan_{i:02d}.json").read_text()) for i in range(len(events))}
        print("old  ", fmt(score(s0, events, s3, plans)))
        return
    grid = [dict(pl.P)]
    if a.sweep:
        grid = [dict(pl.P, tau_live=t, ball_iters=b, vcap_live=v)
                for t, b, v in itertools.product([0.5, 0.8, 1.2, 1.6], [0, 3], [0.9])]
    for cfg in grid:
        pl.P.update(cfg)
        plans = {str(i): pl.plan_clip(s0, e, s3[str(i)]) for i, e in enumerate(events)}
        print(f"tau={cfg['tau_live']} ball_iters={cfg['ball_iters']} vcap={cfg['vcap_live']}  ",
              fmt(score(s0, events, s3, plans)))


def fmt(m):
    return (f"ball_out {m['ball_out']:.0%}  ball_edge {m['ball_edge']:.0%}  "
            f"accel {m['accel']:.1f}%W/s²  swings {m['reversals_per_min']:.1f}/min  (n={m['n']})")


if __name__ == "__main__":
    main()
