"""Stability metrics for talk/interview crop plans (no labels needed).

pan_px_per_min  total horizontal crop travel inside shots (excludes hard cuts), px of source per minute
swings_per_min  direction changes >= 5% of width inside shots
moving_share    share of frames where the crop is moving (> 2 px/frame-second... i.e. > 0.5% W/s)
face_in         share of keyframes whose Drex face focus is inside the crop
python3 scripts/eval_talk.py --name interview
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def metrics(plan, kfs):
    W, fps, c0 = plan["W"], plan["fps"], plan["clip"][0]
    cx, h = np.array(plan["cx"]), np.array(plan["h"])
    # planner fills frames with times >= cut, so the first frame of a segment is ceil((c - c0) * fps)
    b = [0] + [int(np.ceil((c - c0) * fps - 1e-9)) for c in plan["cuts"]] + [len(cx)]
    travel, swings, moving, n = 0.0, 0, 0, 0
    for a_, b_ in zip(b[:-1], b[1:]):
        seg = cx[a_:b_]
        if len(seg) < 3:
            continue
        d = np.diff(seg)
        travel += np.abs(d).sum()
        moving += int((np.abs(d) * fps > 0.005 * W).sum())
        n += len(d)
        ext = [seg[0]]
        for x in seg[1:]:
            if len(ext) >= 2 and np.sign(x - ext[-1]) == np.sign(ext[-1] - ext[-2]):
                ext[-1] = x
            elif abs(x - ext[-1]) >= 0.05 * W or len(ext) == 1:
                ext.append(x)
        swings += max(0, len(ext) - 2)
    dur = len(cx) / fps
    fin = tot = 0
    for k in kfs:
        f = k.get("focus_kf")
        if not f or k["shot"] == "other":
            continue
        i = int(round((k["t"] - c0) * fps))
        if 0 <= i < len(cx):
            tot += 1
            fin += abs(f["x"] * W / k["kf_W"] - cx[i]) <= h[i] * 9 / 32
    return {"pan_px_per_min": travel / dur * 60, "swings_per_min": swings / dur * 60,
            "moving_share": moving / max(n, 1), "face_in": fin / max(tot, 1), "dur": dur}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="interview")
    ap.add_argument("--plans", default="plan_{:02d}.json")
    a = ap.parse_args()
    run = ROOT / "runs" / a.name
    s3 = json.loads((run / "s3.json").read_text())
    rows = []
    for ci in range(len(s3)):
        p = run / a.plans.format(ci)
        if not p.exists():
            continue
        m = metrics(json.loads(p.read_text()), s3[str(ci)])
        rows.append(m)
        print(f"clip {ci}: pan {m['pan_px_per_min']:6.0f} px/min  swings {m['swings_per_min']:4.1f}/min  "
              f"moving {m['moving_share']:.0%}  face_in {m['face_in']:.0%}")
    D = sum(r["dur"] for r in rows)
    agg = {k: sum(r[k] * r["dur"] for r in rows) / D for k in ("pan_px_per_min", "swings_per_min", "moving_share", "face_in")}
    print("ALL   ", f"pan {agg['pan_px_per_min']:.0f} px/min  swings {agg['swings_per_min']:.1f}/min  "
          f"moving {agg['moving_share']:.0%}  face_in {agg['face_in']:.0%}")
    return agg


if __name__ == "__main__":
    main()
