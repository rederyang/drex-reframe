"""Gemini crop director on EXACTLY the clips the Drex pipeline produced, for a fair crop comparison.

Event discovery is not repeated: each clip interval comes from runs/<drex_run>/events.json.
Downstream (geometry, smoothing, ball-in-frame constraint, render) is the same code as Drex, so
differences come from the decision model only.

python3 scripts/gemini_aligned.py --drex-run clasico --events 1,3,5,6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reframe import media  # noqa: E402
from reframe.gemini import Gemini, save  # noqa: E402
from reframe.gemini_baseline import ANCHOR_SCHEMA, api_cost_estimate, extract_clip, validate_anchors  # noqa: E402
from reframe.pipeline import plan_clip, render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drex-run", default="clasico")
    ap.add_argument("--events", default="1,3,5,6")
    ap.add_argument("--name", default="gemini_aligned_clasico")
    ap.add_argument("--model", default="gemini-4-hydrogen")
    a = ap.parse_args()
    drex = ROOT / "runs" / a.drex_run
    s0 = json.loads((drex / "s0.json").read_text())
    events = json.loads((drex / "events.json").read_text())
    video = Path(s0["video"])
    W, H = s0["W"], s0["H"]
    run = ROOT / "runs" / a.name
    run.mkdir(parents=True, exist_ok=True)
    g = Gemini(run, a.model)
    summary = []
    for ei in [int(x) for x in a.events.split(",")]:
        ev = events[ei]
        c0, c1 = ev["clip"]
        dur = c1 - c0
        prefix = f"event_{ei:02d}"
        clip = run / f"{prefix}_input.mp4"
        extract_clip(video, c0, dur, clip)
        t_up = time.monotonic()
        file = g.upload(clip, prefix)
        t_up = time.monotonic() - t_up
        ts = [round(float(t), 3) for t in np.arange(0, dur - 1e-6, 0.5)]
        # Same director prompt as reframe/gemini_baseline.py, minus the event hypothesis (Drex clips
        # have no text description); the commentary snippet is given instead, as Drex s1 also saw it.
        prompt = (
            "You are the camera director reframing this football clip into a 9:16 vertical crop. "
            "Determine the action focus yourself, following ball/receiver/goal during play and the relevant "
            "person during reactions or closeups. Use temporal context; do not lock onto one player throughout. "
            "Return exactly one anchor for EACH timestamp in this list, in order: " + json.dumps(ts) + ". "
            "All timestamps are local seconds from the start of THIS SHORT CLIP. "
            "region is [left,top,right,bottom] of the must-keep action, normalized to [0,1] of the full image; "
            "focus is [x,y] of the most important visible point (the ball in play, or the main face). "
            "Do not output a ball location based on guessing when it is invisible; set uncertain=true and use "
            "a conservative relevant region. Keep focus stable within shots but reset after camera cuts. "
            "Use shot types wide, medium, closeup, crowd or graphic. Describe the focus briefly. "
            f"Broadcast commentary around this moment, for context only: {ev.get('commentary', '')[:600]}"
        )
        kf_path = run / f"{prefix}_keyframes.json"
        t_ask = time.monotonic()
        if kf_path.exists():
            kfs = json.loads(kf_path.read_text())
        else:
            data = g.ask(file, prompt, ANCHOR_SCHEMA, prefix + "_anchors")
            try:
                kfs = validate_anchors(data, dur, c0, W, H)
            except (ValueError, KeyError, TypeError) as e:
                data = g.ask(file, prompt + f" Previous response failed validation: {e}. Produce a corrected full response.",
                             ANCHOR_SCHEMA, prefix + "_anchors_repair")
                kfs = validate_anchors(data, dur, c0, W, H)
            save(kf_path, kfs)
        t_ask = time.monotonic() - t_ask
        plan = plan_clip(s0, ev, kfs)
        save(run / f"{prefix}_plan.json", plan)
        render(video, plan, run / f"{prefix}_vertical.mp4", run / f"{prefix}_debug.mp4", kfs)
        resp = json.loads((run / f"{prefix}_anchors.response.json").read_text())
        usage = resp.get("usage") or {}
        summary.append({"event_index": ei, "clip": [c0, c1], "upload_s": t_up, "decide_s": t_ask,
                        "usage": usage, "usd": (api_cost_estimate(usage, a.model) or {}).get("usd")})
        print(f"[gemini] event {ei} {c0:.0f}-{c1:.0f}s upload {t_up:.1f}s decide {t_ask:.1f}s "
              f"in_tok {usage.get('total_input_tokens')} usd {summary[-1]['usd']}", flush=True)
    save(run / "summary.json", summary)


if __name__ == "__main__":
    main()
