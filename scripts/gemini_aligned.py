"""Gemini as the crop director on EXACTLY the clips the Drex pipeline produced (fair comparison).

Event / highlight discovery is not repeated: each clip interval comes from runs/<drex_run>/events.json.
Downstream geometry, planning and rendering are the same code the Drex run used, so differences
come from the decision model only.

  football: python3 scripts/gemini_aligned.py --profile football --drex-run clasico   --name gemini_aligned_clasico
  talk:     python3 scripts/gemini_aligned.py --profile talk     --drex-run interview --name gemini_aligned_interview
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
from reframe.gemini import Gemini, save  # noqa: E402
from reframe.gemini_baseline import ANCHOR_SCHEMA, api_cost_estimate, extract_clip, validate_anchors  # noqa: E402
from reframe import pipeline as fb  # noqa: E402
from reframe import talk  # noqa: E402

FOOTBALL_PROMPT = (
    "You are the camera director reframing this football clip into a 9:16 vertical crop. "
    "Determine the action focus yourself, following ball/receiver/goal during play and the relevant "
    "person during reactions or closeups. Use temporal context; do not lock onto one player throughout. "
    "Return exactly one anchor for EACH timestamp in this list, in order: {ts}. "
    "All timestamps are local seconds from the start of THIS SHORT CLIP. "
    "region is [left,top,right,bottom] of the must-keep action, normalized to [0,1] of the full image; "
    "focus is [x,y] of the most important visible point (the ball in play, or the main face). "
    "Do not output a ball location based on guessing when it is invisible; set uncertain=true and use "
    "a conservative relevant region. Keep focus stable within shots but reset after camera cuts. "
    "Use shot types wide, medium, closeup, crowd or graphic. Describe the focus briefly. "
    "Broadcast commentary around this moment, for context only: {ctx}"
)
TALK_PROMPT = (
    "You are the camera director reframing this interview clip into a 9:16 vertical crop that always "
    "shows the person who is SPEAKING at each moment (use the audio, mouth movement and gestures). If the "
    "speaker is off-screen, show the most relevant listener. People are seated and barely move, so keep "
    "the framing stable within a camera shot. "
    "Return exactly one anchor for EACH timestamp in this list, in order: {ts}. "
    "All timestamps are local seconds from the start of THIS SHORT CLIP. "
    "region is [left,top,right,bottom] of the speaker's head and shoulders, normalized to [0,1] of the "
    "full image; focus is [x,y] of the speaker's face. Shot types: wide = whole table, medium = two "
    "people side by side, closeup = one person fills the frame, crowd or graphic = b-roll/other. "
    "Describe who you are framing briefly. Transcript of this clip with diarized speaker ids and "
    "seconds from the clip start, for context: {ctx}"
)
TALK_SHOT = {"wide": "group_wide", "medium": "two_shot", "closeup": "single", "crowd": "other", "graphic": "other"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["football", "talk"], default="football")
    ap.add_argument("--drex-run", default="clasico")
    ap.add_argument("--events", default="", help="comma list; default = all clips of the Drex run")
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
    sf = run / "summary.json"
    summary = {e["event_index"]: e for e in json.loads(sf.read_text())} if sf.exists() else {}
    idx = [int(x) for x in a.events.split(",")] if a.events else list(range(len(events)))
    for ei in idx:
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
        if a.profile == "football":
            prompt = FOOTBALL_PROMPT.format(ts=json.dumps(ts), ctx=ev.get("commentary", "")[:600])
        else:
            lines = talk.lines(s0, drex)
            ctx = " ".join(f"[{l['start'] + s0['start'] - c0:.1f}s {l.get('speaker') or '?'}] {l['text']}"
                           for l in lines if l["end"] + s0["start"] >= c0 and l["start"] + s0["start"] <= c1)[:2500]
            prompt = TALK_PROMPT.format(ts=json.dumps(ts), ctx=ctx)
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
            if a.profile == "talk":
                for k in kfs:
                    k["shot"] = TALK_SHOT.get(k["shot"], "other")
            save(kf_path, kfs)
        t_ask = time.monotonic() - t_ask
        plan = fb.plan_clip(s0, ev, kfs) if a.profile == "football" else talk.plan_clip(s0, ev, kfs, drex)
        save(run / f"{prefix}_plan.json", plan)
        out = run / f"{prefix}_vertical.mp4"
        if not out.exists():
            fb.render(video, plan, out, run / f"{prefix}_debug.mp4", kfs)
        usage = {}
        for p in run.glob(f"{prefix}_anchors*.response.json"):
            for k, v in (json.loads(p.read_text()).get("usage") or {}).items():
                if isinstance(v, (int, float)):
                    usage[k] = usage.get(k, 0) + v
        prev = summary.get(ei, {})
        summary[ei] = {"event_index": ei, "clip": [c0, c1],
                       # timings survive cache hits: keep the first real measurement
                       "upload_s": prev.get("upload_s", t_up) if t_up < 1 else t_up,
                       "decide_s": prev.get("decide_s", t_ask) if t_ask < 1 else t_ask,
                       "usage": usage, "usd": (api_cost_estimate(usage, a.model) or {}).get("usd")}
        s = summary[ei]
        print(f"[gemini] {a.profile} clip {ei} {c0:.0f}-{c1:.0f}s upload {s['upload_s']:.1f}s decide {s['decide_s']:.1f}s "
              f"in_tok {usage.get('total_input_tokens')} usd {s['usd']}", flush=True)
        save(sf, sorted(summary.values(), key=lambda e: e["event_index"]))


if __name__ == "__main__":
    main()
