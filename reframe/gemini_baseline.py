"""B1: native Gemini video -> event selection -> crop anchors -> common renderer.

Run: python3 -m reframe.gemini_baseline --video PATH --name NAME
Offline only: Gemini and the common smoother may use future frames.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from . import media
from .gemini import Gemini, ROOT, digest, save
from .pipeline import SHOT_TYPES, plan_clip, render

PROMPT_VERSION = "native-video-v1"


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties)}


NUM = {"type": "number"}
STR = {"type": "string"}
EVENT_SCHEMA = obj({"events": {"type": "array", "items": obj({
    "start_s": NUM, "end_s": NUM, "peak_s": NUM,
    "event": {"type": "string", "enum": ["goal", "own_goal", "big_chance", "skill", "card"]},
    "description": STR, "visual_evidence": STR, "audio_evidence": STR,
})}})
ANCHOR_SCHEMA = obj({"anchors": {"type": "array", "items": obj({
    "t_s": NUM, "shot": {"type": "string", "enum": list(SHOT_TYPES)},
    "region": {"type": "array", "items": NUM, "minItems": 4, "maxItems": 4},
    "focus": {"type": "array", "items": NUM, "minItems": 2, "maxItems": 2},
    "focus_description": STR, "uncertain": {"type": "boolean"},
})}})


def validate_events(data, duration, max_events, max_clip):
    events = data["events"]
    if not isinstance(events, list) or len(events) > max_events:
        raise ValueError("Invalid event count")
    out = []
    for event in events:
        a, b, peak = (float(event[k]) for k in ("start_s", "end_s", "peak_s"))
        if not all(math.isfinite(t) for t in (a, b, peak)):
            raise ValueError("Non-finite event time")
        if not (0 <= a <= peak < b <= duration and 5 <= b - a <= max_clip + 0.05):
            raise ValueError(f"Invalid event interval {a}, {peak}, {b}")
        if any(max(a, p["start_s"]) < min(b, p["end_s"]) for p in out):
            raise ValueError("Overlapping events; live/replay duplication must be resolved")
        out.append({**event, "start_s": a, "end_s": b, "peak_s": peak, "clip": [a, b]})
    return sorted(out, key=lambda e: e["start_s"])


def validate_anchors(data, duration, start, W, H, step=0.5):
    anchors = data["anchors"]
    expected = np.arange(0, duration - 1e-6, step)
    if len(anchors) != len(expected):
        raise ValueError(f"Expected {len(expected)} anchors, received {len(anchors)}")
    out = []
    for a, t in zip(anchors, expected):
        if not math.isfinite(float(a["t_s"])) or abs(float(a["t_s"]) - t) > 0.05:
            raise ValueError(f"Missing/out-of-order anchor at {t:.2f}s")
        box, focus = a["region"], a["focus"]
        if len(box) != 4 or len(focus) != 2:
            raise ValueError("Invalid region/focus dimensions")
        if not all(math.isfinite(float(v)) and 0 <= v <= 1 for v in box + focus):
            raise ValueError("Coordinates must be finite and normalized to [0,1]")
        x0, y0, x1, y1 = box
        if x0 >= x1 or y0 >= y1 or a["shot"] not in SHOT_TYPES:
            raise ValueError("Invalid box order or shot type")
        out.append({"t": start + float(t), "shot": a["shot"],
                    "region_kf": [x0 * W, y0 * H, x1 * W, y1 * H],
                    "focus_kf": {"x": focus[0] * W, "y": focus[1] * H},
                    "x_unc_kf": W * 0.01, "kf_W": W, "kf_H": H, "flags": [],
                    "focus_description": a["focus_description"], "uncertain": a["uncertain"]})
    return out


def check_video(path, expected_duration, dimensions=None):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                       check=True, text=True, capture_output=True)
    info = json.loads(r.stdout)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    if not any(s["codec_type"] == "audio" for s in info["streams"]):
        raise ValueError(f"Audio missing: {path.name}")
    if dimensions and (video["width"], video["height"]) != dimensions:
        raise ValueError(f"Unexpected output dimensions: {path.name}")
    duration = float(info["format"]["duration"])
    if abs(duration - expected_duration) > 0.25:
        raise ValueError(f"Incorrect output duration: {duration} vs {expected_duration}")
    # Full decode of the short output catches encoder failures beyond just metadata.
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", str(path), "-f", "null", "-"],
                   check=True, capture_output=True)
    return {"path": str(path.resolve()), "duration_s": duration, "bytes": path.stat().st_size,
            "width": video["width"], "height": video["height"]}


def extract_clip(video, start, duration, path):
    if not path.exists():
        tmp = path.with_name(path.stem + ".partial.mp4")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                        "-ss", str(start), "-i", str(video), "-t", str(duration),
                        "-map", "0:v:0", "-map", "0:a:0", "-vf", "scale=1280:720",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(tmp)], check=True)
        check_video(tmp, duration, (1280, 720))
        tmp.replace(path)


def diagnostics(plan):
    velocities = []
    x = np.asarray(plan["cx"]) / plan["W"]
    times = plan["clip"][0] + np.arange(len(x)) / plan["fps"]
    for shot in plan["shots"]:
        v = x[(times >= shot["start"]) & (times < shot["end"])]
        velocities.extend((abs(np.diff(v)) * plan["fps"]).tolist())
    return {"within_shot_crop_speed_p95_source_width_per_s": float(np.percentile(velocities, 95)) if velocities else 0,
            "human_quality_evaluation": "not_performed", "note": "Motion metric is not a correctness score"}


def api_cost_estimate(usage, model):
    if model != "gemini-4-hydrogen":
        return None
    inp = usage.get("total_input_tokens", 0)
    tools = usage.get("total_tool_use_tokens", 0)
    out = usage.get("total_output_tokens", 0)
    thought = usage.get("total_thought_tokens", 0)
    cached = usage.get("total_cached_tokens", 0)
    if inp + tools + out + thought != usage.get("total_tokens", 0):
        return {"usd": None, "reason": "Usage totals do not match documented disjoint categories"}
    return {"usd": ((inp + tools - cached) * 1.5 + cached * 0.075 + (out + thought) * 7.5) / 1e6,
            "basis": "Standard paid-tier list-price estimate; not account invoice, excludes local compute",
            "price_checked": "2026-11-14", "price_source": "https://aistudio.google.com/docs/pricing#gemini-4-hydrogen",
            "input_usd_per_million": 1.5, "cached_usd_per_million": 0.075,
            "output_and_thinking_usd_per_million": 7.5}


def run(args):
    video = Path(args.video).resolve()
    run_dir = ROOT / "runs" / args.name
    started = time.monotonic()
    info = media.probe(video)
    config = {"video": str(video), "sha256": digest(video), "model": args.model,
              "event_processing": args.processing, "crop_processing": "static",
              "youtube_url": args.youtube_url,
              "max_events": args.max_events, "max_clip_s": args.max_clip,
              "source_offset_s": args.source_offset, "prompt_version": PROMPT_VERSION,
              "crop": "full-height-9:16", "anchor_step_s": 0.5,
              "code_sha256": {name: digest(ROOT / "reframe" / name) for name in
                              ("gemini.py", "gemini_baseline.py", "pipeline.py", "smooth.py", "geometry.py", "media.py")}}
    manifest = run_dir / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("Run configuration or source changed; choose a new --name")
    save(manifest, config)
    g = Gemini(run_dir, args.model)
    full = {"uri": args.youtube_url} if args.youtube_url else g.upload(video, "match")
    event_processing = ({"type": "static", "start_offset": f"{args.source_offset}s",
                         "end_offset": f"{args.source_offset + info['duration']}s"}
                        if args.youtube_url else args.processing)
    event_prompt = (
        f"You are a football video editor. Watch this entire {info['duration']:.3f}-second continuous broadcast "
        f"with audio. Select up to {args.max_events} DISTINCT best highlight events. Do not invent events. "
        f"Each interval must be 5 to {args.max_clip} seconds, include buildup and outcome, and not overlap another. "
        "Treat replay and live footage of the same action as one event. Use only visible/audible evidence from "
        "this upload, not remembered match results. Prefer important goals, chances, dribbles or cards. "
        "start_s/end_s/peak_s are numeric seconds from the START OF THIS VIDEO, never the scoreboard clock. "
        "Include concise visual and audio evidence; use unknown if audio is unclear. You may return no events."
    )
    if args.youtube_url:
        event_prompt += (f" IMPORTANT TIMELINE OVERRIDE: this is a clipped portion of a longer YouTube video. "
                         f"Use ORIGINAL YOUTUBE playback seconds for start_s/end_s/peak_s, all between "
                         f"{args.source_offset} and {args.source_offset + info['duration']}. "
                         "Do not reset them to zero. Ignore the match scoreboard clock.")
    events_path = run_dir / "events.json"
    if events_path.exists():
        events = validate_events({"events": json.loads(events_path.read_text())}, info["duration"], args.max_events, args.max_clip)
    else:
        data = g.ask(full, event_prompt, EVENT_SCHEMA, "events", event_processing)
        if args.youtube_url:
            for event in data["events"]:
                for field in ("start_s", "end_s", "peak_s"):
                    event[field] = float(event[field]) - args.source_offset
        events = validate_events(data, info["duration"], args.max_events, args.max_clip)
        save(events_path, events)
    print(f"[events] {len(events)} selected", flush=True)
    results = []
    for ei, ev in enumerate(events):
        prefix = f"event_{ei:02d}"
        start, end = ev["clip"]
        duration = end - start
        print(f"[event] {ei}: {start:.1f}–{end:.1f}s {ev['event']}", flush=True)
        clip_path = run_dir / f"{prefix}_input.mp4"
        extract_clip(video, start, duration, clip_path)
        file = g.upload(clip_path, prefix)
        ts = [round(float(t), 3) for t in np.arange(0, duration - 1e-6, 0.5)]
        crop_prompt = (
            "You are the camera director reframing this football clip into a full-height 9:16 crop. "
            "Determine the action focus yourself, following ball/receiver/goal during play and the relevant "
            "person during reactions or closeups. Use temporal context; do not lock onto one player throughout. "
            "Return exactly one anchor for EACH timestamp in this list, in order: " + json.dumps(ts) + ". "
            "All timestamps are local seconds from the start of THIS SHORT CLIP. "
            "region is [left,top,right,bottom] of the must-keep action, normalized to [0,1] of the full image; "
            "focus is [x,y] of the most important visible point if the region is too wide for a vertical crop. "
            "Do not output a ball location based on guessing when it is invisible; set uncertain=true and use "
            "a conservative relevant region. Keep focus stable within shots but reset after camera cuts. "
            "Use shot types wide, medium, closeup, crowd or graphic. Describe the focus briefly. "
            f"Event hypothesis to verify visually, not assume: {ev['description']}"
        )
        kf_path = run_dir / f"{prefix}_keyframes.json"
        if kf_path.exists():
            kfs = json.loads(kf_path.read_text())
        else:
            data = g.ask(file, crop_prompt, ANCHOR_SCHEMA, prefix + "_anchors")
            try:
                kfs = validate_anchors(data, duration, start, info["W"], info["H"])
            except (ValueError, KeyError, TypeError) as e:
                repair = crop_prompt + f" Previous response failed validation: {e}. Produce a corrected full response."
                data = g.ask(file, repair, ANCHOR_SCHEMA, prefix + "_anchors_repair")
                kfs = validate_anchors(data, duration, start, info["W"], info["H"])
            save(kf_path, kfs)
        cut_path = run_dir / f"{prefix}_cuts.json"
        if cut_path.exists():
            cuts = json.loads(cut_path.read_text())
        else:
            cuts = media.shot_cuts(video, start, duration)
            save(cut_path, cuts)
        plan = plan_clip({**info, "cuts": cuts}, ev, kfs, min_h_frac=1.0)
        save(run_dir / f"{prefix}_plan.json", plan)
        out, debug = run_dir / f"{prefix}_vertical.mp4", run_dir / f"{prefix}_debug.mp4"
        done_path = run_dir / f"{prefix}_complete.json"
        if not done_path.exists():
            render(video, plan, out, debug, kfs)
            checked = {"vertical": check_video(out, duration, (720, 1280)),
                       "debug": check_video(debug, duration)}
            save(done_path, checked)
        else:
            checked = json.loads(done_path.read_text())
            for item in checked.values():
                if not Path(item["path"]).exists():
                    raise ValueError("Completed output missing; remove its complete.json marker and resume")
        results.append({"event": ev, "original_video_interval_s": [start + args.source_offset, end + args.source_offset],
                        **checked, "diagnostics": diagnostics(plan)})
        print(f"[rendered] {out}", flush=True)
    usage = {}
    for p in run_dir.glob("*.response.json"):
        r = json.loads(p.read_text())
        for k, v in (r.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v
    report = {"status": "complete", "config": config, "results": results,
              "invocation_elapsed_s": time.monotonic() - started, "usage_totals": usage,
              "api_cost_estimate": api_cost_estimate(usage, args.model),
              "cost_note": "Preflight requests and local compute are separate; this is not a complete production cost",
              "quality_note": "Technical checks passed; no human ground-truth or quality scoring yet"}
    save(run_dir / "report.json", report)
    print(f"[complete] {run_dir / 'report.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="gemini-4-hydrogen")
    ap.add_argument("--processing", choices=["static", "agentic"], default="agentic")
    ap.add_argument("--max-events", type=int, default=3)
    ap.add_argument("--max-clip", type=float, default=45)
    ap.add_argument("--source-offset", type=float, default=0)
    ap.add_argument("--youtube-url", default="", help="Official source URL; requires static mode and source-offset")
    args = ap.parse_args()
    if args.max_events < 1 or not 5 <= args.max_clip <= 90:
        ap.error("max-events must be positive; max-clip must be between 5 and 90 seconds")
    if args.youtube_url and args.processing != "static":
        ap.error("YouTube interval clipping requires --processing static")
    run(args)


if __name__ == "__main__":
    main()
