"""Scene1 adapters. Detection, camera planning and video rendering use main's code."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4

from PIL import Image, ImageDraw
import requests

from reframe import media, nace, pipeline
from reframe.geometry import box_xyxy

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "demo" / "assets"
PRESETS = {
    "Match context": "",
    "Ball & players": "Keep the ball, its carrier and the nearest involved defenders in view.",
    "Passing options": "Emphasize the ball and nearby visible passing options without inventing an intended pass.",
    "Goal area": "Prioritize visible goalmouth action, including the ball and goalkeeper. If the goal is offscreen, follow the visible play.",
    "Custom": None,
}
CLIPS = {"Wing play · 00:12": "wing", "Wide play · 00:32": "wide", "Touchline · 01:00": "touchline",
         "Goal area · 02:00": "goal", "Midfield · 05:00": "midfield", "Goal view · 10:00": "goal-view"}


def load_scene(clip_id="wing", assets_root=None):
    """Read a repository-bundled four-second clip. No original video or Git lookup."""
    directory = Path(assets_root or ASSETS).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    clip = next((item for item in manifest["clips"] if item["id"] == clip_id), None)
    if clip is None:
        raise ValueError("Unknown bundled clip.")
    video = (directory / clip["file"]).resolve()
    if not video.is_relative_to(directory) or not video.is_file():
        raise ValueError("Bundled clip missing. Clone the portable demo branch with its assets.")
    return {"video": video, "W": clip["W"], "H": clip["H"], "fps": clip["fps"],
            "dur": clip["duration"], "start": 0.0, "cuts": clip["cuts"], "scene_start": clip["scene_start"],
            "source_start": clip["source_start"], "clip_id": clip_id, "pipeline_base": manifest["pipeline_base"]}


def load_credentials():
    """Use this clone's .env or an environment variable; never read another checkout."""
    if not os.environ.get("NACE_API_KEY"):
        from dotenv import dotenv_values
        config = dotenv_values(ROOT / ".env")
        if config.get("NACE_API_KEY"):
            os.environ["NACE_API_KEY"] = config["NACE_API_KEY"]


def extract_batch(scene, start):
    if isinstance(start, bool) or not isinstance(start, (int, float)) or not math.isfinite(start):
        raise ValueError("Start time must be a finite number.")
    duration = pipeline.BATCH * pipeline.KF_STEP
    if not 0 <= start <= float(scene["dur"]) - duration:
        raise ValueError("Choose a start time with four seconds remaining in the clip.")
    frames = media.grab_many(scene["video"], float(start), duration, pipeline.KF_STEP, width=1280)
    if len(frames) != pipeline.BATCH:
        raise ValueError("Could not decode a complete eight-frame batch.")
    return frames


def png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def body_for(prepared):
    return {"model": "drex-max-v1.1", "state": prepared["state"], "questions": prepared["questions"],
            "images": [nace.image_to_b64(image) for image in prepared["images"]]}


def digest(prepared):
    return hashlib.sha256(json.dumps(body_for(prepared)).encode()).hexdigest()


def display_request(prepared):
    body = body_for(prepared)
    body["images"] = [{"image": i + 1, "jpeg_bytes": len(base64.b64decode(url.split(",", 1)[1])),
                       "sha256": hashlib.sha256(base64.b64decode(url.split(",", 1)[1])).hexdigest()}
                      for i, url in enumerate(body["images"])]
    return body


def safe_error(error):
    message = str(error)
    key = os.environ.get("NACE_API_KEY")
    if key:
        message = message.replace(key, "[redacted]")
    if isinstance(error, requests.RequestException):
        return "Could not reach NACE. Check the connection and try again."
    return message[:1200]


def execute(scene, frames, goal, prepared, on_event=None, run_root=None):
    """NACE s3 → main plan_clip → main render, with one inspectable run record."""
    load_credentials()
    target = Path(run_root or ROOT / "runs" / "region_demo") / uuid4().hex[:12]
    target.mkdir(parents=True)
    started = time.perf_counter()
    def observe(event, data):
        if event == "prepared" and digest(data) != digest(prepared):
            raise ValueError("Preview and actual request differ; no request was sent.")
        if on_event:
            on_event(event, data)
    detected = pipeline.detect_keyframe_batch(frames, goal or None, on_event=observe)
    # Preserve successful NACE evidence even if planning or encoding later fails.
    (target / "request.json").write_text(json.dumps(body_for(detected["prepared"])))
    (target / "response.json").write_text(json.dumps(detected["response"], ensure_ascii=False, indent=2))
    (target / "s3.json").write_text(json.dumps({"0": detected["keyframes"]}, indent=2))
    for index, (_, image) in enumerate(frames):
        image.save(target / f"input-{index + 1:02d}.png")
    start = frames[0][0]
    event = {"clip": [start, start + pipeline.BATCH * pipeline.KF_STEP]}
    if on_event:
        on_event("planning", None)
    plan = pipeline.plan_clip(scene, event, detected["keyframes"])
    (target / "plan.json").write_text(json.dumps(plan))
    vertical, debug = target / "vertical.mp4", target / "debug.mp4"
    if on_event:
        on_event("rendering", None)
    pipeline.render(scene["video"], plan, vertical, debug, detected["keyframes"])
    info = media.probe(vertical)
    if info["W"] != 720 or info["H"] != 1280 or abs(info["duration"] - 4) > 0.15:
        raise ValueError("Rendered output did not pass the 720 × 1280 / four-second check.")
    revision_id = scene.get("pipeline_base", "source-export")
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        if revision.returncode == 0:
            revision_id = revision.stdout.strip()
    except OSError:
        pass  # A source archive can run the demo without Git installed.
    result = {**detected, "plan": plan, "directory": str(target), "vertical": str(vertical), "debug": str(debug),
              "total_s": time.perf_counter() - started, "goal": goal,
              "source_offset": scene["source_start"], "request_sha256": digest(prepared),
              "pipeline_revision": revision_id}
    serializable = {k: v for k, v in result.items() if k != "prepared"}
    (target / "run.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2))
    return result


def frame_crop(image, result, index):
    plan = result["plan"]
    relative = result["keyframes"][index]["t"] - plan["clip"][0]
    frame = min(round(relative * plan["fps"]), len(plan["cx"]) - 1)
    box = box_xyxy(plan["cx"][frame], plan["cy"][frame], plan["h"][frame])
    # Same rounding, crop and resize as pipeline.render().
    portrait = image.crop([round(v) for v in box]).resize((720, 1280), Image.Resampling.LANCZOS)
    return portrait, box


def overlay(image, result, index):
    output = image.copy()
    draw = ImageDraw.Draw(output)
    decoded = result["decoded"][index]
    draw.rectangle(decoded["region"], outline=(255, 230, 0), width=3)
    _, box = frame_crop(image, result, index)
    draw.rectangle(box, outline=(0, 220, 255), width=4)
    focus = decoded.get("focus")
    if focus:
        x, y = focus["x"], focus["y"]
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(255, 230, 0), outline=(0, 0, 0), width=2)
    return output
