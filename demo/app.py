"""Minimal English interface for main's s3 → plan_clip → render workflow."""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
import requests
import streamlit as st

from demo import backend
from reframe import pipeline

st.set_page_config(page_title="Drex Vertical Reframe - Key Region Detection", page_icon="◧", layout="wide", initial_sidebar_state="collapsed")
st.html("""<style>
.stApp{background:#f8f9f7;color:#19241f}
.block-container{max-width:1250px;padding:2.1rem 2rem 2rem}
h1{font-size:1.7rem!important;font-weight:600!important;letter-spacing:-.045em;margin-bottom:0!important}
h2,h3{font-size:.95rem!important;font-weight:550!important;letter-spacing:-.02em}
[data-testid="stHeader"]{display:none}
[data-testid="stMainBlockContainer"]{padding-top:2rem}
div[data-testid="stVerticalBlock"]{gap:.75rem}
button[kind="primary"]{background:#285f45;border:1px solid #285f45;border-radius:7px}
[data-testid="stImage"] img,video{border-radius:7px}
[data-testid="stCaptionContainer"]{font-size:12px;color:#7c867e}
[data-testid="stExpander"]{background:transparent;border-color:#e1e5df;border-radius:7px}
[data-testid="stExpander"] p{font-size:12px}
.empty{aspect-ratio:9/16;width:100%;max-width:240px;background:#edf0ea;border:1px solid #e3e8df;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#a5b09f;font-size:20px;font-weight:300}
.legend{display:flex;gap:16px;font-size:11px;color:#7a867e;margin-top:4px}.legend i{display:inline-block;width:6px;height:6px;border-radius:2px;background:#e0c500;margin-right:5px}.legend .cyan{background:#00bcd8}
[role="radio"][aria-checked="true"]{color:#285f45!important;border-color:#99b7a0!important;background:#e8f0e6!important}
@media(max-width:700px){.block-container{padding:1.4rem 1rem}h1{font-size:1.4rem!important}}
</style>""")


def timecode(t):
    return f"{int(t) // 60:02d}:{t % 60:04.1f}"


@st.cache_data(show_spinner=False, max_entries=4)
def get_frames(scene, start, mtime):
    return [(t, backend.png(image)) for t, image in backend.extract_batch(scene, start)]


def set_goal():
    value = backend.PRESETS[st.session_state.goal_preset]
    if value is not None:
        st.session_state.goal = value


st.title("Drex Vertical Reframe - Key Region Detection")

source_col, goal_col, action_col = st.columns([2.1, 2.1, 1], vertical_alignment="bottom")
with source_col:
    clip_name = st.selectbox("Clip", list(backend.CLIPS), key="clip")
with goal_col:
    st.selectbox("Focus", list(backend.PRESETS), key="goal_preset", on_change=set_goal)
with action_col:
    run = st.button("Run detection", type="primary", key="run", width="stretch")

start = 0.0
if "goal" not in st.session_state:
    st.session_state.goal = ""
goal = st.text_input("Viewing goal", key="goal", placeholder="Optional viewing goal", max_chars=2000, label_visibility="collapsed")

try:
    scene = backend.load_scene(backend.CLIPS[clip_name])
    with st.spinner("Loading frames…"):
        encoded = get_frames(scene, start, scene["video"].stat().st_mtime_ns)
    frames = [(t, Image.open(io.BytesIO(data)).convert("RGB")) for t, data in encoded]
    prepared = pipeline.prepare_keyframe_batch(frames, goal or None)
except (ValueError, OSError, RuntimeError) as error:
    st.error(backend.safe_error(error))
    st.stop()

signature = hashlib.sha256(json.dumps([clip_name, goal, scene["video"].stat().st_mtime_ns]).encode()).hexdigest()
if st.session_state.get("signature") != signature:
    for key in ("result", "error", "partial_response", "events"):
        st.session_state.pop(key, None)
    st.session_state.signature = signature
    st.session_state.frame = 0
    st.session_state.view = "Frames"
if st.session_state.get("frame") is None:
    st.session_state.frame = 0

if run:
    for key in ("result", "error", "partial_response"):
        st.session_state.pop(key, None)
    st.session_state.events = []
    with st.status("Querying NACE…", expanded=False) as status:
        def observe(event, data):
            st.session_state.events.append(event)
            if event == "prepared":
                st.write(f"{len(frames)} frames · {len(data['questions'])} Choice questions · 1 request")
            elif event == "response":
                st.session_state.partial_response = data
                status.update(label="Decoding regions…")
            elif event == "planning":
                status.update(label="Planning the crop…")
            elif event == "rendering":
                status.update(label="Rendering 9:16…")
        try:
            st.session_state.result = backend.execute(scene, frames, goal, prepared, on_event=observe)
            status.update(label="Complete", state="complete", expanded=False)
        except (RuntimeError, ValueError, OSError, KeyError, requests.RequestException) as error:
            st.session_state.error = backend.safe_error(error)
            status.update(label="Run failed", state="error", expanded=False)

result = st.session_state.get("result")
if st.session_state.get("error"):
    st.error(st.session_state.error)

view_col, info_col = st.columns([3, 2], vertical_alignment="center")
with view_col:
    view = st.segmented_control("View", ["Frames", "Model input", "Video"], key="view", label_visibility="collapsed") or "Frames"
with info_col:
    elapsed = f" · {result['response'].get('_latency_s', 0):.2f}s model" if result else ""
    st.caption(f"8 frames · {len(prepared['questions'])} choices · 1 request{elapsed}")

if view == "Video" and result:
    left, right = st.columns([3.5, 1], gap="medium")
    with left:
        st.caption("Source + crop")
        st.video(result["debug"])
    with right:
        st.caption("9:16 output")
        st.video(result["vertical"])
elif view == "Video":
    st.info("Run detection to generate the vertical clip.")
    st.image(frames[0][1], width="stretch")
else:
    index = st.session_state.get("frame", 0)
    image = frames[index][1]
    source = prepared["images"][index] if view == "Model input" else backend.overlay(image, result, index) if result else image
    left, right = st.columns([4, 1.15], gap="medium")
    with left:
        st.caption("NACE input" if view == "Model input" else "Source")
        st.image(source, width="stretch")
        if result and view == "Frames":
            st.html('<div class="legend"><span><i></i>Region + focus</span><span><i class="cyan"></i>Final crop</span></div>')
    with right:
        st.caption("9:16 output")
        if result:
            portrait, _ = backend.frame_crop(image, result, index)
            st.image(portrait, width="stretch")
        else:
            st.html('<div class="empty">9:16</div>')
    st.segmented_control("Frame", list(range(len(frames))), format_func=lambda i: timecode(frames[i][0] + scene.get("scene_start", 0)),
                         key="frame", selection_mode="single", label_visibility="collapsed")

if result:
    save_video, save_frame, save_data, _ = st.columns([1.4, 1.4, 1.4, 3])
    with save_video:
        st.download_button("Video", Path(result["vertical"]).read_bytes(), "scene1-vertical.mp4", "video/mp4", width="stretch")
    with save_frame:
        portrait, _ = backend.frame_crop(frames[st.session_state.get("frame", 0)][1], result, st.session_state.get("frame", 0))
        st.download_button("Frame", backend.png(portrait), "scene1-portrait.png", "image/png", width="stretch")
    with save_data:
        st.download_button("Run JSON", (Path(result["directory"]) / "run.json").read_bytes(), "scene1-run.json", "application/json", width="stretch")

with st.expander("Inspect pipeline"):
    request_tab, response_tab, region_tab, plan_tab = st.tabs(["Request", "Response", "Region", "Camera path"])
    with request_tab:
        st.caption("POST /v1/systemone · drex-max-v1.1")
        st.json(backend.display_request(prepared), expanded=False)
        st.download_button("Request JSON", json.dumps(backend.body_for(prepared)), "scene1-request.json", "application/json")
        idx = st.session_state.get("frame", 0)
        sent = base64.b64decode(backend.body_for(prepared)["images"][idx].split(",", 1)[1])
        st.download_button("Input JPEG", sent, f"image-{idx + 1}.jpg", "image/jpeg")
    with response_tab:
        response = result["response"] if result else st.session_state.get("partial_response")
        if response:
            st.json(response, expanded=False)
        else:
            st.caption("No response yet.")
    with region_tab:
        if result:
            idx = st.session_state.get("frame", 0)
            selected = st.radio("Question", ["left", "right", "top", "bottom", "focus_x", "focus_y", "shot"], horizontal=True)
            answer = result["response"]["answers"][f"f{idx + 1}_{selected}"]
            rows = [{"Label": label, "Probability": probability} for label, probability in answer.get("probabilities", {}).items()]
            labels = list(answer.get("probabilities", {}))
            st.vega_lite_chart(rows, {"height": 170, "mark": {"type": "bar", "color": "#48765b"}, "encoding": {
                "x": {"field": "Label", "type": "ordinal", "sort": labels, "axis": {"labelAngle": 0}},
                "y": {"field": "Probability", "type": "quantitative", "scale": {"domain": [0, 1]}},
                "tooltip": [{"field": "Label"}, {"field": "Probability", "format": ".4f"}]}}, width="stretch")
            st.json(result["decoded"][idx], expanded=False)
            st.caption("main decoder: subtract the median probability floor, then compute expected coordinates and uncertainty.")
        else:
            st.caption("No region yet.")
    with plan_tab:
        if result:
            plan = result["plan"]
            rows = [{"Time": plan["clip"][0] + i / plan["fps"], "Crop center X": value} for i, value in enumerate(plan["cx"])]
            st.vega_lite_chart(rows, {"height": 170, "mark": {"type": "line", "color": "#00a9c4"}, "encoding": {
                "x": {"field": "Time", "type": "quantitative", "scale": {"zero": False}},
                "y": {"field": "Crop center X", "type": "quantitative", "scale": {"domain": [0, scene["W"]]}}}}, width="stretch")
            st.json({"cuts": plan["cuts"], "shots": plan["shots"], "params": plan["params"]}, expanded=False)
        else:
            st.caption("No camera path yet.")
    st.caption("main · prepare_keyframe_batch → detect_keyframe_batch → plan_clip → render")
    if result:
        st.code(f"base: {result['pipeline_revision']}\nrequest: {result['request_sha256']}\nrun: {result['directory']}", language=None)
