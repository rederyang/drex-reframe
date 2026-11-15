"""Case studies: two recorded end-to-end runs with every stage's input, configuration and model output.

The synced replay plays the debug view (source + crop), the Drex vertical and the Gemini vertical
together, overlays the 16 × 9 grid Drex reads, and shows Drex's answers for the keyframe on screen.
Cases are recorded by scripts/record_demo_case.py into demo/assets/cases/<id>/.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "demo" / "assets" / "cases"
ORDER = [("football-goal-replay", "Football · goal replay & celebration"), ("interview-gigawatt", "Interview · follow the speaker")]
SHOT = {"wide": "Wide", "medium": "Medium", "closeup": "Close-up", "crowd": "Crowd", "graphic": "Graphic",
        "two_shot": "Two-shot", "single": "Single", "group_wide": "Group wide", "other": "Other"}

st.html("""<style>
.stApp{background:#f6f8f6;color:#18231e}
.block-container{max-width:1320px;padding:3.4rem 2rem 3rem}
h1{font-size:1.75rem!important;font-weight:650!important;letter-spacing:-.03em}
h2{font-size:1.15rem!important;font-weight:620!important;margin-top:.6rem!important}
h3{font-size:.98rem!important;font-weight:600!important}
[data-testid="stMetricValue"]{font-size:1.35rem}
[data-testid="stMetricLabel"] p{font-size:.78rem;color:#5d6b63}
.k{font-family:ui-monospace,Menlo,monospace;font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:#6b7a71}
.io{border:1px solid #dfe5df;border-radius:8px;background:#fff;padding:.7rem .9rem}
.tag{display:inline-block;font-size:.72rem;padding:.08rem .5rem;border-radius:999px;background:#e6efe8;color:#245b40;margin:0 .25rem .25rem 0}
.why li{margin:.15rem 0}
</style>""")


@st.cache_data(show_spinner=False)
def load_case(cid: str) -> dict:
    return json.loads((CASES / cid / "case.json").read_text())


@st.cache_data(show_spinner=False)
def b64(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    mime = {"mp4": "video/mp4", "jpg": "image/jpeg", "png": "image/png"}[p.suffix[1:]]
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def tags(items):
    st.html("".join(f'<span class="tag">{i}</span>' for i in items))


def kv(d: dict):
    st.dataframe([{"setting": k, "value": json.dumps(v) if isinstance(v, (list, dict)) else str(v)} for k, v in d.items()],
                 hide_index=True, width="stretch")


def bars(probs: dict, height=150, color="#3f7a59", sort=None):
    rows = [{"label": k, "p": v} for k, v in probs.items()]
    st.vega_lite_chart(rows, {"height": height, "mark": {"type": "bar", "color": color},
                              "encoding": {"x": {"field": "label", "type": "ordinal", "sort": sort or list(probs), "axis": {"labelAngle": 0, "title": None}},
                                           "y": {"field": "p", "type": "quantitative", "title": "probability"},
                                           "tooltip": [{"field": "label"}, {"field": "p", "format": ".3f"}]}}, width="stretch")


# ------------------------------------------------------------------ page
available = [(cid, name) for cid, name in ORDER if (CASES / cid / "case.json").exists()]
if not available:
    st.warning("No recorded cases. Run scripts/record_demo_case.py first.")
    st.stop()

st.html('<div class="k">Drex Vertical Reframe · case studies</div>')
st.title("How a long landscape video becomes a vertical short")

# ------------------------------------------------------------------ why it matters (same block as the results page)
WHY = ROOT / "demo" / "assets" / "why"
why_media = {}
for key, stem in (("FOOTBALL_SOURCE", "football_source"), ("FOOTBALL_DREX", "football_drex"),
                  ("TALK_SOURCE", "talk_source"), ("TALK_DREX", "talk_drex")):
    if (WHY / f"{stem}.mp4").exists():
        why_media[key] = b64(str(WHY / f"{stem}.mp4"))
        why_media[key + "_POSTER"] = b64(str(WHY / f"{stem}.jpg"))
if why_media:
    why_html = Path(__file__).with_name("why_examples.html").read_text().replace("__MEDIA__", json.dumps(why_media))
    why_h = 1095 if "TALK_SOURCE" in why_media else 760
    if hasattr(st, "iframe"):
        st.iframe(why_html, height=why_h)
    else:
        components.html(why_html, height=why_h, scrolling=False)
    st.divider()
st.markdown("## Inside the pipeline: two recorded cases")
choice = st.segmented_control("Case", [n for _, n in available], default=available[0][1], key="case_pick",
                              label_visibility="collapsed") or available[0][1]
cid = next(c for c, n in available if n == choice)
case = load_case(cid)
cdir = CASES / cid
talk = case["scenario"] == "talk"

st.subheader(case["title"])
st.caption(case["subtitle"] + f"  ·  recorded {case['recorded_at']} (Drex was re-run on this clip to keep every raw answer)")

with st.expander("What is new compared with existing auto-reframe", expanded=False):
    st.html("""<ul class="why">
<li><b>Existing auto-reframe</b> (e.g. Premiere Pro Auto Reframe, Firefly Auto reframe) follows the most salient subject.</li>
<li><b>New here: framing follows intent.</b> Drex decides what matters from the picture, the commentary or transcript, and who is speaking.</li>
<li><b>Locating becomes multiple choice.</b> Drex has no box output, so it picks numbered grid lines; its probabilities become confidence weights for the camera.</li>
<li><b>Camera language fits the content:</b> a follow-cam that never loses the ball for sport, locked shots with hard cuts for talk.</li>
</ul>""")

# headline numbers
T, C = case["totals"], case["compare"]
g = C.get("gemini") or {}
m = st.columns(6)
m[0].metric("Clip", f"{case['clip']['duration']:.0f} s")
m[1].metric("Keyframes read by Drex", T["keyframes"], help="one frame every 0.5 s, 8 per request")
m[2].metric("Drex calls · cost", f"{T['drex_calls']} · ${T['drex_usd']:.4f}")
m[3].metric("Drex per-frame decisions", f"{T['s3_latency_s']:.0f} s", help="sum of the s3 request latencies")
m[4].metric(f"Gemini ({g.get('model', 'n/a')})", f"${g.get('usd', 0):.3f}" if g else "–", help="same clip, same planner; includes upload")
m[5].metric("Gemini decision time", f"{g.get('time_s', 0):.0f} s" if g else "–")

# ------------------------------------------------------------------ synced replay
st.markdown("## Synced replay: model output follows the playhead")
st.caption("The debug view shows the source with Drex's keep region (yellow) and the final 9:16 crop (blue). "
           "The faint grid is exactly what Drex reads; the lines it picked for the current keyframe light up. "
           "The panel shows Drex's raw answers for the keyframe on screen. Click the timeline to seek.")

kf_imgs = {r["img"]: b64(str(cdir / r["img"])) for r in case["keyframes"]}
payload = {"talk": talk, "dur": case["clip"]["duration"], "W": case["clip"]["W"], "H": case["clip"]["H"],
           "kf": case["keyframes"], "path": case["path"], "cuts": case["cuts"],
           "turns": case["stages"]["s4"]["output"].get("speaker_turns", []), "shot": SHOT,
           # raw NDI ids (c0:SPEAKER_02) -> "Speaker A/B" in order of first appearance, matching the timeline lane
           "spk": {sid: f"Speaker {'ABC'[min(n, 2)]}" for n, sid in enumerate(dict.fromkeys(
               l["speaker"] for l in case["stages"]["s0"]["output"].get("sentence_lines", [])
               if l.get("speaker") and l["end_rel"] > 0 and l["t_rel"] < case["clip"]["duration"]))}}
html = (Path(__file__).with_name("case_player.html").read_text()
        .replace("__DATA__", json.dumps(payload))
        .replace("__IMGS__", json.dumps(kf_imgs))
        .replace("__DEBUG__", b64(str(cdir / "debug.mp4")))
        .replace("__DREX__", b64(str(cdir / "vertical.mp4")))
        .replace("__GEMINI__", b64(str(cdir / "gemini.mp4")))
        .replace("__GEMLABEL__", f"Gemini · {g.get('model', '')}"))
PLAYER_H = 965 if talk else 940
if hasattr(st, "iframe"):  # Streamlit >= 1.6x
    st.iframe(html, height=PLAYER_H)
else:
    components.html(html, height=PLAYER_H, scrolling=False)

# ------------------------------------------------------------------ stage by stage
st.markdown("## Pipeline, step by step")
st.caption("Each step's input, configuration and output for this clip. Model outputs are the raw answers from the recorded run.")
S = case["stages"]
tabs = st.tabs(["s0 · Structure" + (" (NDI)" if talk else ""), "s1 · Find the moment", "s3 · Drex per frame",
                "s4 · Camera path", "Render & compare"])

with tabs[0]:
    s0 = S["s0"]
    tags(s0["tools"])
    i, c, o = st.columns([1, 1, 2])
    with i:
        st.html('<div class="k">Input</div>')
        st.json(s0["input"], expanded=True)
    with c:
        st.html('<div class="k">Configuration</div>')
        kv(s0["config"])
    with o:
        st.html('<div class="k">Output</div>')
        if talk:
            st.markdown(f"**NDI raw segments** (chunk {s0['output']['ndi_chunk']}; whole 30 min cost ${s0['output']['ndi_total_usd']:.4f})")
            st.dataframe(s0["output"]["ndi_segments_raw"], hide_index=True, width="stretch", height=220)
            st.markdown("**Sentence lines** (Whisper timing + NDI speaker), seconds relative to the clip")
            st.dataframe(s0["output"]["sentence_lines"], hide_index=True, width="stretch", height=220)
            st.caption(s0["output"]["note"])
        else:
            cuts_in = s0["output"]["shot_cuts_in_clip_s"]
            st.markdown("**Shot cuts inside the clip:** " + (", ".join(f"{c:.1f} s" for c in cuts_in) if cuts_in else "none (one continuous shot)"))
            rows = [{"t": t, "dB": db} for t, db in s0["output"]["loudness_db"]]
            st.vega_lite_chart(rows, {"height": 130, "mark": {"type": "line", "color": "#c2410c"},
                                      "encoding": {"x": {"field": "t", "type": "quantitative", "title": "seconds from clip start"},
                                                   "y": {"field": "dB", "type": "quantitative", "title": "crowd loudness (dB)", "scale": {"zero": False}}}}, width="stretch")
            st.markdown("**Commentary transcript** (seconds from clip start)")
            st.dataframe(s0["output"]["commentary"], hide_index=True, width="stretch", height=220)

with tabs[1]:
    s1 = S["s1"]
    tags(s1["tools"])
    if talk:
        i, o = st.columns([3, 2])
        with i:
            st.html('<div class="k">Input · the candidate window, as numbered lines in Drex state</div>')
            st.dataframe(s1["input"]["window_lines"], hide_index=True, width="stretch", height=260)
            st.html('<div class="k">Configuration</div>')
            st.markdown(f"Window length {s1['config']['window_s']} s · score scale: " + " → ".join(s1["config"]["score_scale"]))
            for q in s1["config"]["edge_questions"]:
                st.caption(f"**{q['id']}** (choice over {len(q['options'])} lines): {q['instructions']}")
        with o:
            st.html('<div class="k">Output</div>')
            out = s1["output"]
            st.metric("Drex score for this window", f"{out['window_score']:.2f} / 4", f"rank {out['rank']} of {out['of']} (median {out['score_p50']:.2f})")
            for k in ("start", "end"):
                e = out["edges"][k]
                st.markdown(f"**{k.title()} line → {e['choice']}**")
                bars(e["p"], height=120)
    else:
        st.html('<div class="k">Input · 8 frames around the candidate + nearby commentary</div>')
        cols = st.columns(8)
        for col, f in zip(cols, s1["input"]["frames"]):
            col.image(str(cdir / f["img"]), caption=f"{f['t_rel']:+.0f} s", width="stretch")
        st.caption(f"Recall score {s1['input']['recall_score']} (keywords + crowd noise, z = {s1['input']['crowd_noise_z']}). State: {s1['input']['state']}")
        c, o = st.columns([1, 1])
        with c:
            st.html('<div class="k">Configuration · questions</div>')
            for q in s1["config"]["questions"]:
                st.caption(f"**{q['id']}** ({q['type']}, {len(q['options'])} options): {q['instructions']}")
            st.caption(s1["config"]["clip_rule"])
        with o:
            st.html('<div class="k">Output</div>')
            a = s1["output"]["answers"]
            st.markdown(f"**Event → {a['event']['choice']}** · excitement **{a['excitement']['score']:.2f} / 4** · peak **{a['peak']['choice']}** "
                        f"· {s1['output']['latency_s']} s · ${s1['output']['usd']:.5f}")
            bars(a["event"]["p"], height=140)

with tabs[2]:
    s3 = S["s3"]
    tags(s3["tools"])
    c, b = st.columns([1, 1])
    with c:
        st.html('<div class="k">Configuration</div>')
        kv({k: v for k, v in s3["config"].items() if k != "question_template"})
    with b:
        st.html('<div class="k">Requests</div>')
        st.dataframe([{k: v for k, v in x.items() if k != "state"} for x in s3["batches"]], hide_index=True, width="stretch")
    st.html('<div class="k">Questions asked about every frame (shown for Image 1)</div>')
    st.dataframe([{"id": q["id"], "type": q["type"], "options": ", ".join(map(str, q["options"])) if isinstance(q["options"], list) else q["options"],
                   "instructions": q["instructions"]} for q in s3["config"]["question_template"]], hide_index=True, width="stretch")
    with st.expander("Drex state for the first request"):
        st.code(s3["batches"][0]["state"], language=None)
    st.html('<div class="k">Inspect one keyframe</div>')
    idx = st.slider("Keyframe", 0, len(case["keyframes"]) - 1, len(case["keyframes"]) // 2, format="%d", label_visibility="collapsed")
    r = case["keyframes"][idx]
    left, right = st.columns([3, 2])
    with left:
        st.image(str(cdir / r["img"]), caption=f"Model input at {r['t_rel']:.1f} s (Image {r['image_k']} of request {r['batch'] + 1})", width="stretch")
    with right:
        qsel = st.radio("Question", list(r["answers"]), horizontal=True, key="qsel")
        ans = r["answers"][qsel]
        st.markdown(f"**{qsel} → {ans['choice']}** (confidence {ans['confidence']})")
        bars(ans["p"], height=140)
        st.json({"decoded": r["decoded"], "final_crop": r["crop"], "pinned": r.get("pinned")}, expanded=False)

with tabs[3]:
    s4 = S["s4"]
    tags(s4["tools"])
    c, o = st.columns([1, 2])
    with c:
        st.html('<div class="k">Configuration</div>')
        kv(s4["config"])
    with o:
        st.html('<div class="k">Output</div>')
        rows = [{"t": t, "crop center (0 = left, 1 = right)": x} for t, x, _ in case["path"]]
        pts = [{"t": r_["t_rel"], "Drex keep-region center": (r_["decoded"]["region_px"][0] + r_["decoded"]["region_px"][2]) / 2 / case["clip"]["W"]}
               for r_ in case["keyframes"]]
        st.vega_lite_chart({"layer": [
            {"data": {"values": rows}, "mark": {"type": "line", "color": "#0a8fb0", "strokeWidth": 3},
             "encoding": {"x": {"field": "t", "type": "quantitative", "title": "seconds"},
                          "y": {"field": "crop center (0 = left, 1 = right)", "type": "quantitative", "scale": {"domain": [0, 1]}}}},
            {"data": {"values": pts}, "mark": {"type": "point", "color": "#b88a00", "filled": True},
             "encoding": {"x": {"field": "t", "type": "quantitative"}, "y": {"field": "Drex keep-region center", "type": "quantitative"}}}],
            "height": 220}, width="stretch")
        st.json(s4["output"], expanded=False)

with tabs[4]:
    st.markdown(S["render"]["output"])
    v1, v2, v3 = st.columns([1, 1, 2])
    v1.caption("Drex vertical"); v1.video(str(cdir / "vertical.mp4"))
    if (cdir / "gemini.mp4").exists():
        v2.caption(f"Gemini vertical ({g.get('model', '')})"); v2.video(str(cdir / "gemini.mp4"))
    with v3:
        st.html('<div class="k">Drex vs Gemini on this clip</div>')
        acc = C.get("this_run") or C.get("flyer_run")
        rows = [{"model": "Drex", "cost": f"${T['drex_usd']:.4f}", "decision time": f"{T['s3_latency_s']:.0f} s"},
                {"model": f"Gemini ({g.get('model', '')})", "cost": f"${g.get('usd', 0):.4f}", "decision time": f"{g.get('time_s', 0):.0f} s"}]
        if acc:
            rows[0][f"{C['metric']} (Drex ref / Gemini ref)"] = f"{acc['drex'][0]:.0%} / {acc['drex'][1]:.0%}"
            rows[1][f"{C['metric']} (Drex ref / Gemini ref)"] = f"{acc['gemini'][0]:.0%} / {acc['gemini'][1]:.0%}"
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("No human labels: “in frame” is measured against each model's own reading. Gemini makes the per-frame decision "
                   "on the same clip, and the same planner and renderer turn it into video. "
                   + ("Football accuracy numbers are from the results-page run." if not talk else ""))
