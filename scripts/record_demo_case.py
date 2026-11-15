"""Record one fully inspectable case for the playground's "Case studies" page.

Re-runs the Drex steps for a single clip so every stage's input, configuration and raw output is
kept (the batch pipeline only stores decoded regions), then plans and renders it with the same code.
NDI output is taken from the original interview run (raw job JSON), not re-requested.

  python3 scripts/record_demo_case.py --profile football --run clasico   --clip 2 --id football-goal-replay
  python3 scripts/record_demo_case.py --profile talk     --run interview --clip 1 --id interview-gigawatt

Writes demo/assets/cases/<id>/{case.json, debug.mp4, vertical.mp4, gemini.mp4, kf/*.jpg, s1/*.jpg}.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reframe import media, pipeline as fb, talk  # noqa: E402
from reframe.nace import drex  # noqa: E402

PRICE = 0.03 / 1e6  # drex-max-v1.1: 30 nUSD per input token
OUT = ROOT / "demo" / "assets" / "cases"


def sh(*a):
    subprocess.run(a, check=True, capture_output=True)


def jpg(img, path: Path, width=640, q=72):
    im = img.copy()
    im.thumbnail((width, width))
    im.save(path, quality=q)
    return path.name


def answers_compact(ans: dict, k: int, keys) -> dict:
    """{question: {choice, confidence, p: {label: prob}}} for Image k."""
    out = {}
    for key in keys:
        a = ans.get(f"f{k}_{key}")
        if a:
            out[key] = {"choice": a.get("choice"), "confidence": round(a.get("confidence") or 0, 3),
                        "p": {lab: round(v, 4) for lab, v in (a.get("probabilities") or {}).items()}}
    return out


def question_template(questions: dict) -> list[dict]:
    """The questions asked about Image 1, with their full instructions and options."""
    rows = []
    for qid, q in questions.items():
        if qid.startswith("f1_"):
            crit = q.get("criteria")
            rows.append({"id": qid[3:], "type": q["type"], "instructions": q.get("instructions", ""),
                         "options": list(crit) if isinstance(crit, dict) else crit})
    return rows


def run_s3(frames, detect, keys, case_dir, said_fn=None):
    """All s3 batches for the clip, keeping each request and raw response."""
    kdir = case_dir / "kf"
    kdir.mkdir(parents=True, exist_ok=True)
    batches, keyframes, kf_rows = [], [], []
    for b in range(0, len(frames), fb.BATCH):
        fr = frames[b:b + fb.BATCH]
        said = said_fn(fr) if said_fn else None
        t0 = time.time()
        det = detect(fr, said) if said_fn else detect(fr)
        dt = time.time() - t0
        prep, resp = det["prepared"], det["response"]
        usage = resp.get("usage") or {}
        batches.append({"index": len(batches), "frames": len(fr), "t0": fr[0][0], "state": prep["state"],
                        "n_questions": len(prep["questions"]), "latency_s": round(resp.get("_latency_s", dt), 2),
                        "input_tokens": usage.get("input_tokens"), "image_tokens": usage.get("image_tokens"),
                        "usd": round((usage.get("input_tokens") or 0) * PRICE, 6),
                        "request_id": resp.get("request_id")})
        for k, (row, dec, img) in enumerate(zip(det["keyframes"], det["decoded"], prep["images"]), start=1):
            name = jpg(img, kdir / f"{len(keyframes):03d}.jpg")
            keyframes.append(row)
            kf_rows.append({"t": row["t"], "img": f"kf/{name}", "batch": len(batches) - 1, "image_k": k,
                            "answers": answers_compact(resp["answers"], k, keys),
                            "decoded": {"region_px": [round(v) for v in dec["region"]],
                                        "focus_px": ({"x": round(dec["focus"]["x"]), "y": round(dec["focus"]["y"])}
                                                     if dec.get("focus") else None),
                                        "x_uncertainty_px": round(dec["x_uncertainty_px"], 1),
                                        "flags": dec["flags"]}})
    return batches, keyframes, kf_rows, question_template(prep["questions"])


def finish_plan(plan, kf_rows, keyframes):
    W, H, fps, c0 = plan["W"], plan["H"], plan["fps"], plan["clip"][0]
    pinned = {round(k["t"], 3) for k in plan.get("kf", []) if k.get("pinned")}
    for r in kf_rows:
        i = min(max(int(np.ceil((r["t"] - c0) * fps - 1e-9)), 0), len(plan["cx"]) - 1)
        r["t_rel"] = round(r["t"] - c0, 2)
        r["crop"] = {"cx_px": round(plan["cx"][i]), "cy_px": round(plan["cy"][i]), "h_px": round(plan["h"][i]),
                     "cx_line": round(plan["cx"][i] / W * 16, 2), "h_frac": round(plan["h"][i] / H, 3)}
        r["pinned"] = round(r["t"], 3) in pinned
    path = [[round(i / fps, 2), round(plan["cx"][i] / W, 4), round(plan["h"][i] / H, 3)]
            for i in range(0, len(plan["cx"]), 3)]
    return path


def render(video, plan, keyframes, case_dir, gem_src):
    raw_v, raw_d = case_dir / "_vertical.mp4", case_dir / "_debug.mp4"
    fb.render(video, plan, raw_v, raw_d, keyframes)
    sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_v), "-vf", "scale=360:640", "-c:v", "libx264", "-crf", "26",
       "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart",
       str(case_dir / "vertical.mp4"))
    sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_d), "-an", "-c:v", "libx264", "-crf", "26", "-preset",
       "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(case_dir / "debug.mp4"))
    raw_v.unlink(); raw_d.unlink()
    if gem_src.exists():
        sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(gem_src), "-vf", "scale=360:640", "-an", "-c:v", "libx264",
           "-crf", "26", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           str(case_dir / "gemini.mp4"))


def gemini_numbers(gem_run: Path, ei: int):
    sm = {e["event_index"]: e for e in json.loads((gem_run / "summary.json").read_text())}
    e = sm.get(ei)
    if not e:
        return None
    req = json.loads((gem_run / f"event_{ei:02d}_anchors.request.json").read_text())
    return {"model": req["payload"]["model"], "usd": round(e["usd"] or 0, 4), "time_s": round(e["upload_s"] + e["decide_s"], 1),
            "input_tokens": (e["usage"] or {}).get("total_input_tokens"),
            "output_tokens": ((e["usage"] or {}).get("total_output_tokens", 0) + (e["usage"] or {}).get("total_thought_tokens", 0))}


# ---------------------------------------------------------------- football

def record_football(a, case_dir):
    run = ROOT / "runs" / a.run
    s0 = json.loads((run / "s0.json").read_text())
    events = json.loads((run / "events.json").read_text())
    s1 = json.loads((run / "s1.json").read_text())
    ev = events[a.clip]
    video = Path(s0["video"])
    c0, c1 = ev["clip"]
    cand = min(s1, key=lambda c: abs(c["t"] - ev["t"]))

    # s0: structure around the clip
    hop = s0["loud_hop"]
    loud = np.array(s0["loud_db"])
    i0, i1 = int((c0 - 20 - s0["start"]) / hop), int((c1 + 10 - s0["start"]) / hop)
    loud_win = [[round(s0["start"] + i * hop - c0, 1), round(float(loud[i]), 1)] for i in range(max(i0, 0), min(i1, len(loud)), 2)]
    commentary = [{"t_rel": round(x["start"] + s0["start"] - c0, 1), "text": x["text"].strip()}
                  for x in s0["transcript"] if c0 - 20 <= x["start"] + s0["start"] <= c1 + 5]
    cuts = [round(c - c0, 2) for c in s0["cuts"] if c0 < c < c1]

    # s1: re-run the Drex event confirmation for this candidate, keeping I/O
    s1dir = case_dir / "s1"; s1dir.mkdir(exist_ok=True)
    ts = [cand["t"] - 10 + i * 20 / 7 for i in range(8)]
    imgs = [media.grab(video, t, 640) for t in ts]
    commentary_near = fb._transcript_near(s0, cand["t"])
    state = (f"Eight frames (Image 1..Image 8, in time order, ~3 s apart) from a football match broadcast "
             f"around one moment. Commentary near this moment: \"{commentary_near or '(none)'}\"")
    q = {"event": {"type": "choice", "criteria": fb.EVENT_TYPES,
                   "instructions": "What is the most important thing happening across these frames?"},
         "excitement": {"type": "score", "criteria": ["routine", "mildly interesting", "notable",
                                                      "highlight-worthy", "match-defining"],
                        "instructions": "How much does this moment deserve a place in a short vertical highlight reel?"},
         "peak": {"type": "choice", "criteria": {f"Image {i + 1}": None for i in range(8)},
                  "instructions": "In which image does the key action (shot, goal, tackle, card) happen? If it is "
                                  "only reactions or replays, pick the frame closest to the decisive moment."}}
    r1 = drex(state, q, imgs, tag="demo_case_s1")
    s1_frames = [{"t_rel": round(t - c0, 1), "img": f"s1/{jpg(im, s1dir / f'{i}.jpg', 480)}"} for i, (t, im) in enumerate(zip(ts, imgs))]
    s1_out = {k: {"choice": v.get("choice"), "score": v.get("score"), "confidence": round(v.get("confidence") or 0, 3),
                  "p": {lab: round(p, 4) for lab, p in (v.get("probabilities") or {}).items()}}
              for k, v in r1["answers"].items()}

    # s3: per-frame decisions
    frames = media.grab_many(video, c0, c1 - c0, fb.KF_STEP, width=1280)
    keys = ["shot", "left", "right", "top", "bottom", "focus_x", "focus_y"]
    batches, keyframes, kf_rows, qtpl = run_s3(frames, fb.detect_keyframe_batch, keys, case_dir)

    # s4 + render
    plan = fb.plan_clip(s0, ev, keyframes)
    path = finish_plan(plan, kf_rows, keyframes)
    render(video, plan, keyframes, case_dir, ROOT / "runs" / "gemini_aligned_clasico" / f"event_{a.clip:02d}_vertical.mp4")
    for r in kf_rows:  # commentary heard at this moment
        r["heard"] = next((c["text"] for c in reversed(commentary) if c["t_rel"] <= r["t_rel"] + 0.01), "")

    cmp = json.loads((ROOT / "runs" / "compare" / "compare.json").read_text())
    cc = next((c for c in cmp["clips"] if c["event_index"] == a.clip), None)
    s3_usd = sum(b["usd"] for b in batches)
    return {
        "scenario": "football", "title": "Football · 73′ Rakitić goal, replay and celebration",
        "subtitle": "Real Madrid 2–3 Barcelona (2017). The 22 s clip Drex cut right after the goal: the replay, then the celebration.",
        "clip": {"start": c0, "end": c1, "duration": round(c1 - c0, 2), "W": s0["W"], "H": s0["H"], "fps": s0["fps"],
                 "source_window": [s0["start"], s0["start"] + s0["dur"]]},
        "stages": {
            "s0": {"tools": ["ffmpeg scene detection", "RMS loudness", "commentary ASR (Nace NDI; local Whisper fallback in this run)"],
                   "input": {"video": "El Clásico 2017 broadcast, 1280×720, 25 fps", "window": "01:12:30–01:42:30 (30 min)"},
                   "config": {"scene_threshold": 0.3, "loudness_hop_s": hop, "asr": s0["asr"],
                              "ndi_note": "NDI upload was blocked by the venue firewall (HTTP 403) during this run, so the pipeline fell back to local Whisper base.en"},
                   "output": {"shot_cuts_in_clip_s": cuts, "commentary": commentary, "loudness_db": loud_win,
                              "totals": {"cuts_in_window": len(s0["cuts"]), "transcript_segments": len(s0["transcript"])}}},
            "s1": {"tools": ["keyword + crowd-noise recall", "Drex confirm (1 request)"],
                   "input": {"candidate_t_rel": round(cand["t"] - c0, 1), "recall_score": round(cand["recall_score"], 2),
                             "crowd_noise_z": round(cand["loud_z"], 2), "frames": s1_frames, "state": state},
                   "config": {"model": "drex-max-v1.1", "questions": [{"id": k, "type": v["type"], "instructions": v["instructions"],
                                                                     "options": list(v["criteria"]) if isinstance(v["criteria"], dict) else v["criteria"]}
                                                                    for k, v in q.items()],
                              "clip_rule": f"clip = peak − {fb.PRE:.0f} s … peak + {fb.POST:.0f} s"},
                   "output": {"answers": s1_out, "latency_s": round(r1.get("_latency_s", 0), 2),
                              "input_tokens": (r1.get("usage") or {}).get("input_tokens"),
                              "usd": round(((r1.get("usage") or {}).get("input_tokens") or 0) * PRICE, 6)}},
            "s3": {"tools": ["numbered grid overlay", "Drex Max (8 frames / request)"],
                   "config": {"model": "drex-max-v1.1", "frame_every_s": fb.KF_STEP, "frames_per_request": fb.BATCH,
                              "grid": "16 × 9 cells: vertical lines 0–16, horizontal rows A–J", "questions_per_frame": len(qtpl),
                              "question_template": qtpl},
                   "batches": batches},
            "s4": {"tools": ["9:16 geometry", "confidence-weighted least squares", "ball-in-frame constraint"],
                   "config": {**fb.P, "min_h_frac": 0.55, "aspect": "9:16"},
                   "output": {"cuts_rel": [round(c - c0, 2) for c in plan["cuts"]], "shots": [
                       {"start": round(s["start"] - c0, 2), "end": round(s["end"] - c0, 2), "type": s["type"]} for s in plan["shots"]],
                       "pinned_keyframes": sum(1 for r in kf_rows if r["pinned"])}},
            "render": {"output": "720×1280 vertical with audio + side-by-side debug view (source with region and crop)"},
        },
        "keyframes": kf_rows, "path": path, "cuts": [round(c - c0, 2) for c in plan["cuts"]],
        "totals": {"drex_calls": 1 + len(batches), "drex_usd": round(s3_usd + ((r1.get("usage") or {}).get("input_tokens") or 0) * PRICE, 5),
                   "s3_latency_s": round(sum(b["latency_s"] for b in batches), 1), "keyframes": len(kf_rows)},
        "compare": {"gemini": gemini_numbers(ROOT / "runs" / "gemini_aligned_clasico", a.clip),
                    "metric": "ball in frame",
                    "flyer_run": ({"drex": [cc["drex"]["ball_in_ref_drex"], cc["drex"]["ball_in_ref_gemini"]],
                                   "gemini": [cc["gemini"]["ball_in_ref_drex"], cc["gemini"]["ball_in_ref_gemini"]]} if cc else None)},
    }


# ---------------------------------------------------------------- interview

def record_talk(a, case_dir):
    run = ROOT / "runs" / a.run
    s0 = json.loads((run / "s0.json").read_text())
    events = json.loads((run / "events.json").read_text())
    s1 = json.loads((run / "s1.json").read_text())
    ev = events[a.clip]
    video = Path(s0["video"])
    c0, c1 = ev["clip"]
    segs = [s for s in talk.lines(s0, run) if s["text"].strip()]

    # s0: raw NDI output for this stretch, plus the sentence lines built from it
    chunk = int((c0 - s0["start"]) // 600)
    job = json.loads((run / f"ndi_part{chunk}.json").read_text())
    off = chunk * 600
    ndi_raw = [{"kind": x["kind"], "start_ms": x["start_ms"], "end_ms": x["end_ms"], "speaker": x.get("speaker"),
                "transcript": x.get("transcript", "")}
               for x in job["result"]["transcript"]["segments"]
               if x["end_ms"] / 1000 + off >= c0 - s0["start"] - 5 and x["start_ms"] / 1000 + off <= c1 - s0["start"] + 5]
    ndi_cost = sum(float(json.loads(p.read_text())["cost"]["total_usd"]) for p in sorted(run.glob("ndi_part*.json")))
    lines_clip = [{"t_rel": round(l["start"] + s0["start"] - c0, 1), "end_rel": round(l["end"] + s0["start"] - c0, 1),
                   "speaker": l.get("speaker"), "text": l["text"]}
                  for l in segs if l["end"] + s0["start"] >= c0 - 3 and l["start"] + s0["start"] <= c1 + 3]

    # s1: this window's score among all windows, then re-run the opening/closing-line choice
    sc = sorted((w["score"] for w in s1), reverse=True)
    win = max(s1, key=lambda w: min(w["end"], ev["end"]) - max(w["start"], ev["start"]))
    # same request as talk.refine_edges, kept here so the raw answers are recorded
    i, j = win["i"], win["j"]
    lo, hi = max(0, i - 6), min(len(segs) - 1, j + 6)
    txt = [f"[L{k}] ({segs[k].get('speaker') or '?'}) {segs[k]['text']}" for k in range(lo, hi + 1)]
    starts = list(range(max(lo, i - 5), min(j, i + 5) + 1))
    ends = list(range(max(i + 1, j - 5), min(hi, j + 5) + 1))
    qe = {"start": {"type": "choice", "criteria": {f"L{k}": None for k in starts},
                    "instructions": "Which line should a standalone vertical short START on? Pick the first line of a complete "
                                    "thought or question that hooks a viewer with no prior context. Never pick a line that "
                                    "continues a sentence or answers something not shown."},
          "end": {"type": "choice", "criteria": {f"L{k}": None for k in ends},
                  "instructions": "Which line should the short END on? Pick the line that completes the thought or lands the "
                                  "punchline. Never end mid-sentence or on a filler reply."}}
    state_e = "Transcript lines from an interview (diarized speaker id, text):\n" + "\n".join(txt)
    edge_resp = drex(state_e, qe, tag="demo_case_s1_edges")

    # s3
    frames = media.grab_many(video, c0, c1 - c0, fb.KF_STEP, width=1280)
    keys = ["shot", "who", "left", "right", "top", "bottom", "focus_x", "focus_y"]
    batches, keyframes, kf_rows, qtpl = run_s3(
        frames, talk.detect_talk_batch, keys, case_dir,
        said_fn=lambda fr: talk.said_between(s0, run, fr[0][0], fr[-1][0] + fb.KF_STEP))

    plan = talk.plan_clip(s0, ev, keyframes, run)
    path = finish_plan(plan, kf_rows, keyframes)
    render(video, plan, keyframes, case_dir, ROOT / "runs" / "gemini_aligned_interview" / f"event_{a.clip:02d}_vertical.mp4")
    for r in kf_rows:
        cur = next((l for l in lines_clip if l["t_rel"] <= r["t_rel"] + 0.01 <= l["end_rel"] + 0.6), None)
        r["heard"], r["speaker"] = (cur["text"], cur["speaker"]) if cur else ("", None)

    # Drex vs Gemini on this clip (same locked planner)
    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_talk
    gem_run = ROOT / "runs" / "gemini_aligned_interview"
    gk = json.loads((gem_run / f"event_{a.clip:02d}_keyframes.json").read_text())
    gp = json.loads((gem_run / f"event_{a.clip:02d}_plan.json").read_text())
    acc = {"drex": [eval_talk.metrics(plan, keyframes)["face_in"], eval_talk.metrics(plan, gk)["face_in"]],
           "gemini": [eval_talk.metrics(gp, keyframes)["face_in"], eval_talk.metrics(gp, gk)["face_in"]]}
    s3_usd = sum(b["usd"] for b in batches)
    e_usd = ((edge_resp.get("usage") or {}).get("input_tokens") or 0) * PRICE
    order = []
    for _, _, spk in plan["speaker_turns"]:
        if spk not in order:
            order.append(spk)
    return {
        "scenario": "talk", "title": "Interview · “a gigawatt cluster for a year”",
        "subtitle": "A 30-min three-person interview. One guest works through what it costs to run a gigawatt cluster; the camera stays locked on whoever is talking.",
        "clip": {"start": c0, "end": c1, "duration": round(c1 - c0, 2), "W": s0["W"], "H": s0["H"], "fps": s0["fps"],
                 "source_window": [s0["start"], s0["start"] + s0["dur"]]},
        "stages": {
            "s0": {"tools": ["Nace NDI Parse (transcript + speaker diarization)", "local Whisper sentence timing", "ffmpeg scene detection"],
                   "input": {"audio": "30 min mono 16 kHz, split into 10-min chunks (NDI's internal Whisper rejects > 50 MB)",
                             "ndi_calls": ["POST /v1/documents/upload-grants", "POST <upload_url> (multipart, X-Upload-Token)",
                                           "POST /v1/documents/parse?wait_seconds=60  {source: {type: workspace_file, ...}}",
                                           "GET /v1/documents/jobs/{job_id} (poll until succeeded)"]},
                   "config": {"chunk_s": 600, "chunks": len(list(run.glob("ndi_part*.json"))), "parse_mode": "default (low)"},
                   "output": {"ndi_chunk": chunk, "ndi_segments_raw": ndi_raw, "ndi_total_usd": round(ndi_cost, 4),
                              "sentence_lines": lines_clip,
                              "note": "NDI segments are coarse (median ~17 s); each Whisper sentence inherits the NDI speaker it overlaps most."}},
            "s1": {"tools": ["Drex text scoring (70 windows, 3 requests)", "Drex opening/closing-line choice"],
                   "input": {"window_lines": [{"line": f"L{k}", "speaker": segs[k].get("speaker"), "text": segs[k]["text"]} for k in range(win["i"], win["j"] + 1)]},
                   "config": {"model": "drex-max-v1.1", "window_s": "25–60", "score_scale": ["unusable on its own", "weak", "decent", "strong", "must-post"],
                              "edge_questions": [{"id": k, "instructions": v["instructions"], "options": list(v["criteria"])} for k, v in qe.items()]},
                   "output": {"window_score": round(win["score"], 2), "rank": sc.index(win["score"]) + 1, "of": len(sc),
                              "score_p50": round(float(np.median(sc)), 2),
                              "edges": {k: {"choice": v.get("choice"), "p": {lab: round(p, 4) for lab, p in (v.get("probabilities") or {}).items()}}
                                        for k, v in edge_resp["answers"].items()},
                              "latency_s": round(edge_resp.get("_latency_s", 0), 2)}},
            "s3": {"tools": ["numbered grid overlay", "Drex Max (8 frames + transcript / request)"],
                   "config": {"model": "drex-max-v1.1", "frame_every_s": fb.KF_STEP, "frames_per_request": fb.BATCH,
                              "grid": "16 × 9 cells: vertical lines 0–16, horizontal rows A–J", "questions_per_frame": len(qtpl),
                              "question_template": qtpl},
                   "batches": batches},
            "s4": {"tools": ["locked framing per shot", "hard cuts only"],
                   "config": {**{k: v for k, v in talk.TP.items() if k in ("mode", "out_frac", "min_seg_s", "safe")}, "min_h_frac": talk.MIN_H_FRAC,
                              "min_turn_s": talk.MIN_TURN_S},
                   "output": {"camera_cuts_rel": [round(c - c0, 2) for c in plan["camera_cuts"]],
                              "speaker_cuts_rel": [round(c - c0, 2) for c in plan["virtual_cuts"]],
                              "reframe_cuts_rel": [round(c - c0, 2) for c in plan.get("reframe_cuts", [])],
                              "shots": [{"start": round(s["start"] - c0, 2), "end": round(s["end"] - c0, 2), "type": s["type"]} for s in plan["shots"]],
                              "speaker_turns": [[round(a_ - c0, 2), round(b_ - c0, 2), "ABC"[min(order.index(k), 2)]] for a_, b_, k in plan["speaker_turns"]]}},
            "render": {"output": "720×1280 vertical with audio + side-by-side debug view (source with region and crop)"},
        },
        "keyframes": kf_rows, "path": path, "cuts": [round(c - c0, 2) for c in plan["cuts"]],
        "totals": {"drex_calls": 1 + len(batches), "drex_usd": round(s3_usd + e_usd, 5),
                   "s3_latency_s": round(sum(b["latency_s"] for b in batches), 1), "keyframes": len(kf_rows),
                   "ndi_usd_whole_30min": round(ndi_cost, 4)},
        "compare": {"gemini": gemini_numbers(gem_run, a.clip), "metric": "face in frame", "this_run": acc},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["football", "talk"], required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--clip", type=int, required=True)
    ap.add_argument("--id", required=True)
    a = ap.parse_args()
    case_dir = OUT / a.id
    case_dir.mkdir(parents=True, exist_ok=True)
    data = record_football(a, case_dir) if a.profile == "football" else record_talk(a, case_dir)
    data["id"] = a.id
    data["recorded_at"] = time.strftime("%Y-%m-%d %H:%M")
    (case_dir / "case.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    size = sum(p.stat().st_size for p in case_dir.rglob("*") if p.is_file())
    print(f"[case] {a.id}: {len(data['keyframes'])} keyframes, {data['totals']['drex_calls']} Drex calls, "
          f"${data['totals']['drex_usd']:.4f}, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
