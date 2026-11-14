"""End-to-end: long landscape match -> vertical highlight clips.

Stages (each cached as JSON under runs/<name>/):
  s0  structure   shot cuts (ffmpeg scene), loudness (RMS), transcript (NDI Parse, Whisper fallback)
  s1  events      recall = commentary keywords ∪ loudness peaks; Drex confirms type + excitement
  s3  keyframes   every KF_STEP s inside each event clip, 8 frames / Drex request,
                  per frame: shot type + must-keep region edges (grid line choices)
  s4  plan        region -> 9:16 box -> per-shot confidence-weighted smoothing, hard cut at shot edges
  render          vertical mp4 + debug side-by-side mp4 per event
(Shot-type classification, the design's Stage 2, is folded into the s3 requests.)

Usage:
  python3 -m reframe.pipeline --video <mp4> --start 4350 --dur 1800 --name clasico [--stages s0,s1,s3,s4,render]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import media
from .geometry import ASPECT, box_xyxy, clamp_center, fit_vertical
from .grid import Grid, decode_region, edge_questions
from .nace import drex, ndi_parse_file
from .smooth import deadzone, solve_path, speed_cap

ROOT = Path(__file__).resolve().parents[1]
KF_STEP = 0.5          # seconds between Drex keyframes inside clips
BATCH = 8              # Drex Max max images / request
PRE, POST = 14.0, 8.0  # clip = [t_event - PRE, t_event + POST]
MAX_EVENTS = 8
GRID = Grid(16, 9)

KEYWORDS = {
    "goal": 3.0, "scores": 3.0, "scored": 3.0, "equalis": 3.0, "equaliz": 3.0, "winner": 2.5,
    "red card": 3.0, "sent off": 3.0, "penalty": 2.0, "own goal": 3.0,
    "what a": 1.5, "unbelievable": 1.5, "incredible": 1.5, "brilliant": 1.0, "magnificent": 1.5,
    "save": 1.5, "saved": 1.5, "post": 1.0, "crossbar": 1.5, "chance": 0.8, "header": 0.8,
    "foul": 0.8, "yellow card": 1.5, "offside": 0.8,
    "converted": 3.0, "made it \\d": 3.0, "would you believe": 2.0, "finish": 1.5,
    "back into (this|the) game": 2.0, "level": 1.0, "last[- ]minute": 2.0,
}
SHOT_TYPES = {
    "wide": "main high camera showing a large part of the pitch with many small players",
    "medium": "tighter live-play view with a few players fairly large",
    "closeup": "one or two people fill much of the frame (player, coach, referee)",
    "crowd": "fans, bench or stadium, not the play",
    "graphic": "full-screen graphic, lineup, stats, logo transition or heavy overlay",
}
EVENT_TYPES = {
    "goal": "a goal is scored (ball goes in, celebrations, kickoff restart after)",
    "own_goal": "a defender puts the ball into his own net",
    "big_chance": "a shot, save, woodwork or near miss without a goal",
    "card": "a referee shows a red or yellow card / sending off",
    "skill": "a notable dribble or run that is not a shot",
    "foul_or_stoppage": "foul, injury, substitution or other stoppage",
    "none": "routine play, nothing notable",
}


def _cache(path: Path, fn):
    if path.exists():
        return json.loads(path.read_text())
    val = fn()
    path.write_text(json.dumps(val, indent=1))
    return val


# ---------------- s0 ----------------

def _transcript_segments(job: dict) -> list[dict]:
    """Normalize an NDI Parse audio result to [{start, end, text}] (seconds, window-relative)."""
    tr = (job.get("result") or {}).get("transcript") or {}
    segs = []

    def walk(o):
        if isinstance(o, dict):
            if "start_ms" in o and "end_ms" in o:
                txt = o.get("text") or o.get("content") or o.get("transcript") or ""
                if isinstance(txt, list):
                    txt = " ".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in txt)
                if (o.get("type") or o.get("kind")) != "silence" and txt:
                    segs.append({"start": o["start_ms"] / 1000, "end": o["end_ms"] / 1000, "text": str(txt),
                                 "speaker": o.get("speaker")})
                    return
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(tr)
    return segs


NDI_CHUNK_S = 600  # NDI's internal Whisper rejects >50 MB (≈ 26 min of 16 kHz WAV); 10-min chunks are safe


def _ndi_chunked(audio: Path, dur: float, run: Path) -> list[dict]:
    """NDI Parse in 10-minute chunks (parallel). Speaker ids are only consistent WITHIN a chunk,
    so they are prefixed with the chunk index (c0:SPEAKER_01)."""
    chunks = []
    for i, c0 in enumerate(np.arange(0, dur, NDI_CHUNK_S)):
        part = run / f"audio_part{i}.m4a"
        if not part.exists():
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(c0), "-t",
                            str(NDI_CHUNK_S), "-i", str(audio), "-c", "copy", str(part)], check=True)
        chunks.append((i, float(c0), part))

    def one(ch):
        i, c0, part = ch
        cache = run / f"ndi_part{i}.json"
        job = json.loads(cache.read_text()) if cache.exists() else ndi_parse_file(part)
        cache.write_text(json.dumps(job, indent=1))
        segs = _transcript_segments(job)
        for sg in segs:
            sg["start"] += c0
            sg["end"] += c0
            if sg.get("speaker"):
                sg["speaker"] = f"c{i}:{sg['speaker']}"
        return segs
    with ThreadPoolExecutor(3) as ex:
        return [sg for segs in ex.map(one, chunks) for sg in segs]


def _whisper_segments(audio: Path) -> list[dict]:
    import whisper
    model = whisper.load_model("base.en")
    out = model.transcribe(str(audio), language="en", fp16=False, verbose=False)
    return [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in out["segments"]]


def stage0(video: Path, start: float, dur: float, run: Path, asr: str) -> dict:
    def build():
        info = media.probe(video)
        t = time.time()
        cuts = media.shot_cuts(video, start, dur)
        t_cuts = time.time() - t
        t = time.time()
        loud = media.loudness(video, start, dur).tolist()
        audio = media.extract_audio(video, start, dur, run / "audio.m4a")
        transcript, asr_used, asr_err = [], None, None
        if asr in ("ndi", "auto"):
            try:
                transcript, asr_used = _ndi_chunked(audio, dur, run), "ndi"
            except Exception as e:  # keep the demo moving
                asr_err = repr(e)[:500]
        if not transcript and asr in ("whisper", "auto"):
            transcript, asr_used = _whisper_segments(audio), "whisper"
        return {"video": str(video), "start": start, "dur": dur, **info, "cuts": cuts,
                "loud_hop": 0.5, "loud_db": loud, "transcript": transcript, "asr": asr_used,
                "asr_error": asr_err, "timing_s": {"cuts": t_cuts, "audio+asr": time.time() - t}}
    return _cache(run / "s0.json", build)


# ---------------- s1 ----------------

def recall_candidates(s0: dict, k: int = 20, nms_s: float = 35.0) -> list[dict]:
    start, hop = s0["start"], s0["loud_hop"]
    loud = np.array(s0["loud_db"])
    # crowd/commentary excitement: 3 s mean vs 60 s rolling median
    w = int(3 / hop)
    sm = np.convolve(loud, np.ones(w) / w, mode="same")
    base = np.array([np.median(sm[max(0, i - 60):i + 60]) for i in range(len(sm))])
    z = (sm - base) / (np.std(sm - base) + 1e-6)
    score = np.clip(z, 0, None)
    for seg in s0["transcript"]:
        txt = seg["text"].lower()
        txt = re.sub(r"goal ?(kick|keeper|keeping|line|post|mouth|scorer)s?", " ", txt)
        kw = sum(v for kk, v in KEYWORDS.items() if re.search(r"\b" + kk, txt))
        # elongated shouts ("gooooal") count as goal
        if re.search(r"\bgo+a+l+\b", txt):
            kw = max(kw, 3.0)
        if kw:
            i0, i1 = int(seg["start"] / hop), int(seg["end"] / hop) + 1
            score[i0:i1] += kw
    cands = []
    order = np.argsort(-score)
    for i in order:
        if score[i] <= 0.5 or len(cands) >= k:
            break
        t = start + i * hop
        if all(abs(t - c["t"]) > nms_s for c in cands):
            cands.append({"t": t, "recall_score": float(score[i]), "loud_z": float(z[i])})
    return sorted(cands, key=lambda c: c["t"])


def _transcript_near(s0, t, before=12, after=8) -> str:
    rel = t - s0["start"]
    return " ".join(s["text"].strip() for s in s0["transcript"]
                    if s["end"] >= rel - before and s["start"] <= rel + after)[:1200]


def confirm_event(video: Path, s0: dict, c: dict) -> dict:
    ts = [c["t"] - 10 + i * 20 / 7 for i in range(8)]
    imgs = [media.grab(video, t, 640) for t in ts]
    commentary = _transcript_near(s0, c["t"])
    state = (f"Eight frames (Image 1..Image 8, in time order, ~3 s apart) from a football match broadcast "
             f"around one moment. Commentary near this moment: \"{commentary or '(none)'}\"")
    q = {
        "event": {"type": "choice", "criteria": EVENT_TYPES,
                  "instructions": "What is the most important thing happening across these frames?"},
        "excitement": {"type": "score", "criteria": ["routine", "mildly interesting", "notable",
                                                     "highlight-worthy", "match-defining"],
                       "instructions": "How much does this moment deserve a place in a short vertical highlight reel?"},
        "peak": {"type": "choice", "criteria": {f"Image {i+1}": None for i in range(8)},
                 "instructions": "In which image does the key action (shot, goal, tackle, card) happen? If it is "
                                 "only reactions or replays, pick the frame closest to the decisive moment."},
    }
    r = drex(state, q, imgs, tag="s1_confirm")
    a = r["answers"]
    peak_i = int(a["peak"]["choice"].split()[-1]) - 1
    return {**c, "event": a["event"]["choice"], "event_probs": a["event"]["probabilities"],
            "excitement": a["excitement"]["score"], "t_peak": ts[peak_i], "commentary": commentary,
            "usage": r.get("usage")}


def stage1(video: Path, s0: dict, run: Path) -> list[dict]:
    def build():
        cands = recall_candidates(s0)
        with ThreadPoolExecutor(2) as ex:
            conf = list(ex.map(lambda c: confirm_event(video, s0, c), cands))
        return conf
    return _cache(run / "s1.json", build)


def pick_events(s1: list[dict], s0: dict) -> list[dict]:
    keep = [e for e in s1 if e["event"] not in ("none", "foul_or_stoppage") or e["excitement"] >= 2.5]
    keep = sorted(keep, key=lambda e: -(e["excitement"] + (1 if e["event"] in ("goal", "own_goal") else 0)))
    keep = keep[:MAX_EVENTS]
    lo, hi = s0["start"], s0["start"] + s0["dur"]
    for e in keep:
        e["clip"] = [max(lo, e["t_peak"] - PRE), min(hi, e["t_peak"] + POST)]
    return sorted(keep, key=lambda e: e["t"])


# ---------------- s3 ----------------

def kf_questions(k: int) -> dict:
    q = {f"f{k}_shot": {"type": "choice", "criteria": SHOT_TYPES,
                        "instructions": f"What kind of broadcast shot is Image {k}?"}}
    q.update(edge_questions(GRID, image_k=k, prefix=f"f{k}_"))
    return q


def prepare_keyframe_batch(frames: list, viewing_goal: str | None = None) -> dict:
    """Build the s3 request, optionally adding a user viewing preference to state.

    With no goal the payload matches the existing pipeline. The numbered grid,
    seven questions per frame, probability decoding and focus point are shared.
    """
    if not 1 <= len(frames) <= BATCH:
        raise ValueError(f"A keyframe batch must contain 1 to {BATCH} frames")
    if viewing_goal is not None and (not isinstance(viewing_goal, str) or len(viewing_goal) > 2000):
        raise ValueError("Viewing goal must be text of at most 2000 characters")
    images = [GRID.draw(image) for _, image in frames]
    questions = {}
    for k in range(1, len(frames) + 1):
        questions.update(kf_questions(k))
    state = (f"{len(frames)} consecutive frames (Image 1..Image {len(frames)}, {KF_STEP}s apart) from a "
             f"football broadcast that will be reframed to a 9:16 vertical crop. Use the sequence "
             f"to understand where the play is going.")
    if viewing_goal and viewing_goal.strip():
        state += (f" Viewing goal: {viewing_goal.strip()} "
                  "Use this preference within the must-keep region definition. Rely on visible evidence; "
                  "do not invent offscreen subjects or events.")
    return {"state": state, "questions": questions, "images": images}


def detect_keyframe_batch(frames: list, viewing_goal: str | None = None, *, on_event=None) -> dict:
    """Execute and decode one s3 batch for both the pipeline and the demo."""
    prepared = prepare_keyframe_batch(frames, viewing_goal)
    if on_event:
        on_event("prepared", prepared)
    response = drex(prepared["state"], prepared["questions"], prepared["images"], tag="s3_keyframes")
    if on_event:
        on_event("response", response)
    W, H = frames[0][1].size
    rows, decoded = [], []
    for k, (t, _) in enumerate(frames, start=1):
        dec = decode_region(response["answers"], GRID, W, H, prefix=f"f{k}_")
        shot = response["answers"][f"f{k}_shot"]
        rows.append({"t": t, "shot": shot["choice"], "shot_probs": shot["probabilities"],
                     "region_kf": dec["region"], "focus_kf": dec.get("focus"), "x_unc_kf": dec["x_uncertainty_px"],
                     "flags": dec["flags"], "kf_W": W, "kf_H": H})
        decoded.append(dec)
    if on_event:
        on_event("decoded", rows)
    return {"keyframes": rows, "decoded": decoded, "response": response, "prepared": prepared}


def stage3(video: Path, s0: dict, events: list[dict], run: Path) -> dict:
    def build():
        out = {}
        jobs = []
        for ei, e in enumerate(events):
            c0, c1 = e["clip"]
            frames = media.grab_many(video, c0, c1 - c0, KF_STEP, width=1280)
            for b in range(0, len(frames), BATCH):
                jobs.append((ei, frames[b:b + BATCH]))

        def run_batch(job):
            ei, fr = job
            detected = detect_keyframe_batch(fr)
            return ei, detected["keyframes"], detected["response"].get("usage")

        with ThreadPoolExecutor(2) as ex:
            for ei, rows, usage in ex.map(run_batch, jobs):
                out.setdefault(str(ei), []).extend(rows)
        for v in out.values():
            v.sort(key=lambda r: r["t"])
        return out
    return _cache(run / "s3.json", build)


# ---------------- s4 ----------------

# Camera-path parameters (see plan_clip). Tuned by scripts/eval_path.py.
P = {"tau_live": 0.8, "tau_close": 1.0, "dz_live": 0.04, "dz_close": 0.08,
     "vcap_live": 0.9, "safe": 0.8, "ball_iters": 3}

def plan_clip(s0: dict, ev: dict, kfs: list[dict], min_h_frac=0.55) -> dict:
    W, H, fps = s0["W"], s0["H"], s0["fps"]
    c0, c1 = ev["clip"]
    n = int(round((c1 - c0) * fps))
    times = c0 + np.arange(n) / fps
    cuts = [c for c in s0["cuts"] if c0 < c < c1]
    bounds = [c0] + cuts + [c1]
    cx = np.zeros(n); cy = np.zeros(n); hh = np.zeros(n)
    shots = []
    kf_log = []  # for the report's camera-path chart
    for s_a, s_b in zip(bounds[:-1], bounds[1:]):
        idx = np.where((times >= s_a) & (times < s_b))[0]
        if len(idx) == 0:
            continue
        ks = [k for k in kfs if s_a <= k["t"] < s_b]
        if not ks:  # shot shorter than KF_STEP: borrow nearest keyframe
            ks = [min(kfs, key=lambda k: abs(k["t"] - (s_a + s_b) / 2))]
        sx = W / ks[0]["kf_W"]
        tgt = []
        for k in ks:
            x0, y0, x1, y1 = [v * sx for v in k["region_kf"]]
            if k["shot"] in ("graphic", "crowd"):
                x0, x1, y0, y1 = W * 0.3, W * 0.7, 0, H  # nothing to follow: wide centered
            fk = k.get("focus_kf")
            foc = (fk["x"] * sx, fk["y"] * sx) if fk and k["shot"] not in ("graphic", "crowd") else None
            b = fit_vertical([x0, y0, x1, y1], W, H, min_h_frac=min_h_frac, focus=foc)
            sig = max(k["x_unc_kf"] * sx, 1.0)
            tgt.append((k["t"], b["cx"], b["cy"], b["h"], sig, b["flags"]))
        kt = [t[0] for t in tgt]
        sig = [t[4] for t in tgt]
        shot_type = max(set(k["shot"] for k in ks), key=[k["shot"] for k in ks].count)
        live = shot_type in ("wide", "medium")
        # dead zone: ignore keyframe jitter smaller than this (closeups hold steadier)
        dz = (P["dz_live"] if live else P["dz_close"]) * W
        tx = deadzone([t[1] for t in tgt], dz)
        ty = deadzone([t[2] for t in tgt], 0.08 * H)
        th = deadzone([t[3] for t in tgt], 0.10 * H)
        t0 = times[idx[0]]
        # Smoothing strength from a time constant tau (seconds). Data weights are normalized to
        # mean 1 so Drex confidence only sets *relative* trust between keyframes; the earlier
        # absolute 1/sigma^2 weights (~5e-5) were ~40,000x weaker than the smoothness term,
        # which flattened the camera path and let the ball leave the frame 37% of the time.
        tau = P["tau_live"] if live else P["tau_close"]
        lam2 = (tau * fps) ** 4 / (fps * KF_STEP)
        nrm = lambda s: list(np.array(s) / np.sqrt(np.mean(np.square(s))))
        sg = nrm(sig)
        ph = solve_path(len(idx), fps, t0, kt, th, sg, 0.0, 0.0, lam2 * 4)
        ph = np.clip(speed_cap(ph, 0.25 * H / fps), min_h_frac * H, H)
        py = speed_cap(solve_path(len(idx), fps, t0, kt, ty, sg, 0.0, 0.0, lam2), 0.25 * H / fps)
        vcap = (P["vcap_live"] if live else 0.45) * W / fps
        # Ball-in-frame constraint: wherever Drex's focus point (ball / face) falls outside the
        # crop's safe zone, pin the camera there with a strong keyframe and re-solve.
        fkt, fx = [], []
        for k in ks:
            fk = k.get("focus_kf")
            if fk and k["shot"] not in ("graphic", "crowd"):
                fkt.append(k["t"]); fx.append(fk["x"] * sx)
        ckt, cv, cs = list(kt), list(tx), list(sg)
        px = speed_cap(solve_path(len(idx), fps, t0, ckt, cv, cs, 0.0, 0.0, lam2), vcap)
        for _ in range(P["ball_iters"]):
            added = 0
            for t, f in zip(fkt, fx):
                j = int(round((t - t0) * fps))
                if not 0 <= j < len(idx):
                    continue
                w = ph[j] * ASPECT
                safe = P["safe"] * w / 2
                if abs(f - px[j]) > safe:
                    ckt.append(t); cv.append(f - np.sign(f - px[j]) * safe * 0.8); cs.append(0.15)
                    added += 1
            if not added:
                break
            px = speed_cap(solve_path(len(idx), fps, t0, ckt, cv, cs, 0.0, 0.0, lam2), vcap)
        pinned = set(ckt[len(kt):])
        for t, raw in zip(kt, tgt):
            f = next((x for tt, x in zip(fkt, fx) if tt == t), None)
            kf_log.append({"t": t, "target": raw[1], "focus": f, "pinned": t in pinned})
        for j, i in enumerate(idx):
            cx[i], cy[i] = clamp_center(px[j], py[j], ph[j], W, H)
            hh[i] = ph[j]
        shots.append({"start": float(s_a), "end": float(s_b), "type": shot_type,
                      "too_wide_frac": float(np.mean(["too_wide" in t[5] for t in tgt]))})
    return {"clip": ev["clip"], "fps": fps, "W": W, "H": H, "cx": cx.tolist(), "cy": cy.tolist(),
            "h": hh.tolist(), "shots": shots, "cuts": cuts, "kf": kf_log, "params": dict(P)}


# ---------------- render ----------------

def render(video: Path, plan: dict, out_v: Path, out_dbg: Path, kfs: list[dict]):
    W, H, fps = plan["W"], plan["H"], plan["fps"]
    c0, c1 = plan["clip"]
    ow, oh = 720, 1280
    dec = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{c0:.3f}", "-t",
                            f"{c1 - c0:.3f}", "-i", str(video), "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    enc_args = lambda w, h, path: ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                                   "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
                                   "-ss", f"{c0:.3f}", "-t", f"{c1 - c0:.3f}", "-i", str(video),
                                   "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast",
                                   "-crf", "21", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)]
    dbg_w = 640 + 10 + round(360 * ASPECT * 360 / 360 * 1)  # 640x360 source + 9:16 strip of height 360
    strip_w = round(360 * ASPECT)
    dbg_w = 640 + 10 + strip_w
    dbg_w += dbg_w % 2
    enc = subprocess.Popen(enc_args(ow, oh, out_v), stdin=subprocess.PIPE)
    encd = subprocess.Popen(enc_args(dbg_w, 360, out_dbg), stdin=subprocess.PIPE)
    fsz = W * H * 3
    n = len(plan["cx"])
    sx = W / kfs[0]["kf_W"] if kfs else 1
    for i in range(n):
        buf = dec.stdout.read(fsz)
        if len(buf) < fsz:
            break
        im = Image.fromarray(np.frombuffer(buf, np.uint8).reshape(H, W, 3))
        bx = [round(v) for v in box_xyxy(plan["cx"][i], plan["cy"][i], plan["h"][i])]
        crop = im.crop(bx).resize((ow, oh), Image.LANCZOS)
        enc.stdin.write(crop.tobytes())
        # debug: source with keyframe region (yellow) and smoothed crop (cyan)
        t = c0 + i / fps
        dbg = im.copy()
        d = ImageDraw.Draw(dbg)
        near = [k for k in kfs if abs(k["t"] - t) <= KF_STEP / 2]
        if near:
            r = [v * sx for v in near[0]["region_kf"]]
            d.rectangle(r, outline=(255, 230, 0), width=max(3, W // 400))
            d.text((10, 10), near[0]["shot"], fill=(255, 230, 0))
        d.rectangle(bx, outline=(0, 220, 255), width=max(4, W // 300))
        canvas = Image.new("RGB", (dbg_w, 360), (15, 15, 15))
        canvas.paste(dbg.resize((640, 360)), (0, 0))
        canvas.paste(crop.resize((strip_w, 360)), (650, 0))
        encd.stdin.write(canvas.tobytes())
    for p in (enc, encd):
        p.stdin.close()
        p.wait()
    dec.stdout.close()
    dec.wait()


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--dur", type=float, default=1800)
    ap.add_argument("--name", required=True)
    ap.add_argument("--asr", default="auto", choices=["auto", "ndi", "whisper"])
    ap.add_argument("--stages", default="s0,s1,s3,s4,render")
    ap.add_argument("--only-events", default="", help="comma list of event indices to render")
    a = ap.parse_args()
    video = Path(a.video)
    run = ROOT / "runs" / a.name
    run.mkdir(parents=True, exist_ok=True)
    st = set(a.stages.split(","))
    T = {}

    t = time.time(); s0 = stage0(video, a.start, a.dur, run, a.asr); T["s0"] = time.time() - t
    print(f"[s0] {len(s0['cuts'])} cuts, {len(s0['transcript'])} transcript segs via {s0['asr']}"
          f"{' (ndi error: ' + s0['asr_error'][:120] + ')' if s0.get('asr_error') else ''}")
    if "s1" not in st:
        return
    t = time.time(); s1 = stage1(video, s0, run); T["s1"] = time.time() - t
    events = pick_events(s1, s0)
    (run / "events.json").write_text(json.dumps(events, indent=1))
    for i, e in enumerate(s1):
        print(f"[s1] cand {e['t'] - s0['start']:7.1f}s ({_hms(e['t'])}) {e['event']:<16} exc {e['excitement']:.2f} "
              f"recall {e['recall_score']:.1f}")
    print(f"[s1] kept {len(events)} events: " + ", ".join(f"{_hms(e['t_peak'])} {e['event']}" for e in events))
    if "s3" not in st:
        return
    t = time.time(); s3 = stage3(video, s0, events, run); T["s3"] = time.time() - t
    print(f"[s3] {sum(len(v) for v in s3.values())} keyframes decided")
    if "s4" not in st:
        return
    only = {int(x) for x in a.only_events.split(",") if x}
    for ei, e in enumerate(events):
        if only and ei not in only:
            continue
        kfs = s3.get(str(ei), [])
        if not kfs:
            continue
        plan = plan_clip(s0, e, kfs)
        (run / f"plan_{ei:02d}.json").write_text(json.dumps(plan))
        if "render" in st:
            t = time.time()
            tag = f"{ei:02d}_{_hms(e['t_peak']).replace(':', '-')}_{e['event']}"
            render(video, plan, run / f"vertical_{tag}.mp4", run / f"debug_{tag}.mp4", kfs)
            print(f"[render] {tag} in {time.time() - t:.1f}s")
    # cached stages finish in ~0s; keep the last real timing for those
    tf = run / "timing.json"
    prev = json.loads(tf.read_text()) if tf.exists() else {}
    tf.write_text(json.dumps({k: (v if v > 1 else prev.get(k, v)) for k, v in T.items()}, indent=1))


def _hms(t: float) -> str:
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


if __name__ == "__main__":
    main()
