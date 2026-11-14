"""Talk / interview profile: long landscape conversation -> vertical clips that follow the speaker.

Differences from the football pipeline (reframe/pipeline.py), which this module reuses for s0, grid
decoding, geometry, smoothing and rendering:
  s1  highlights come from WHAT IS SAID: the NDI transcript (with speaker diarization) is put into
      Drex `state` as numbered lines, and one `score` question per candidate window rates it as a
      standalone short. Windows start/end on transcript segment boundaries (never mid-sentence).
  s3  per keyframe Drex answers: shot type, where the CURRENT SPEAKER sits (left/center/right/none),
      and the grid edges of the speaker's head-and-shoulders + face focus. The transcript lines
      spoken during each 8-frame batch go into `state`.
  s4  speaker turns from NDI diarization become VIRTUAL CUTS (hard cut to the new speaker instead
      of a whip pan), kept only when the new turn lasts >= MIN_TURN_S and the framing really moves.
      Inside a sub-shot the crop is near-static (long tau, big dead zone) with a face-in-frame pin.

python3 -m reframe.talk --video interview.mp4 --name interview [--stages s0,s1,s3,s4,render] [--only-clips 0]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import media
from .geometry import ASPECT, clamp_center, fit_vertical
from .grid import Grid, decode_region, edge_questions
from .nace import drex
from .pipeline import BATCH, KF_STEP, ROOT, _cache, _hms, render, stage0
from .smooth import deadzone, solve_path, speed_cap

GRID = Grid(16, 9)
N_CLIPS = 5
WIN_MIN_S, WIN_MAX_S = 25.0, 60.0
MIN_TURN_S = 1.5          # shorter speaker turns ("yeah", "right") never trigger a cut
MIN_H_FRAC = 0.7          # 720p source: deeper zoom upscales faces too much
TP = {"tau": 1.6, "dz": 0.10, "vcap": 0.35, "safe": 0.7, "iters": 3,
      # "locked": one fixed framing per segment, changes only by hard cut (default for talk).
      # "follow": the smoothed follow-cam used for football (kept for A/B in scripts/eval_talk.py).
      "mode": "locked", "out_frac": 0.25, "min_seg_s": 2.0}

TALK_TARGET = (
    "The 'must-keep region' is the smallest rectangle that has to stay visible when this landscape "
    "interview video is reframed for a vertical phone screen: the head and shoulders of the person "
    "who is SPEAKING at this moment (judge from mouth movement, gestures, where the others look, and "
    "the transcript). If no visible person is speaking, use the most relevant listener's face. Do "
    "not add other people just to be safe."
)
TALK_FOCUS = "If only ONE spot could stay in view, it would be the face of the person speaking right now."
SHOT_TYPES = {
    "two_shot": "two people side by side fill the frame",
    "single": "one person fills most of the frame (over-the-shoulder counts)",
    "group_wide": "wide shot of the whole table or room, people small in frame",
    "other": "b-roll, graphics, hands, objects or a transition",
}
SPEAKER_POS = {
    "left": "the person speaking now is in the left third of the image",
    "center": "the person speaking now is in the middle third",
    "right": "the person speaking now is in the right third",
    "none": "the speaker is not visible, or nobody is speaking",
}


# ---------------- transcript lines: Whisper sentences + NDI speakers ----------------

def lines(s0: dict, run: Path) -> list[dict]:
    """NDI segments are coarse (median ~17 s) and their text edges can fall mid-word, but their
    speaker diarization is what we need. Whisper gives sentence-level timing. Combine: each Whisper
    sentence takes the NDI speaker it overlaps most."""
    def build():
        from .pipeline import _whisper_segments
        ws = _whisper_segments(run / "audio.m4a")
        nd = [x for x in s0["transcript"] if x.get("speaker")]
        out = []
        for w in ws:
            best, ov = None, 0.0
            for x in nd:
                o = min(w["end"], x["end"]) - max(w["start"], x["start"])
                if o > ov:
                    best, ov = x["speaker"], o
            out.append({"start": w["start"], "end": w["end"], "text": w["text"].strip(), "speaker": best})
        return out
    return _cache(run / "lines.json", build)


# ---------------- s1: highlights from the transcript ----------------

def candidate_windows(segs: list[dict]) -> list[dict]:
    """Every run of consecutive transcript segments lasting WIN_MIN_S..WIN_MAX_S."""
    wins = []
    for i in range(len(segs)):
        new_turn = i == 0 or segs[i].get("speaker") != segs[i - 1].get("speaker") or segs[i]["start"] - segs[i - 1]["end"] > 0.6
        if not new_turn:
            continue
        for j in range(i, len(segs)):
            d = segs[j]["end"] - segs[i]["start"]
            if d > WIN_MAX_S:
                break
            if d >= WIN_MIN_S:
                wins.append({"i": i, "j": j, "start": segs[i]["start"], "end": segs[j]["end"]})
                break  # shortest window from this start; the next start gives the next candidate
    return wins


def stage1(s0: dict, run: Path) -> list[dict]:
    def build():
        segs = [s for s in lines(s0, run) if s["text"].strip()]
        wins = candidate_windows(segs)
        # one Drex request per ~10 minutes of transcript: state = numbered lines, one score per window
        groups = {}
        for w in wins:
            groups.setdefault(int(w["start"] // 600), []).append(w)

        def run_group(item):
            g, ws = item
            lo = min(w["i"] for w in ws)
            hi = max(w["j"] for w in ws)
            txt = [f"[L{k}] ({_hms(s0['start'] + segs[k]['start'])}, {segs[k].get('speaker') or '?'}) "
                   f"{segs[k]['text'].strip()}" for k in range(lo, hi + 1)]
            state = ("Transcript of a long-form interview video, one numbered line per sentence "
                     "(timestamp, diarized speaker id, text):\n" + "\n".join(txt))
            q = {}
            for n, w in enumerate(ws):
                q[f"w{n}"] = {
                    "type": "score",
                    "criteria": ["unusable on its own", "weak", "decent", "strong", "must-post"],
                    "instructions": (
                        f"Consider ONLY lines L{w['i']} to L{w['j']} ({w['end'] - w['start']:.0f} s) as a "
                        "standalone vertical short for social media. Strong clips open with a hook, make "
                        "one clear self-contained point, story or surprising claim, have a quotable line "
                        "or real emotion, need no outside context, and end on a finished thought. "
                        "Penalize starting mid-thought, ending mid-sentence, or small talk.")}
            r = drex(state, q, tag="talk_s1")
            for n, w in enumerate(ws):
                a = r["answers"][f"w{n}"]
                w["score"] = a["score"]
                w["text"] = " ".join(segs[k]["text"].strip() for k in range(w["i"], w["j"] + 1))
            return ws, r.get("usage")

        with ThreadPoolExecutor(2) as ex:
            out = [w for ws, _ in ex.map(run_group, sorted(groups.items())) for w in ws]
        return out
    return _cache(run / "s1.json", build)


def refine_edges(c: dict, segs: list[dict], s0: dict) -> dict:
    """Let Drex pick where the short should start and end (choice over nearby sentence lines),
    so it opens on the first line of a thought and closes on a finished one."""
    i, j = c["i"], c["j"]
    lo, hi = max(0, i - 6), min(len(segs) - 1, j + 6)
    txt = [f"[L{k}] ({segs[k].get('speaker') or '?'}) {segs[k]['text']}" for k in range(lo, hi + 1)]
    starts = [k for k in range(max(lo, i - 5), min(j, i + 5) + 1)]
    ends = [k for k in range(max(i + 1, j - 5), min(hi, j + 5) + 1)]
    q = {"start": {"type": "choice", "criteria": {f"L{k}": None for k in starts},
                   "instructions": "Which line should a standalone vertical short START on? Pick the first line of "
                                   "a complete thought or question that hooks a viewer with no prior context. "
                                   "Never pick a line that continues a sentence or answers something not shown."},
         "end": {"type": "choice", "criteria": {f"L{k}": None for k in ends},
                 "instructions": "Which line should the short END on? Pick the line that completes the thought "
                                 "or lands the punchline. Never end mid-sentence or on a filler reply."}}
    r = drex("Transcript lines from an interview (diarized speaker id, text):\n" + "\n".join(txt), q, tag="talk_s1_edges")
    ni = int(r["answers"]["start"]["choice"][1:])
    nj = int(r["answers"]["end"]["choice"][1:])
    d = segs[nj]["end"] - segs[ni]["start"]
    if nj > ni and 15 <= d <= 75:
        c = {**c, "i": ni, "j": nj, "start": segs[ni]["start"], "end": segs[nj]["end"], "refined": True,
             "text": " ".join(segs[k]["text"] for k in range(ni, nj + 1))}
    return c


def pick_clips(s1: list[dict], s0: dict, run: Path) -> list[dict]:
    segs = [s for s in lines(s0, run) if s["text"].strip()]
    chosen = []
    for w in sorted(s1, key=lambda w: -w["score"]):
        if all(w["end"] + 5 < c["start"] or w["start"] > c["end"] + 5 for c in chosen):
            chosen.append(w)
        if len(chosen) == N_CLIPS:
            break
    with ThreadPoolExecutor(2) as ex:
        chosen = list(ex.map(lambda c: refine_edges(c, segs, s0), chosen))
    for c in chosen:
        c["clip"] = [s0["start"] + max(0.0, c["start"] - 0.2), s0["start"] + min(s0["dur"], c["end"] + 0.3)]
    return sorted(chosen, key=lambda c: c["start"])


# ---------------- s3: who is speaking, where ----------------

def stage3(video: Path, s0: dict, clips: list[dict], run: Path, cache: str = "s3.json") -> dict:
    def build():
        jobs = []
        for ci, c in enumerate(clips):
            c0, c1 = c["clip"]
            frames = media.grab_many(video, c0, c1 - c0, KF_STEP, width=1280)
            for b in range(0, len(frames), BATCH):
                jobs.append((ci, frames[b:b + BATCH]))

        def said(t0, t1):
            rel0, rel1 = t0 - s0["start"], t1 - s0["start"]
            return " ".join(f"[{s.get('speaker') or '?'}] {s['text'].strip()}" for s in lines(s0, run)
                            if s["end"] >= rel0 and s["start"] <= rel1)[:900]

        def run_batch(job):
            ci, fr = job
            imgs = [GRID.draw(im) for _, im in fr]
            q = {}
            for k in range(1, len(fr) + 1):
                q[f"f{k}_shot"] = {"type": "choice", "criteria": SHOT_TYPES,
                                   "instructions": f"What kind of shot is Image {k}?"}
                q[f"f{k}_who"] = {"type": "choice", "criteria": SPEAKER_POS,
                                  "instructions": f"In Image {k}, where is the person who is speaking at that moment?"}
                q.update(edge_questions(GRID, image_k=k, prefix=f"f{k}_", target_def=TALK_TARGET,
                                        focus_def=TALK_FOCUS))
            state = (f"{len(fr)} consecutive frames (Image 1..Image {len(fr)}, {KF_STEP}s apart) from an "
                     f"interview video that will be reframed to a 9:16 vertical crop that follows whoever is "
                     f"speaking. Transcript of what is said during these frames (diarized speaker ids): "
                     f"\"{said(fr[0][0], fr[-1][0] + KF_STEP)}\"")
            r = drex(state, q, imgs, tag="talk_s3")
            W, H = fr[0][1].size
            rows = []
            for k, (t, _) in enumerate(fr, start=1):
                dec = decode_region(r["answers"], GRID, W, H, prefix=f"f{k}_")
                rows.append({"t": t, "shot": r["answers"][f"f{k}_shot"]["choice"],
                             "who": r["answers"][f"f{k}_who"]["choice"],
                             "who_probs": r["answers"][f"f{k}_who"]["probabilities"],
                             "region_kf": dec["region"], "focus_kf": dec.get("focus"),
                             "x_unc_kf": dec["x_uncertainty_px"], "flags": dec["flags"], "kf_W": W, "kf_H": H})
            return ci, rows

        out = {}
        with ThreadPoolExecutor(2) as ex:
            for ci, rows in ex.map(run_batch, jobs):
                out.setdefault(str(ci), []).extend(rows)
        for v in out.values():
            v.sort(key=lambda r: r["t"])
        return out
    return _cache(run / cache, build)


# ---------------- s4: virtual cuts + near-static framing ----------------

def speaker_turns(s0: dict, c0: float, c1: float, run: Path) -> list[tuple[float, float, str]]:
    """(start, end, speaker) turns inside the clip, merging consecutive same-speaker sentences."""
    turns = []
    for s in lines(s0, run):
        a, b = s0["start"] + s["start"], s0["start"] + s["end"]
        if b <= c0 or a >= c1 or not s.get("speaker"):
            continue
        a, b = max(a, c0), min(b, c1)
        if turns and turns[-1][2] == s["speaker"]:
            turns[-1] = (turns[-1][0], b, s["speaker"])
        else:
            turns.append((a, b, s["speaker"]))
    return turns


def _target(k, W, H, sx):
    x0, y0, x1, y1 = [v * sx for v in k["region_kf"]]
    if k["shot"] == "other":
        return None
    fk = k.get("focus_kf")
    foc = (fk["x"] * sx, fk["y"] * sx) if fk else None
    return fit_vertical([x0, y0, x1, y1], W, H, min_h_frac=MIN_H_FRAC, focus=foc), foc


def plan_clip(s0: dict, clip: dict, kfs: list[dict], run: Path) -> dict:
    W, H, fps = s0["W"], s0["H"], s0["fps"]
    c0, c1 = clip["clip"]
    n = int(round((c1 - c0) * fps))
    times = c0 + np.arange(n) / fps
    sx = W / kfs[0]["kf_W"]
    cam_cuts = [c for c in s0["cuts"] if c0 < c < c1]

    # candidate virtual cuts at speaker turns that last long enough
    turns = speaker_turns(s0, c0, c1, run)
    virt = [a for (a, b, _), prev in zip(turns[1:], turns[:-1]) if b - a >= MIN_TURN_S and a - c0 > 0.5]

    def median_x(a, b):
        xs = [t[0]["cx"] for k in kfs if a <= k["t"] < b and (t := _target(k, W, H, sx))]
        return float(np.median(xs)) if xs else None

    # keep a virtual cut only if the framing on each side really differs (otherwise it is a no-op jump)
    bounds = sorted(set([c0] + cam_cuts + [c1]))
    kept_virtual = []
    for v in virt:
        left = max(b for b in bounds + kept_virtual if b < v)
        right = min(b for b in bounds + kept_virtual + [c1] if b > v)
        a, b = median_x(left, v), median_x(v, right)
        if a is not None and b is not None and abs(a - b) > 0.15 * W and min(v - left, right - v) >= MIN_TURN_S:
            kept_virtual.append(v)
    all_cuts = sorted(cam_cuts + kept_virtual)
    bounds = [c0] + all_cuts + [c1]

    if TP.get("mode", "locked") == "locked":
        return _plan_locked(s0, clip, kfs, bounds, cam_cuts, kept_virtual, turns, W, H, fps, n, times, sx)

    cx = np.zeros(n); cy = np.zeros(n); hh = np.zeros(n)
    shots, kf_log = [], []
    for s_a, s_b in zip(bounds[:-1], bounds[1:]):
        idx = np.where((times >= s_a) & (times < s_b))[0]
        if len(idx) == 0:
            continue
        ks = [k for k in kfs if s_a <= k["t"] < s_b] or [min(kfs, key=lambda k: abs(k["t"] - (s_a + s_b) / 2))]
        tg = [(k, *t) for k in ks if (t := _target(k, W, H, sx))]
        t0 = times[idx[0]]
        if not tg:  # b-roll / graphics: full-height center
            px = np.full(len(idx), W / 2); py = np.full(len(idx), H / 2); ph = np.full(len(idx), float(H))
        else:
            kt = [k["t"] for k, _, _ in tg]
            sig = np.array([max(k["x_unc_kf"] * sx, 1.0) for k, _, _ in tg])
            sg = list(sig / np.sqrt(np.mean(sig ** 2)))
            lam2 = (TP["tau"] * fps) ** 4 / (fps * KF_STEP)
            tx = deadzone([b["cx"] for _, b, _ in tg], TP["dz"] * W)
            ty = deadzone([b["cy"] for _, b, _ in tg], 0.08 * H)
            th = [float(np.median([b["h"] for _, b, _ in tg]))] * len(tg)  # one zoom per sub-shot
            ph = np.clip(solve_path(len(idx), fps, t0, kt, th, sg, 0.0, 0.0, lam2), MIN_H_FRAC * H, H)
            py = solve_path(len(idx), fps, t0, kt, ty, sg, 0.0, 0.0, lam2)
            vcap = TP["vcap"] * W / fps
            ckt, cv, cs = list(kt), list(tx), list(sg)
            px = speed_cap(solve_path(len(idx), fps, t0, ckt, cv, cs, 0.0, 0.0, lam2), vcap)
            fpts = [(k["t"], f[0]) for k, _, f in tg if f]
            for _ in range(TP["iters"]):  # face-in-frame pin
                added = 0
                for t, f in fpts:
                    j = int(round((t - t0) * fps))
                    if 0 <= j < len(idx):
                        safe = TP["safe"] * ph[j] * ASPECT / 2
                        if abs(f - px[j]) > safe:
                            ckt.append(t); cv.append(f - np.sign(f - px[j]) * safe * 0.8); cs.append(0.15); added += 1
                if not added:
                    break
                px = speed_cap(solve_path(len(idx), fps, t0, ckt, cv, cs, 0.0, 0.0, lam2), vcap)
            for k, b, f in tg:
                kf_log.append({"t": k["t"], "target": b["cx"], "focus": f[0] if f else None, "pinned": k["t"] in ckt[len(kt):]})
        for j, i in enumerate(idx):
            cx[i], cy[i] = clamp_center(px[j], py[j], ph[j], W, H)
            hh[i] = ph[j]
        types = [k["shot"] for k in ks]
        shots.append({"start": float(s_a), "end": float(s_b), "type": max(set(types), key=types.count),
                      "virtual": s_a in kept_virtual})
    return {"clip": clip["clip"], "fps": fps, "W": W, "H": H, "cx": cx.tolist(), "cy": cy.tolist(), "h": hh.tolist(),
            "shots": shots, "cuts": all_cuts, "camera_cuts": cam_cuts, "virtual_cuts": kept_virtual,
            "speaker_turns": turns, "kf": kf_log}


def _wmedian(vals, w):
    o = np.argsort(vals)
    v, w = np.asarray(vals)[o], np.asarray(w)[o]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, c[-1] / 2)])


def _locked_framing(tg, W, H, sx):
    """One fixed crop for a run of keyframe targets: confidence-weighted median position, median zoom."""
    w = [1.0 / max(k["x_unc_kf"] * sx, 1.0) ** 2 for k, _, _ in tg]
    h = float(np.clip(np.median([b["h"] for _, b, _ in tg]), MIN_H_FRAC * H, H))
    cx = _wmedian([b["cx"] for _, b, _ in tg], w)
    cy = _wmedian([b["cy"] for _, b, _ in tg], w)
    return clamp_center(cx, cy, h, W, H) + (h,)


def _out_frac(tg, cx, h):
    fs = [f[0] for _, _, f in tg if f]
    if not fs:
        return 0.0
    safe = TP["safe"] * h * ASPECT / 2
    return float(np.mean([abs(f - cx) > safe for f in fs]))


def _longest_out_run(tg, cx, h):
    """Longest stretch (seconds) of consecutive keyframes whose face is outside the fixed frame."""
    safe = TP["safe"] * h * ASPECT / 2
    best = run = 0
    for _, _, f in tg:
        run = run + 1 if (f and abs(f[0] - cx) > safe) else 0
        best = max(best, run)
    return best * KF_STEP


def _split_locked(tg, s_a, s_b, W, H, sx, depth=0):
    """Fixed framing for [s_a, s_b). Hard-cut once at the best change point (each side >= min_seg_s)
    and recurse when the face leaves the frame too often overall OR for >= min_seg_s in a row
    (a long shot must not dilute a real change, e.g. the other person starts talking).
    Returns [(start, end, cx, cy, h)]."""
    cx, cy, h = _locked_framing(tg, W, H, sx)
    needs_split = _out_frac(tg, cx, h) > TP["out_frac"] or _longest_out_run(tg, cx, h) >= TP["min_seg_s"]
    if depth < 3 and s_b - s_a >= 2 * TP["min_seg_s"] and needs_split:
        xs = np.array([b["cx"] for _, b, _ in tg])
        best, bt = None, None
        for i in range(1, len(tg)):
            t = tg[i][0]["t"]
            if t - s_a < TP["min_seg_s"] or s_b - t < TP["min_seg_s"]:
                continue
            cost = xs[:i].var() * i + xs[i:].var() * (len(xs) - i)
            if best is None or cost < best:
                best, bt = cost, i
        if bt is not None and abs(np.median(xs[:bt]) - np.median(xs[bt:])) > 0.10 * W:
            t = tg[bt][0]["t"]
            return (_split_locked(tg[:bt], s_a, t, W, H, sx, depth + 1)
                    + _split_locked(tg[bt:], t, s_b, W, H, sx, depth + 1))
    return [(s_a, s_b, cx, cy, h)]


def _plan_locked(s0, clip, kfs, bounds, cam_cuts, kept_virtual, turns, W, H, fps, n, times, sx):
    cx = np.zeros(n); cy = np.zeros(n); hh = np.zeros(n)
    segs, shots, kf_log = [], [], []
    for s_a, s_b in zip(bounds[:-1], bounds[1:]):
        ks = [k for k in kfs if s_a <= k["t"] < s_b] or [min(kfs, key=lambda k: abs(k["t"] - (s_a + s_b) / 2))]
        tg = [(k, *t) for k in ks if (t := _target(k, W, H, sx))]
        parts = _split_locked(tg, s_a, s_b, W, H, sx) if tg else [(s_a, s_b, W / 2, H / 2, float(H))]
        segs.extend(parts)
        types = [k["shot"] for k in ks]
        for a, b, *_ in parts:
            shots.append({"start": float(a), "end": float(b), "type": max(set(types), key=types.count),
                          "virtual": a in kept_virtual, "reframe_cut": a != s_a})
        for k, b, f in tg:
            kf_log.append({"t": k["t"], "target": b["cx"], "focus": f[0] if f else None, "pinned": False})
    for a, b, x, y, h in segs:
        idx = np.where((times >= a) & (times < b))[0]
        cx[idx], cy[idx], hh[idx] = x, y, h
    reframe_cuts = [a for a, *_ in segs if a not in bounds]
    return {"clip": clip["clip"], "fps": fps, "W": W, "H": H, "cx": cx.tolist(), "cy": cy.tolist(), "h": hh.tolist(),
            "shots": shots, "cuts": sorted(cam_cuts + kept_virtual + reframe_cuts), "camera_cuts": cam_cuts,
            "virtual_cuts": kept_virtual, "reframe_cuts": reframe_cuts, "speaker_turns": turns, "kf": kf_log,
            "mode": "locked"}


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--dur", type=float, default=0, help="0 = whole video")
    ap.add_argument("--stages", default="s0,s1,s3,s4,render")
    ap.add_argument("--only-clips", default="")
    a = ap.parse_args()
    video = Path(a.video)
    run = ROOT / "runs" / a.name
    run.mkdir(parents=True, exist_ok=True)
    dur = a.dur or media.probe(video)["duration"] - a.start
    st = set(a.stages.split(","))
    T = {}
    t = time.time(); s0 = stage0(video, a.start, dur, run, "ndi"); T["s0"] = time.time() - t
    spk = len({s.get("speaker") for s in s0["transcript"] if s.get("speaker")})
    print(f"[s0] {len(s0['cuts'])} camera cuts, {len(s0['transcript'])} transcript segs via {s0['asr']}, "
          f"{spk} chunk-local speaker ids{' | ndi error: ' + s0['asr_error'][:200] if s0.get('asr_error') else ''}")
    if "s1" not in st:
        return
    t = time.time(); s1 = stage1(s0, run); T["s1"] = time.time() - t
    sc = np.array([w["score"] for w in s1])
    print(f"[s1] {len(s1)} windows scored; score p10/p50/p90 = {np.percentile(sc, 10):.2f}/"
          f"{np.percentile(sc, 50):.2f}/{np.percentile(sc, 90):.2f}")
    ef = run / "events.json"
    clips = json.loads(ef.read_text()) if ef.exists() else pick_clips(s1, s0, run)
    ef.write_text(json.dumps(clips, indent=1))
    for ci, c in enumerate(clips):
        print(f"[s1] clip {ci} {_hms(c['clip'][0])}–{_hms(c['clip'][1])} score {c['score']:.2f}: {c['text'][:110]}…")
    if "s3" not in st:
        return
    only = {int(x) for x in a.only_clips.split(",") if x}
    sel = [c for i, c in enumerate(clips) if not only or i in only]
    cache = "s3.json" if not only else "s3_only_" + "_".join(map(str, sorted(only))) + ".json"
    t = time.time(); s3 = stage3(video, s0, sel, run, cache); T["s3"] = time.time() - t
    if "s4" not in st:
        return
    for ci, c in enumerate(clips):
        if only and ci not in only:
            continue
        key = str(ci) if not only else str(sorted(only).index(ci))
        kfs = s3.get(key, [])
        if not kfs:
            continue
        plan = plan_clip(s0, c, kfs, run)
        (run / f"plan_{ci:02d}.json").write_text(json.dumps(plan))
        print(f"[s4] clip {ci}: {len(plan['camera_cuts'])} camera cuts, {len(plan['virtual_cuts'])} speaker cuts "
              f"(from {len(plan['speaker_turns'])} turns), {len(plan.get('reframe_cuts', []))} reframe cuts")
        if "render" in st:
            tag = f"{ci:02d}_{_hms(c['clip'][0]).replace(':', '-')}"
            render(video, plan, run / f"vertical_{tag}.mp4", run / f"debug_{tag}.mp4", kfs)
            print(f"[render] {tag}")
    tf = run / "timing.json"
    prev = json.loads(tf.read_text()) if tf.exists() else {}
    tf.write_text(json.dumps({k: (v if v > 1 else prev.get(k, v)) for k, v in T.items()}, indent=1))


if __name__ == "__main__":
    main()
