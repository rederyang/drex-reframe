"""Build a static results page (runs/report/) from pipeline + grid-experiment outputs.

python3 scripts/build_report.py --name clasico --variant g16f
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reframe.pipeline import _hms  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import overview_svg  # noqa: E402
import scene_svg  # noqa: E402

# Ground truth for the El Clásico window, from the broadcast commentary (transcript) + match record.
KNOWN = [
    (4885, "73′ Rakitić goal"),
    (5167, "77′ Ramos red card"),
    (5653, "85′ James equalizer"),
    (6062, "92′ Messi winner"),
]
EVENT_ZH = {"goal": "Goal", "own_goal": "Own goal", "big_chance": "Shot / chance", "card": "Card",
            "skill": "Skill move", "foul_or_stoppage": "Foul / stoppage", "none": "No event"}
CAT_ZH = {"wide": "Wide", "closeup": "Close-up", "replay": "Replay"}


def sh(*a):
    subprocess.run(a, check=True, capture_output=True)


def verdict(e):
    t = e["t_peak"]
    for kt, name in KNOWN:
        if abs(t - kt) <= 15:
            return "hit", f"Hit: {name}"
    for kt, name in KNOWN:
        if 0 < t - kt <= 90:
            return "dup", f"Duplicate: replay or celebration after {name}"
    return "extra", "Extra candidate (not one of the 4 key events)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="clasico")
    ap.add_argument("--variant", default="g16f")
    a = ap.parse_args()
    run = ROOT / "runs" / a.name
    out = ROOT / "runs" / "report"
    (out / "media").mkdir(parents=True, exist_ok=True)

    s0 = json.loads((run / "s0.json").read_text())
    s1 = json.loads((run / "s1.json").read_text())
    events = json.loads((run / "events.json").read_text())
    usage = [json.loads(l) for l in (ROOT / "runs" / "usage.jsonl").read_text().splitlines() if l.strip()]
    timing = json.loads((run / "timing.json").read_text()) if (run / "timing.json").exists() else {}

    # ---- media: re-encode clips small
    clips = []
    for ei, e in enumerate(events):
        tag = f"{ei:02d}_{_hms(e['t_peak']).replace(':', '-')}_{e['event']}"
        v, d = run / f"vertical_{tag}.mp4", run / f"debug_{tag}.mp4"
        if not v.exists():
            continue
        vo, do = out / "media" / f"v{ei:02d}.mp4", out / "media" / f"d{ei:02d}.mp4"
        sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(v), "-vf", "scale=360:640", "-c:v", "libx264",
           "-crf", "28", "-preset", "veryfast", "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(vo))
        sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(d), "-an", "-c:v", "libx264", "-crf", "28",
           "-preset", "veryfast", "-movflags", "+faststart", str(do))
        sh("ffmpeg", "-y", "-loglevel", "error", "-ss", "8", "-i", str(do), "-frames:v", "1", "-q:v", "6",
           str(out / "media" / f"dp{ei:02d}.jpg"))
        poster = out / "media" / f"p{ei:02d}.jpg"
        sh("ffmpeg", "-y", "-loglevel", "error", "-ss", "8", "-i", str(vo), "-frames:v", "1", "-q:v", "6", str(poster))
        kind, note = verdict(e)
        plan = json.loads((run / f"plan_{ei:02d}.json").read_text())
        s3k = {round(k["t"], 3): k for k in json.loads((run / "s3.json").read_text())[str(ei)]}
        c0 = plan["clip"][0]
        step = 5
        path = [[round(i / plan["fps"], 2), round(plan["cx"][i]), round(plan["h"][i] * 9 / 32)]
                for i in range(0, len(plan["cx"]), step)]
        kfp = []
        for k in plan.get("kf", []):
            src = s3k.get(round(k["t"], 3), {})
            fs = (src.get("focus_kf") or {}).get("x_std_px")
            sx = plan["W"] / src.get("kf_W", plan["W"])
            kfp.append([round(k["t"] - c0, 2), round(k["target"]), None if k["focus"] is None else round(k["focus"]),
                        None if fs is None else round(fs * sx), int(k["pinned"]), src.get("shot", "")])
        chart = {"W": plan["W"], "dur": round(plan["clip"][1] - c0, 2), "path": path, "kf": kfp,
                 "cuts": [round(c - c0, 2) for c in plan["cuts"]],
                 "shots": [[round(sh_["start"] - c0, 2), round(sh_["end"] - c0, 2), sh_["type"]] for sh_ in plan["shots"]]}
        clips.append({**e, "i": ei, "v": vo.name, "d": do.name, "p": poster.name, "kind": kind, "note": note,
                      "chart": chart})
    # center-crop baseline for the equaliser, if present
    base = run / "baseline" / "center_5655.mp4"
    has_base = base.exists()
    if has_base:
        sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(base), "-vf", "scale=360:640", "-c:v", "libx264",
           "-crf", "28", "-preset", "veryfast", "-an", "-movflags", "+faststart", str(out / "media" / "baseline.mp4"))

    # ---- grid experiment overlays
    gdir = ROOT / "runs" / "grid_exp" / a.variant
    grecs = [json.loads(l) for l in (gdir / "results.jsonl").read_text().splitlines() if l.strip()]
    for r in grecs:
        im = Image.open(gdir / "overlays" / f"{r['frame_id']}.jpg")
        im = im.resize((960, round(im.height * 960 / im.width)))
        im.save(out / "media" / f"g_{r['frame_id']}.jpg", quality=72)

    # ---- numbers
    by_tag = defaultdict(lambda: {"calls": 0, "in": 0, "img": 0, "lat": []})
    for u in usage:
        t = u["tag"]
        if t.startswith("gridexp_") and not t.endswith(a.variant):
            continue
        if t in ("smoke", "smoke2"):
            continue
        b = by_tag[t]
        b["calls"] += 1
        b["in"] += (u.get("usage") or {}).get("input_tokens", 0) or 0
        b["img"] += (u.get("usage") or {}).get("image_tokens", 0) or 0
        b["lat"].append(u["latency_s"])
    tag_zh = {"s1_confirm": "s1 event confirmation", "s3_keyframes": "s3 keyframe crop", f"gridexp_{a.variant}": "Grid experiment (30 frames)"}
    cost_rows = []
    for t, b in by_tag.items():
        lat = sorted(b["lat"])
        cost_rows.append((tag_zh.get(t, t), b["calls"], b["in"], b["img"], lat[len(lat) // 2] if lat else 0))
    hits = sum(c["kind"] == "hit" for c in clips)
    found = {n for kt, n in KNOWN if any(abs(c["t_peak"] - kt) <= 15 for c in clips)}
    n_kf = sum(len(v) for v in json.loads((run / "s3.json").read_text()).values()) if (run / "s3.json").exists() else 0
    pipe_in = sum(b["in"] for t, b in by_tag.items() if t in ("s1_confirm", "s3_keyframes"))
    pipe_calls = sum(b["calls"] for t, b in by_tag.items() if t in ("s1_confirm", "s3_keyframes"))

    data = {
        "clips": [{k: c[k] for k in ("i", "v", "d", "p", "kind", "note", "event", "excitement", "commentary", "clip", "chart")}
                  | {"peak": _hms(c["t_peak"]), "event_zh": EVENT_ZH.get(c["event"], c["event"])} for c in clips],
        "grid": [{"id": r["frame_id"], "cat": r["category"], "cat_zh": CAT_ZH.get(r["category"], r["category"]),
                  "img": f"g_{r['frame_id']}.jpg", "unc": round(r["decoded"]["x_uncertainty_px"]),
                  "wide": "too_wide" in r["box"]["flags"]} for r in grecs],
        "cands": [{"t": _hms(e["t"]), "event_zh": EVENT_ZH.get(e["event"], e["event"]),
                   "exc": round(e["excitement"], 2), "p": round(max(e["event_probs"].values()), 2),
                   "kept": any(abs(e["t_peak"] - c["t_peak"]) < 1 for c in clips)} for e in s1],
    }
    stats = {
        "window": f"{_hms(s0['start'])}–{_hms(s0['start'] + s0['dur'])}",
        "cuts": len(s0["cuts"]), "segs": len(s0["transcript"]), "asr": s0["asr"],
        "cands": len(s1), "kept": len(clips), "found": len(found), "hits": hits,
        "kf": n_kf, "pipe_calls": pipe_calls, "pipe_in": pipe_in,
        "t_s0": round(s0["timing_s"]["cuts"] + s0["timing_s"]["audio+asr"]),
        "t_s1": round(timing.get("s1", 0)), "t_s3": round(timing.get("s3", 0)),
    }
    # ---- baseline comparison on aligned clips
    cmp = None
    cf = ROOT / "runs" / "compare" / "compare.json"
    if cf.exists():
        cmp = json.loads(cf.read_text())
        gem_run = ROOT / "runs" / "gemini_aligned_clasico"
        for c in cmp["clips"]:
            ei = c["event_index"]
            for src, dst in ((ROOT / "runs" / "compare" / f"center_{ei:02d}.mp4", f"c_center_{ei:02d}.mp4"),
                             (gem_run / f"event_{ei:02d}_vertical.mp4", f"c_gem_{ei:02d}.mp4")):
                if not (out / "media" / dst).exists():
                    sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", "scale=360:640", "-c:v", "libx264",
                       "-crf", "28", "-preset", "veryfast", "-an", "-movflags", "+faststart", str(out / "media" / dst))
            c["center_v"], c["gemini_v"], c["drex_v"] = f"c_center_{ei:02d}.mp4", f"c_gem_{ei:02d}.mp4", f"v{ei:02d}.mp4"
            for v in (c["center_v"], c["gemini_v"], c["drex_v"]):
                sh("ffmpeg", "-y", "-loglevel", "error", "-ss", "8", "-i", str(out / "media" / v), "-frames:v", "1",
                   "-q:v", "6", str(out / "media" / v.replace(".mp4", ".jpg")))
            ev = events[ei]
            c["label"] = verdict(ev)[1].replace("Hit: ", "")
    # ---- per-clip Drex vs Gemini numbers for the football rows
    s3fb = json.loads((run / "s3.json").read_text())
    n_kf_all = sum(len(v) for v in s3fb.values())
    tok_kf, sec_kf = drex_per_kf("s3_keyframes", n_kf_all, timing.get("s3", 0), n_kf_all)
    cmp_by = {c["event_index"]: c for c in (cmp or {}).get("clips", [])}
    for c in data["clips"]:
        plan_c = json.loads((run / f"plan_{c['i']:02d}.json").read_text())
        c["track"] = kf_track(s3fb.get(str(c["i"]), []), plan_c, plan_c["clip"][0])
        nk = len(s3fb.get(str(c["i"]), []))
        c["drex_n"] = {"usd": nk * tok_kf * PRICE_DREX, "t": nk * sec_kf, "tokens": round(nk * tok_kf)}
        c["gem"] = gem_media(ROOT / "runs" / "gemini_aligned_clasico", c["i"], out, "fg_v")
        m = cmp_by.get(c["i"])
        if m:
            c["acc"] = {"drex": [m["drex"]["ball_in_ref_drex"], m["drex"]["ball_in_ref_gemini"]],
                        "gemini": [m["gemini"]["ball_in_ref_drex"], m["gemini"]["ball_in_ref_gemini"]]}
    talk = build_talk(out)
    tpl = (ROOT / "scripts" / "report_template.html").read_text()
    page = (tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False))
               .replace("__STATS__", json.dumps(stats, ensure_ascii=False))
               .replace("__COST__", json.dumps(cost_rows, ensure_ascii=False))
               .replace("__COMPARE__", json.dumps(cmp, ensure_ascii=False) if cmp else "null")
               .replace("__OVERVIEW__", overview_svg.build())
               .replace("__SCENE_LAND__", scene_svg.build()[0])
               .replace("__SCENE_PHONE__", scene_svg.build()[1])
               .replace("__TALK_LAND__", scene_svg.build_talk()[0])
               .replace("__TALK_PHONE__", scene_svg.build_talk()[1])
               .replace("__TALK__", json.dumps(talk, ensure_ascii=False) if talk else "null")
               .replace("__GEM_MODEL__", gem_model()))
    (out / "index.html").write_text(page)
    files = sorted(p.name for p in (out / "media").iterdir())
    (out / "files.json").write_text(json.dumps({f"media/{f}": str(out / "media" / f) for f in files}, indent=1))
    total = sum((out / "media" / f).stat().st_size for f in files)
    print(f"page ok: {len(clips)} clips, {len(grecs)} grid frames, media {total/1e6:.1f} MB in {len(files)} files")


def kf_track(kfs, plan, c0, talk=False):
    """Per-keyframe Drex intermediate results in grid units (16 x 9 cells), for the live panel."""
    W, H, fps = plan["W"], plan["H"], plan["fps"]
    rows = []
    for k in kfs:
        gx, gy = 16 / k["kf_W"], 9 / k["kf_H"]
        x0, y0, x1, y1 = k["region_kf"]
        f = k.get("focus_kf") or {}
        i = min(max(int(round((k["t"] - c0) * fps)), 0), len(plan["cx"]) - 1)
        sp = k.get("shot_probs") or {}
        r = {"t": round(k["t"] - c0, 2), "shot": k["shot"], "shot_p": round(max(sp.values()), 2) if sp else None,
             "x": [round(x0 * gx, 1), round(x1 * gx, 1)], "y": [round(y0 * gy, 1), round(y1 * gy, 1)],
             "f": [round(f["x"] * gx, 1), round(f["y"] * gy, 1)] if f else None,
             "u": round(k["x_unc_kf"] * gx, 1),
             "c": [round(plan["cx"][i] / W * 16, 1), round(plan["h"][i] / H, 2), round(plan["cx"][i])]}
        if talk:
            wp = k.get("who_probs") or {}
            r["who"], r["who_p"] = k.get("who"), round(max(wp.values()), 2) if wp else None
        rows.append(r)
    return rows


def gem_model() -> str:
    """Gemini model id actually used, read from the saved request payloads."""
    ids = {json.loads(p.read_text())["payload"]["model"]
           for p in (ROOT / "runs").glob("gemini_*/*anchors.request.json")}
    return " / ".join(sorted(ids)) or "gemini"


PRICE_DREX = 0.03 / 1e6  # drex-max-v1.1: 30 nUSD per input token


def gem_media(gem_run: Path, ei: int, out: Path, tag: str):
    """Re-encode one Gemini-directed vertical clip for the page; return its media names + cost/time."""
    src = gem_run / f"event_{ei:02d}_vertical.mp4"
    sm = {e["event_index"]: e for e in json.loads((gem_run / "summary.json").read_text())} \
        if (gem_run / "summary.json").exists() else {}
    if not src.exists() or ei not in sm:
        return None
    v, p = f"{tag}{ei:02d}.mp4", f"{tag}{ei:02d}.jpg"
    sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", "scale=360:640", "-c:v", "libx264", "-crf", "28",
       "-preset", "veryfast", "-an", "-movflags", "+faststart", str(out / "media" / v))
    sh("ffmpeg", "-y", "-loglevel", "error", "-ss", "6", "-i", str(out / "media" / v), "-frames:v", "1", "-q:v", "6",
       str(out / "media" / p))
    e = sm[ei]
    return {"v": v, "p": p, "usd": e["usd"] or 0.0, "t": e["upload_s"] + e["decide_s"],
            "tokens": (e["usage"] or {}).get("total_input_tokens", 0)}


def drex_per_kf(tag: str, n_kf_logged: int, wall_s: float, n_kf_wall: int):
    """Drex calls are 8-frame batches across clips, so per-clip cost/time is allocated by keyframes."""
    usage = [json.loads(l) for l in (ROOT / "runs" / "usage.jsonl").read_text().splitlines() if l.strip()]
    tok = sum((u["usage"] or {}).get("input_tokens", 0) for u in usage if u["tag"] == tag)
    return tok / max(n_kf_logged, 1), wall_s / max(n_kf_wall, 1)


def build_talk(out: Path, name: str = "interview"):
    """Interview / talk case: clips, speaker lanes, cost. Returns None if the run is missing."""
    run = ROOT / "runs" / name
    if not (run / "events.json").exists():
        return None
    events = json.loads((run / "events.json").read_text())
    s3 = json.loads((run / "s3.json").read_text())
    clips = []
    for ci, e in enumerate(events):
        tag = f"{ci:02d}_{_hms(e['clip'][0]).replace(':', '-')}"
        v, d = run / f"vertical_{tag}.mp4", run / f"debug_{tag}.mp4"
        if not v.exists():
            continue
        vo, do, po = out / "media" / f"t_v{ci:02d}.mp4", out / "media" / f"t_d{ci:02d}.mp4", out / "media" / f"t_p{ci:02d}.jpg"
        sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(v), "-vf", "scale=360:640", "-c:v", "libx264", "-crf", "28",
           "-preset", "veryfast", "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(vo))
        sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(d), "-an", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
           "-movflags", "+faststart", str(do))
        sh("ffmpeg", "-y", "-loglevel", "error", "-ss", "6", "-i", str(vo), "-frames:v", "1", "-q:v", "6", str(po))
        dpo = out / "media" / f"t_dp{ci:02d}.jpg"
        sh("ffmpeg", "-y", "-loglevel", "error", "-ss", "6", "-i", str(do), "-frames:v", "1", "-q:v", "6", str(dpo))
        plan = json.loads((run / f"plan_{ci:02d}.json").read_text())
        c0, c1 = plan["clip"]
        order = []
        for _, _, spk in plan["speaker_turns"]:
            if spk not in order:
                order.append(spk)
        turns = [[round(a - c0, 2), round(b - c0, 2), min(order.index(k), 3)] for a, b, k in plan["speaker_turns"]]
        words = e["text"].split()
        clips.append({
            "v": vo.name, "d": do.name, "p": po.name, "dp": dpo.name, "start": _hms(c0), "dur": round(c1 - c0),
            "score": e["score"], "n_spk": len({k for a, b, k in turns if b - a >= 1.0}),
            "head": " ".join(words[:28]) + (" …" if len(words) > 28 else ""),
            "tail": ("… " if len(words) > 28 else "") + " ".join(words[-22:]),
            "turns": turns, "cuts": [round(c - c0, 2) for c in plan["cuts"]],
            "who": [[round(k["t"] - c0, 2), k["who"]] for k in s3[str(ci)]],
            "path": [[round(i / plan["fps"], 2), round(plan["cx"][i] / plan["W"], 3)] for i in range(0, len(plan["cx"]), 6)],
            "dur_s": round(c1 - c0, 2),
        })
    # per-clip Drex vs Gemini (same clips, same locked planner)
    import eval_talk
    gem_run = ROOT / "runs" / "gemini_aligned_interview"
    n_kf_full = sum(len(v) for v in s3.values())
    extra = sum(len(v) for f in run.glob("s3_only_*.json") for v in json.loads(f.read_text()).values())
    tok_kf, sec_kf = drex_per_kf("talk_s3", n_kf_full + extra, json.loads((run / "timing.json").read_text()).get("s3", 0), n_kf_full)
    agg_rows = {"drex": [], "gemini": []}
    for ci, c in enumerate(clips):
        plan_c = json.loads((run / f"plan_{ci:02d}.json").read_text())
        c["track"] = kf_track(s3[str(ci)], plan_c, plan_c["clip"][0], talk=True)
        nk = len(s3[str(ci)])
        c["drex_n"] = {"usd": nk * tok_kf * PRICE_DREX, "t": nk * sec_kf, "tokens": round(nk * tok_kf)}
        c["gem"] = gem_media(gem_run, ci, out, "tg_v")
        gp, gk = gem_run / f"event_{ci:02d}_plan.json", gem_run / f"event_{ci:02d}_keyframes.json"
        if c["gem"] and gp.exists():
            dplan = json.loads((run / f"plan_{ci:02d}.json").read_text())
            gplan = json.loads(gp.read_text())
            gkfs = json.loads(gk.read_text())
            md_d, md_g = eval_talk.metrics(dplan, s3[str(ci)]), eval_talk.metrics(dplan, gkfs)
            mg_d, mg_g = eval_talk.metrics(gplan, s3[str(ci)]), eval_talk.metrics(gplan, gkfs)
            c["acc"] = {"drex": [md_d["face_in"], md_g["face_in"]], "gemini": [mg_d["face_in"], mg_g["face_in"]]}
            for k, m_ref_d, m_ref_g, cost in (("drex", md_d, md_g, c["drex_n"]), ("gemini", mg_d, mg_g, c["gem"])):
                agg_rows[k].append({"dur": m_ref_d["dur"], "face_d": m_ref_d["face_in"], "face_g": m_ref_g["face_in"],
                                    "pan": m_ref_d["pan_px_per_min"], "usd": cost["usd"], "t": cost["t"],
                                    "cuts": len((dplan if k == "drex" else gplan)["cuts"])})
    vs = None
    if agg_rows["gemini"]:
        def summ(rows):
            D = sum(r["dur"] for r in rows)
            return {"face_d": sum(r["face_d"] * r["dur"] for r in rows) / D,
                    "face_g": sum(r["face_g"] * r["dur"] for r in rows) / D,
                    "pan": sum(r["pan"] * r["dur"] for r in rows) / D,
                    "cuts_per_min": sum(r["cuts"] for r in rows) / D * 60,
                    "usd": sum(r["usd"] for r in rows) / len(rows), "t": sum(r["t"] for r in rows) / len(rows)}
        vs = {"drex": summ(agg_rows["drex"]), "gemini": summ(agg_rows["gemini"]), "n": len(agg_rows["gemini"])}
    usage = [json.loads(l) for l in (ROOT / "runs" / "usage.jsonl").read_text().splitlines() if l.strip()]
    def agg(tag):
        us = [u for u in usage if u["tag"] == tag]
        tok = sum((u["usage"] or {}).get("input_tokens", 0) for u in us)
        return len(us), tok, sum(u["latency_s"] for u in us)
    ndi_usd = sum(float(json.loads((run / f"ndi_part{i}.json").read_text())["cost"]["total_usd"])
                  for i in range(3) if (run / f"ndi_part{i}.json").exists())
    PRICE = 0.03 / 1e6
    n1, t1, l1 = agg("talk_s1")
    n2, t2, l2 = agg("talk_s1_edges")
    n3, t3, _ = agg("talk_s3")
    # the s3 log also holds a one-clip trial run; count only the full run
    n3, t3 = round(n3 * n_kf_full / (n_kf_full + extra)), round(tok_kf * n_kf_full)
    timing = json.loads((run / "timing.json").read_text())
    rows = [["NDI transcription + diarization (3 × 10 min, parallel)", 3, "30 min audio", "~25s", ndi_usd],
            ["Drex · score candidate windows (text)", n1, f"{t1:,} token", f"{l1:.0f}s", t1 * PRICE],
            ["Drex · pick opening and closing sentences (text)", n2, f"{t2:,} token", f"{l2:.0f}s", t2 * PRICE],
            ["Drex · per-frame speaker and framing (vision)", n3, f"{t3:,} token", f"{timing.get('s3', 0):.0f}s", t3 * PRICE]]
    drex_usd = (t1 + t2 + t3) * PRICE
    s1 = json.loads((run / "s1.json").read_text())
    # camera stability: first version (smoothed follow, saved in follow_v1/) vs current locked framing
    def stab(fmt_):
        rows = []
        for ci in range(len(events)):
            pf = run / fmt_.format(ci)
            if pf.exists():
                rows.append(eval_talk.metrics(json.loads(pf.read_text()), s3[str(ci)]))
        D = sum(r["dur"] for r in rows) or 1
        return {k: sum(r[k] * r["dur"] for r in rows) / D for k in ("pan_px_per_min", "swings_per_min", "moving_share", "face_in")}
    stability = {"follow": stab("follow_v1/plan_{:02d}.json"), "locked": stab("plan_{:02d}.json")} \
        if (run / "follow_v1").exists() else None
    return {"clips": clips, "windows": len(s1), "stability": stability, "vs": vs,
            "cost": {"rows": rows, "ndi": ndi_usd, "drex": drex_usd, "total": ndi_usd + drex_usd},
            # measured wall-clock of the runs: s0+NDI ~55s, Whisper sentences + s1 ~48s, s3 + render ~108s
            "time": {"total": 55 + 48 + 108}}


if __name__ == "__main__":
    main()
