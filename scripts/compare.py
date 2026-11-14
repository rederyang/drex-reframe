"""Compare crop deciders on the SAME clips: center crop (no model), Gemini director, Drex grid.

No human labels exist, so "ball in frame" is measured against each model's own ball/face estimate
AND against the other model's, and reported for both references. Smoothness uses the same metric
as scripts/eval_path.py.

python3 scripts/compare.py --drex-run clasico --gemini-run gemini_aligned_clasico --events 1,3,5,6
Writes runs/compare/compare.json and renders center-crop clips.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reframe.gemini_baseline import api_cost_estimate  # noqa: E402
from reframe.pipeline import render  # noqa: E402

# drex-max-v1.1: 30 nUSD per input token, output not billed.
# Source: https://console.nace.ai/docs/drex/reference/pricing ("Worked examples"), checked 2026-11-14.
DREX_USD_PER_M_INPUT = 0.03


def ball_in(plan, ref_kfs):
    """Share of live-play reference keyframes whose ball/face point lies inside the crop."""
    W, fps, c0 = plan["W"], plan["fps"], plan["clip"][0]
    cx, h = np.array(plan["cx"]), np.array(plan["h"])
    n = inside = 0
    for k in ref_kfs:
        if k["shot"] not in ("wide", "medium") or not k.get("focus_kf"):
            continue
        i = int(round((k["t"] - c0) * fps))
        if not 0 <= i < len(cx):
            continue
        fx = k["focus_kf"]["x"] * W / k["kf_W"]
        n += 1
        inside += abs(fx - cx[i]) <= h[i] * 9 / 32
    return inside, n


def motion(plan):
    W, fps, c0 = plan["W"], plan["fps"], plan["clip"][0]
    cx = np.array(plan["cx"])
    b = [0] + [int(round((c - c0) * fps)) for c in plan["cuts"]] + [len(cx)]
    swings, dur = 0, 0.0
    for a_, b_ in zip(b[:-1], b[1:]):
        seg = cx[a_:b_]
        if len(seg) < 5:
            continue
        ext = [seg[0]]
        for x in seg[1:]:
            if len(ext) >= 2 and np.sign(x - ext[-1]) == np.sign(ext[-1] - ext[-2]):
                ext[-1] = x
            elif abs(x - ext[-1]) >= 0.05 * W or len(ext) == 1:
                ext.append(x)
        swings += max(0, len(ext) - 2)
        dur += len(seg) / fps
    return swings, dur


def center_plan(drex_plan):
    p = dict(drex_plan)
    n = len(p["cx"])
    p["cx"] = [p["W"] / 2] * n
    p["cy"] = [p["H"] / 2] * n
    p["h"] = [p["H"]] * n
    p["kf"] = []
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drex-run", default="clasico")
    ap.add_argument("--gemini-run", default="gemini_aligned_clasico")
    ap.add_argument("--events", default="", help="default: every clip that has a Gemini plan")
    a = ap.parse_args()
    drex = ROOT / "runs" / a.drex_run
    gem = ROOT / "runs" / a.gemini_run
    out = ROOT / "runs" / "compare"
    out.mkdir(parents=True, exist_ok=True)
    s0 = json.loads((drex / "s0.json").read_text())
    s3 = json.loads((drex / "s3.json").read_text())
    usage = [json.loads(l) for l in (ROOT / "runs" / "usage.jsonl").read_text().splitlines() if l.strip()]
    s3u = [u for u in usage if u["tag"] == "s3_keyframes"]
    n_clips_s3 = len(s3)
    timing = json.loads((drex / "timing.json").read_text())
    gsum = {e["event_index"]: e for e in json.loads((gem / "summary.json").read_text())}
    idx = ([int(x) for x in a.events.split(",")] if a.events
           else sorted(i for i in gsum if (gem / f"event_{i:02d}_plan.json").exists()))

    acc = {m: {"in_d": 0, "n_d": 0, "in_g": 0, "n_g": 0, "sw": 0, "dur": 0.0} for m in ("center", "gemini", "drex")}
    clips = []
    for ei in idx:
        dplan = json.loads((drex / f"plan_{ei:02d}.json").read_text())
        gplan = json.loads((gem / f"event_{ei:02d}_plan.json").read_text())
        gkfs = json.loads((gem / f"event_{ei:02d}_keyframes.json").read_text())
        dkfs = s3[str(ei)]
        cplan = center_plan(dplan)
        cv = out / f"center_{ei:02d}.mp4"
        if not cv.exists():
            render(Path(s0["video"]), cplan, cv, out / f"center_{ei:02d}_debug.mp4", [])
        row = {"event_index": ei, "clip": dplan["clip"]}
        for m, p in (("center", cplan), ("gemini", gplan), ("drex", dplan)):
            i_d, n_d = ball_in(p, dkfs)
            i_g, n_g = ball_in(p, gkfs)
            sw, dur = motion(p)
            A = acc[m]
            A["in_d"] += i_d; A["n_d"] += n_d; A["in_g"] += i_g; A["n_g"] += n_g; A["sw"] += sw; A["dur"] += dur
            row[m] = {"ball_in_ref_drex": i_d / max(n_d, 1), "ball_in_ref_gemini": i_g / max(n_g, 1),
                      "swings_per_min": sw / dur * 60 if dur else 0}
        clips.append(row)

    # cost / latency per clip
    d_tok = sum((u["usage"] or {}).get("input_tokens", 0) for u in s3u) / n_clips_s3
    d_lat = timing.get("s3", 0) / n_clips_s3
    g_in = [gsum[i]["usage"].get("total_input_tokens", 0) for i in idx]
    g_out = [gsum[i]["usage"].get("total_output_tokens", 0) + gsum[i]["usage"].get("total_thought_tokens", 0) for i in idx]
    g_usd = [gsum[i]["usd"] or 0 for i in idx]
    g_lat = [gsum[i]["upload_s"] + gsum[i]["decide_s"] for i in idx]
    summary = {}
    for m in acc:
        A = acc[m]
        summary[m] = {"ball_in_ref_drex": A["in_d"] / max(A["n_d"], 1), "ball_in_ref_gemini": A["in_g"] / max(A["n_g"], 1),
                      "n_ref_drex": A["n_d"], "n_ref_gemini": A["n_g"], "swings_per_min": A["sw"] / A["dur"] * 60}
    summary["center"].update({"latency_s_per_clip": 0, "tokens_per_clip": 0, "usd_per_clip": 0})
    summary["gemini"].update({"latency_s_per_clip": float(np.mean(g_lat)), "tokens_per_clip": float(np.mean(g_in)),
                              "out_tokens_per_clip": float(np.mean(g_out)), "usd_per_clip": float(np.mean(g_usd))})
    summary["drex"].update({"latency_s_per_clip": d_lat, "tokens_per_clip": d_tok,
                            "usd_per_clip": d_tok * DREX_USD_PER_M_INPUT / 1e6,
                            "usd_per_m_input": DREX_USD_PER_M_INPUT})

    # event discovery comparison (temporal grounding)
    known = [(4885, "73′ Rakitić goal"), (5167, "77′ Ramos red card"), (5653, "85′ James equalizer"), (6062, "92′ Messi winner")]
    b1 = ROOT / "runs" / "gemini_hydrogen_clasico_static_v1"
    b1_events = json.loads((b1 / "events.json").read_text())
    b1_usage = [json.loads(l) for l in (b1 / "usage.jsonl").read_text().splitlines() if l.strip()]
    b1_ev = next(u for u in b1_usage if u.get("tag") == "events")
    off = 4350.0
    gem_ev = []
    for e in b1_events:
        t = e["peak_s"] + off
        # match by what Gemini SAYS happened (its description), not by nearest time
        claim = {"Rakitic": 0, "Ramos": 1, "red card": 1, "James": 2, "Messi": 3}
        ki = next((v for k, v in claim.items() if k.lower() in e["description"].lower()), None)
        near = known[ki] if ki is not None else min(known, key=lambda k: abs(k[0] - t))
        gem_ev.append({"peak_abs": t, "event": e["event"], "desc": e["description"],
                       "nearest_known": near[1], "err_s": t - near[0]})
    d_events = json.loads((drex / "events.json").read_text())
    drex_found = [k[1] for k in known if any(abs(e["t_peak"] - k[0]) <= 15 for e in d_events)]
    s1u = [u for u in usage if u["tag"] == "s1_confirm"]
    discovery = {
        "drex": {"found_on_time": len(drex_found), "of": 4, "calls": len(s1u),
                 "tokens": sum((u["usage"] or {}).get("input_tokens", 0) for u in s1u),
                 "usd": sum((u["usage"] or {}).get("input_tokens", 0) for u in s1u) * DREX_USD_PER_M_INPUT / 1e6,
                 "latency_s": timing.get("s1", 0) + 70.0, "latency_note": "s0 local transcription ~70s + s1 26s"},
        "gemini": {"picked": len(b1_events), "on_time": sum(abs(g["err_s"]) <= 15 for g in gem_ev),
                   "calls": 1, "tokens": b1_ev["usage"]["total_input_tokens"], "latency_s": b1_ev["elapsed_s"],
                   "usd": (api_cost_estimate(b1_ev["usage"], "gemini-4-hydrogen") or {}).get("usd"),
                   "events": gem_ev},
    }
    res = {"clips": clips, "summary": summary, "discovery": discovery}
    (out / "compare.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for m, s in summary.items():
        print(f"{m:<7} ball_in(ref Drex) {s['ball_in_ref_drex']:.0%}  ball_in(ref Gemini) {s['ball_in_ref_gemini']:.0%}  "
              f"swings {s['swings_per_min']:.1f}/min  latency {s['latency_s_per_clip']:.1f}s  tokens {s['tokens_per_clip']:.0f}  "
              f"usd {s.get('usd_per_clip')}")
    print("drex usd/clip", summary["drex"]["usd_per_clip"], "vs gemini", summary["gemini"]["usd_per_clip"])
    print(json.dumps(discovery, ensure_ascii=False, indent=1)[:1500])


if __name__ == "__main__":
    main()
