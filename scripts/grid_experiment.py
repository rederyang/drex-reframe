"""Stage-3 feasibility: can Drex read the must-keep region off a numbered grid?

Runs one Drex request per keyframe (4 choice questions: left/right/top/bottom), then writes:
  runs/grid_exp/<variant>/results.jsonl   raw answers + decoded region + 9:16 box
  runs/grid_exp/<variant>/overlays/*.jpg  frame | predicted region (yellow) + 9:16 box (cyan) | vertical crop
  runs/grid_exp/<variant>/contact.jpg     all overlays grouped by category
If samples/football-keyframes/labels.json exists (from tools/label.html), also scores against it.

Usage: python3 scripts/grid_experiment.py --variant g16 [--limit N]
Variants: g16 (17x10 lines), g32 (33x19 lines), clean_g16 (clean frame + gridded frame, 2 images)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reframe.geometry import box_xyxy, fit_vertical  # noqa: E402
from reframe.grid import Grid, edge_questions, decode_region  # noqa: E402
from reframe.nace import drex  # noqa: E402

SAMPLES = ROOT / "samples" / "football-keyframes"
VARIANTS = {"g16": (16, 9, False), "g16f": (16, 9, False), "g32": (32, 18, False), "clean_g16": (16, 9, True)}


def run_one(row: dict, variant: str) -> dict:
    nx, ny, with_clean = VARIANTS[variant]
    grid = Grid(nx, ny)
    img = Image.open(SAMPLES / row["file"]).convert("RGB")
    W, H = img.size
    gridded = grid.draw(img)
    images = [img, gridded] if with_clean else [gridded]
    k = 2 if with_clean else 1
    state = ("Frames from a professional football (soccer) match broadcast that will be reframed "
             "from 16:9 landscape to a 9:16 vertical crop.")
    if with_clean:
        state += " Image 1 is the clean frame; Image 2 is the identical frame with a reference grid."
    res = drex(state, edge_questions(grid, image_k=k), images, tag=f"gridexp_{variant}")
    dec = decode_region(res["answers"], grid, W, H)
    foc = (dec["focus"]["x"], dec["focus"]["y"]) if dec.get("focus") else None
    box = fit_vertical(dec["region"], W, H, focus=foc)
    return {"frame_id": row["frame_id"], "file": row["file"], "category": row["category"],
            "shot_scale": row["shot_scale"], "variant": variant, "W": W, "H": H,
            "answers": res["answers"], "usage": res.get("usage"), "latency_s": res["_latency_s"],
            "decoded": dec, "box": box}


def overlay(rec: dict, out: Path) -> Image.Image:
    img = Image.open(SAMPLES / rec["file"]).convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = rec["decoded"]["region"]
    d.rectangle([x0, y0, x1, y1], outline=(255, 230, 0), width=4)
    bx = box_xyxy(rec["box"]["cx"], rec["box"]["cy"], rec["box"]["h"])
    d.rectangle(bx, outline=(0, 220, 255), width=4)
    f = rec["decoded"].get("focus")
    if f:
        d.ellipse([f["x"] - 9, f["y"] - 9, f["x"] + 9, f["y"] + 9], outline=(255, 60, 60), width=4)
    e = rec["decoded"]["edges"]
    txt = (f"{rec['frame_id']}  L{e['left']['choice']} R{e['right']['choice']} "
           f"T{e['top']['choice']} B{e['bottom']['choice']}  ±{rec['decoded']['x_uncertainty_px']:.0f}px "
           f"{' '.join(rec['decoded']['flags'] + rec['box']['flags'])}")
    d.rectangle([0, H - 30, W, H], fill=(0, 0, 0))
    d.text((8, H - 26), txt, fill=(255, 255, 255))
    crop = Image.open(SAMPLES / rec["file"]).convert("RGB").crop([round(v) for v in bx])
    crop = crop.resize((round(H * 9 / 16), H))
    canvas = Image.new("RGB", (W + crop.width + 10, H), (20, 20, 20))
    canvas.paste(img, (0, 0))
    canvas.paste(crop, (W + 10, 0))
    canvas.save(out, quality=85)
    return canvas


def contact(recs: list[dict], ovdir: Path, out: Path, thumb_w: int = 480) -> None:
    recs = sorted(recs, key=lambda r: (r["category"], r["frame_id"]))
    thumbs = []
    for r in recs:
        im = Image.open(ovdir / f"{r['frame_id']}.jpg")
        thumbs.append(im.resize((thumb_w, round(im.height * thumb_w / im.width))))
    cols = 3
    th = thumbs[0].height
    sheet = Image.new("RGB", (cols * thumb_w, ((len(thumbs) + cols - 1) // cols) * th), (0, 0, 0))
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * thumb_w, (i // cols) * th))
    sheet.save(out, quality=82)


def score(recs: list[dict], labels: dict) -> dict:
    """Edge error (px) and whether the 9:16 box contains the labeled region."""
    rows = []
    for r in recs:
        lab = labels.get(r["frame_id"])
        if not lab:
            continue
        lx0, ly0, lx1, ly1 = lab["region"]
        px0, py0, px1, py1 = r["decoded"]["region"]
        bx0, by0, bx1, by1 = box_xyxy(r["box"]["cx"], r["box"]["cy"], r["box"]["h"])
        inter = max(0, min(bx1, lx1) - max(bx0, lx0)) * max(0, min(by1, ly1) - max(by0, ly0))
        area = (lx1 - lx0) * (ly1 - ly0) or 1
        lcx, pcx = (lx0 + lx1) / 2, (px0 + px1) / 2
        rows.append({"frame_id": r["frame_id"], "category": r["category"],
                     "center_err_px": abs(lcx - pcx), "center_err_frac_cropw": abs(lcx - pcx) / r["box"]["w"],
                     "label_coverage": inter / area})
    if not rows:
        return {}
    by = {}
    for row in rows:
        by.setdefault(row["category"], []).append(row)
        by.setdefault("all", []).append(row)
    summ = {c: {"n": len(v),
                "mean_center_err_px": sum(x["center_err_px"] for x in v) / len(v),
                "coverage>=0.9": sum(x["label_coverage"] >= 0.9 for x in v) / len(v)}
            for c, v in by.items()}
    return {"per_frame": rows, "summary": summ}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="g16", choices=list(VARIANTS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rescore", action="store_true", help="skip API calls; reuse results.jsonl")
    a = ap.parse_args()

    outdir = ROOT / "runs" / "grid_exp" / a.variant
    ovdir = outdir / "overlays"
    ovdir.mkdir(parents=True, exist_ok=True)
    resf = outdir / "results.jsonl"

    if a.rescore:
        recs = [json.loads(l) for l in resf.read_text().splitlines()]
    else:
        rows = list(csv.DictReader((SAMPLES / "manifest.csv").open(encoding="utf-8-sig")))
        if a.limit:
            rows = rows[: a.limit]
        with ThreadPoolExecutor(a.workers) as ex:
            recs = list(ex.map(lambda r: run_one(r, a.variant), rows))
        resf.write_text("\n".join(json.dumps(r) for r in recs) + "\n")

    for r in recs:
        overlay(r, ovdir / f"{r['frame_id']}.jpg")
    contact(recs, ovdir, outdir / "contact.jpg")

    tok = sum((r["usage"] or {}).get("input_tokens", 0) for r in recs)
    lat = sorted(r["latency_s"] for r in recs)
    flags = sum(bool(r["decoded"]["flags"]) for r in recs)
    print(f"{len(recs)} frames | input_tokens total {tok} (~{tok/len(recs):.0f}/frame) | "
          f"latency p50 {lat[len(lat)//2]:.2f}s max {lat[-1]:.2f}s | inverted-edge frames {flags}")
    for r in recs:
        e = r["decoded"]["edges"]
        print(f"  {r['frame_id']:<34} L{e['left']['choice']:>2} R{e['right']['choice']:>2} "
              f"T{e['top']['choice']} B{e['bottom']['choice']}  x±{r['decoded']['x_uncertainty_px']:5.1f}px "
              f"{' '.join(r['decoded']['flags'] + r['box']['flags'])}")

    labf = SAMPLES / "labels.json"
    if labf.exists():
        s = score(recs, json.loads(labf.read_text()))
        (outdir / "score.json").write_text(json.dumps(s, indent=2))
        print("score:", json.dumps(s.get("summary"), indent=2))
    else:
        print("no labels.json yet -> open tools/label.html to label ground truth")


if __name__ == "__main__":
    main()
