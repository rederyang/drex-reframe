"""Numbered grid overlay + Drex question builder for 'must-keep region' edges.

Vertical lines are numbered 0..nx (left→right); horizontal lines are lettered A.. (top→bottom).
Numbers vs letters keeps the two axes from being confused.
"""
from __future__ import annotations

import math
import string
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

FONT_PATHS = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/Library/Fonts/Arial.ttf"]


def _font(size: int):
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass
class Grid:
    nx: int = 16  # segments horizontally -> nx+1 vertical lines
    ny: int = 9   # segments vertically   -> ny+1 horizontal lines

    @property
    def xlabels(self) -> list[str]:
        return [str(i) for i in range(self.nx + 1)]

    @property
    def ylabels(self) -> list[str]:
        return list(string.ascii_uppercase[: self.ny + 1])

    def x_of(self, idx: float, W: int) -> float:
        return idx / self.nx * W

    def y_of(self, idx: float, H: int) -> float:
        return idx / self.ny * H

    def draw(self, img: Image.Image) -> Image.Image:
        W, H = img.size
        out = img.convert("RGB").copy()
        ov = Image.new("RGBA", out.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        lw = max(2, W // 640)
        fs = max(14, int(min(W / self.nx, H / self.ny) * 0.32))
        f = _font(fs)
        for i, lab in enumerate(self.xlabels):
            x = round(self.x_of(i, W))
            x = min(max(x, lw // 2), W - 1 - lw // 2)
            d.line([(x, 0), (x, H)], fill=(255, 255, 0, 150), width=lw)
            for y, anchor in ((2, "mt"), (H - 2, "mb")):
                tx = min(max(x, fs), W - fs)
                d.text((tx, y), lab, font=f, fill=(255, 255, 0, 255), anchor=anchor,
                       stroke_width=3, stroke_fill=(0, 0, 0, 255))
        for j, lab in enumerate(self.ylabels):
            y = round(self.y_of(j, H))
            y = min(max(y, lw // 2), H - 1 - lw // 2)
            d.line([(0, y), (W, y)], fill=(0, 255, 255, 150), width=lw)
            for x, anchor in ((4, "lm"), (W - 4, "rm")):
                ty = min(max(y, fs), H - fs)
                d.text((x, ty), lab, font=f, fill=(0, 255, 255, 255), anchor=anchor,
                       stroke_width=3, stroke_fill=(0, 0, 0, 255))
        return Image.alpha_composite(out.convert("RGBA"), ov).convert("RGB")

    def legend(self) -> str:
        return (f"The image is overlaid with a reference grid. Yellow VERTICAL lines are numbered "
                f"0 to {self.nx} from left to right (labels at the top and bottom edges). Cyan "
                f"HORIZONTAL lines are lettered {self.ylabels[0]} to {self.ylabels[-1]} from top to "
                f"bottom (labels at the left and right edges).")


TARGET_DEF = (
    "The 'must-keep region' is the smallest rectangle that has to stay visible when this "
    "landscape football broadcast is reframed for a vertical phone screen, so that a viewer still "
    "understands what is happening right now. In live play: the ball plus the players directly "
    "involved (ball carrier, nearest defenders, intended receiver, goalkeeper/goal if a shot is "
    "on). In a close-up: the main person's face and upper body (plus a second person if they "
    "interact). Ignore scoreboards, logos and watermarks. Keep the region tight; do not include "
    "empty grass or crowd just to be safe."
)


FOCUS_DEF = ("If only ONE spot could stay in view, it would be the ball (in live play) or the main "
             "person's face (in a close-up).")


def edge_questions(grid: Grid, image_k: int = 1, prefix: str = "", target_def: str = TARGET_DEF,
                   focus_def: str = FOCUS_DEF) -> dict:
    """Six choice questions: region left/right/top/bottom + focus x/y, as grid line labels."""
    xs = {lab: None for lab in grid.xlabels}
    ys = {lab: None for lab in grid.ylabels}
    base = f"Look at Image {image_k}. {grid.legend()} {target_def} "
    return {
        f"{prefix}left": {"type": "choice", "criteria": xs,
                          "instructions": base + "Which numbered vertical line is closest to the LEFT edge of the must-keep region?"},
        f"{prefix}right": {"type": "choice", "criteria": xs,
                           "instructions": base + "Which numbered vertical line is closest to the RIGHT edge of the must-keep region?"},
        f"{prefix}top": {"type": "choice", "criteria": ys,
                         "instructions": base + "Which lettered horizontal line is closest to the TOP edge of the must-keep region?"},
        f"{prefix}bottom": {"type": "choice", "criteria": ys,
                            "instructions": base + "Which lettered horizontal line is closest to the BOTTOM edge of the must-keep region?"},
        # Focus point: used to center the crop when the region is wider than a 9:16 box can hold.
        f"{prefix}focus_x": {"type": "choice", "criteria": xs,
                             "instructions": base + focus_def + " Which numbered vertical line is closest to that spot?"},
        f"{prefix}focus_y": {"type": "choice", "criteria": ys,
                             "instructions": base + focus_def + " Which lettered horizontal line is closest to that spot?"},
    }


def _stats(probs: dict, labels: list[str]) -> tuple[float, float, int]:
    """Expected index, std (in grid units), argmax index from a choice distribution."""
    p = [float(probs.get(l, 0.0)) for l in labels]
    # Drex spreads a near-uniform floor (~0.01-0.03) over every label; with 17+ labels that tail
    # drags the mean to the center and inflates std. Subtract the median as the floor first.
    floor = sorted(p)[len(p) // 2]
    q = [max(v - floor, 0.0) for v in p]
    if sum(q) > 0:
        p = q
    s = sum(p) or 1.0
    p = [v / s for v in p]
    mean = sum(i * v for i, v in enumerate(p))
    var = sum(v * (i - mean) ** 2 for i, v in enumerate(p))
    return mean, math.sqrt(var), max(range(len(p)), key=lambda i: p[i])


def decode_region(answers: dict, grid: Grid, W: int, H: int, prefix: str = "") -> dict:
    """Turn the four edge answers into a pixel region with uncertainty."""
    out = {}
    for edge, labels, to_px, size in (("left", grid.xlabels, grid.x_of, W),
                                      ("right", grid.xlabels, grid.x_of, W),
                                      ("top", grid.ylabels, grid.y_of, H),
                                      ("bottom", grid.ylabels, grid.y_of, H)):
        a = answers[f"{prefix}{edge}"]
        mean, std, am = _stats(a.get("probabilities", {}), labels)
        out[edge] = {"choice": a.get("choice"), "argmax_idx": am, "exp_idx": mean, "std_idx": std,
                     "px": to_px(mean, size), "std_px": to_px(std, size),
                     "confidence": a.get("confidence")}
    flags = []
    x0, x1 = out["left"]["px"], out["right"]["px"]
    y0, y1 = out["top"]["px"], out["bottom"]["px"]
    if x0 >= x1:
        flags.append("inverted_x")
        x0, x1 = min(x0, x1), max(x0, x1)
    if y0 >= y1:
        flags.append("inverted_y")
        y0, y1 = min(y0, y1), max(y0, y1)
    unc = (out["left"]["std_px"] + out["right"]["std_px"]) / 2
    focus = None
    if f"{prefix}focus_x" in answers and f"{prefix}focus_y" in answers:
        fx, fxs, _ = _stats(answers[f"{prefix}focus_x"].get("probabilities", {}), grid.xlabels)
        fy, _, _ = _stats(answers[f"{prefix}focus_y"].get("probabilities", {}), grid.ylabels)
        focus = {"x": grid.x_of(fx, W), "y": grid.y_of(fy, H), "x_std_px": grid.x_of(fxs, W)}
    return {"edges": out, "region": [x0, y0, x1, y1], "flags": flags, "x_uncertainty_px": unc,
            "focus": focus}
