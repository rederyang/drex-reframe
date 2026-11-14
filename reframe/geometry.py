"""Must-keep region -> 9:16 crop box, with zoom limits."""
from __future__ import annotations

ASPECT = 9 / 16  # w / h of the output


def fit_vertical(region, W: int, H: int, min_h_frac: float = 0.5, pad: float = 0.08,
                 focus: tuple[float, float] | None = None) -> dict:
    """Smallest 9:16 box (h in [min_h_frac*H, H]) containing `region` padded by `pad`.

    Returns cx, cy, h (box center + height) and flags. If the region is wider than a full-height
    9:16 box can hold, flag 'too_wide' and center on `focus` (ball / face) when given, keeping as
    much of the region as fits; otherwise center on the region.
    """
    x0, y0, x1, y1 = region
    rw, rh = (x1 - x0) * (1 + pad), (y1 - y0) * (1 + pad)
    h = max(rh, rw / ASPECT, min_h_frac * H)
    flags = []
    if h > H:
        if rw / ASPECT > H:
            flags.append("too_wide")
        h = H
    w = h * ASPECT
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if "too_wide" in flags and focus is not None:
        # center on the focus, then slide toward the region center while the focus stays inside
        fx = focus[0]
        cx = min(max(cx, fx - w * 0.35), fx + w * 0.35)
    cx, cy = clamp_center(cx, cy, h, W, H)
    return {"cx": cx, "cy": cy, "h": h, "w": w, "flags": flags}


def clamp_center(cx: float, cy: float, h: float, W: int, H: int) -> tuple[float, float]:
    w = h * ASPECT
    cx = min(max(cx, w / 2), W - w / 2)
    cy = min(max(cy, h / 2), H - h / 2)
    return cx, cy


def box_xyxy(cx, cy, h):
    w = h * ASPECT
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
