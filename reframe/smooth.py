"""Offline camera-path planning inside one shot.

Keyframe targets (t_i, value_i, sigma_i) -> per-frame path minimizing
    sum_i w_i (x(t_i) - value_i)^2 + lam1 * sum (dx)^2 + lam2 * sum (d2x)^2
with w_i = 1 / (sigma_i^2 + sigma0^2). Uncertain Drex answers pull the path less.
A dead zone on keyframe targets avoids micro-corrections; a speed cap bounds pans.
Offline = the solve sees future keyframes, so the camera starts moving before the action.
"""
from __future__ import annotations

import numpy as np


def deadzone(values: list[float], dz: float) -> list[float]:
    """Hold the previous target unless the new one moves more than dz."""
    out, hold = [], None
    for v in values:
        if hold is None or abs(v - hold) > dz:
            hold = v
        out.append(hold)
    return out


def solve_path(n_frames: int, fps: float, t0: float, key_t, key_v, key_sigma,
               sigma0: float, lam1: float, lam2: float) -> np.ndarray:
    n = n_frames
    if n == 0:
        return np.zeros(0)
    A = np.zeros((n, n))
    b = np.zeros(n)
    for t, v, s in zip(key_t, key_v, key_sigma):
        i = int(round((t - t0) * fps))
        if 0 <= i < n:
            w = 1.0 / (s * s + sigma0 * sigma0)
            A[i, i] += w
            b[i] += w * v
    if A.trace() == 0:  # no keyframes inside: hold mean
        return np.full(n, float(np.mean(key_v)) if len(key_v) else 0.0)
    if n > 1:
        D1 = np.diff(np.eye(n), 1, axis=0)
        A += lam1 * D1.T @ D1
    if n > 2:
        D2 = np.diff(np.eye(n), 2, axis=0)
        A += lam2 * D2.T @ D2
    A += np.eye(n) * 1e-9
    return np.linalg.solve(A, b)


def speed_cap(x: np.ndarray, max_per_frame: float) -> np.ndarray:
    y = x.copy()
    for i in range(1, len(y)):
        d = y[i] - y[i - 1]
        if abs(d) > max_per_frame:
            y[i] = y[i - 1] + np.sign(d) * max_per_frame
    # symmetric backward pass so the cap doesn't only lag
    for i in range(len(y) - 2, -1, -1):
        d = y[i] - y[i + 1]
        if abs(d) > max_per_frame:
            y[i] = y[i + 1] + np.sign(d) * max_per_frame
    return y
