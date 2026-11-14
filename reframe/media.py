"""ffmpeg-based media helpers: probe, shot cuts, loudness, frame grabs, audio extraction."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"W": s["width"], "H": s["height"], "fps": float(num) / float(den),
            "duration": float(j["format"]["duration"])}


def shot_cuts(path: Path, start: float, dur: float, thresh: float = 0.3) -> list[float]:
    """Absolute timestamps of hard cuts (ffmpeg scene score) within [start, start+dur)."""
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-ss", str(start), "-t", str(dur),
                        "-i", str(path), "-an", "-vf",
                        f"scale=320:-2,select='gt(scene,{thresh})',showinfo", "-f", "null", "-"],
                       capture_output=True, text=True)
    ts = [float(m) + start for m in re.findall(r"pts_time:([\d.]+)", p.stderr)]
    # drop cuts closer than 0.4s (dissolves fire several in a row)
    out = []
    for t in ts:
        if not out or t - out[-1] > 0.4:
            out.append(t)
    return out


def loudness(path: Path, start: float, dur: float, hop: float = 0.5, sr: int = 8000) -> np.ndarray:
    """RMS dB per hop over the window. Returns array of shape (n,) aligned to start + i*hop."""
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start), "-t",
                          str(dur), "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le",
                          "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    n = int(len(x) // (hop * sr))
    x = x[: int(n * hop * sr)].reshape(n, -1)
    return 20 * np.log10(np.sqrt((x ** 2).mean(1)) + 1e-6)


def extract_audio(path: Path, start: float, dur: float, out: Path) -> Path:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start), "-t",
                    str(dur), "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k",
                    str(out)], check=True)
    return out


def grab(path: Path, t: float, width: int = 1280) -> Image.Image:
    """Single frame at absolute time t, scaled to `width` (keeps aspect)."""
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}", "-i",
                          str(path), "-frames:v", "1", "-vf", f"scale={width}:-2", "-f", "image2pipe",
                          "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    import io
    return Image.open(io.BytesIO(raw)).convert("RGB")


def grab_many(path: Path, start: float, dur: float, step: float, width: int = 1280) -> list[tuple[float, Image.Image]]:
    """Frames every `step` seconds over [start, start+dur) in one decode pass."""
    info = probe(path)
    h = int(round(info["H"] * width / info["W"] / 2) * 2)
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-t",
                          f"{dur:.3f}", "-i", str(path), "-an", "-vf", f"fps=1/{step},scale={width}:{h}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    fsz = width * h * 3
    frames = []
    for i in range(len(raw) // fsz):
        arr = np.frombuffer(raw[i * fsz:(i + 1) * fsz], dtype=np.uint8).reshape(h, width, 3)
        frames.append((start + i * step, Image.fromarray(arr)))
    return frames
