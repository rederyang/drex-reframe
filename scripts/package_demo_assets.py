"""Maintainer tool: cut six portable clips; the original video is never bundled."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reframe.media import probe

SPECS = [
    ("wing", "Wing play · 00:12", 12.0),
    ("wide", "Wide play · 00:32", 32.0),
    ("touchline", "Touchline · 01:00", 60.0),
    ("goal", "Goal area · 02:00", 120.0),
    ("midfield", "Midfield · 05:00", 300.0),
    ("goal-view", "Goal view · 10:00", 600.0),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", type=Path, required=True, help="Original Scene1 directory, used only when regenerating assets")
    parser.add_argument("--destination", type=Path, default=ROOT / "demo" / "assets")
    args = parser.parse_args()
    original = json.loads((args.source_scene / "s0.json").read_text())
    video = args.source_scene / "el-clasico-2017_01h12m30s-01h42m30s.mp4"
    target = args.destination.resolve()
    (target / "scene1").mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "dataset": "Scene1 / 2017 El Clásico", "pipeline_base": "ab86fd8a75bff844b0fc77c73ef5340e5dfdc80a", "clips": []}
    for identity, label, start in SPECS:
        path = target / "scene1" / f"{identity}.mp4"
        if not path.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", str(start), "-i", str(video), "-t", "4",
                            "-map", "0:v:0", "-map", "0:a:0", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                            "-threads", "2", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(path)], check=True)
        metadata = probe(path)
        if abs(metadata["duration"] - 4) > 0.15:
            raise ValueError(f"Unexpected clip duration: {identity}")
        offset = float(original["start"]) + start
        manifest["clips"].append({"id": identity, "label": label, "file": f"scene1/{identity}.mp4", "scene_start": start,
                                  "source_start": offset, "duration": 4.0, "W": metadata["W"], "H": metadata["H"], "fps": metadata["fps"],
                                  "cuts": [float(t) - offset for t in original["cuts"] if offset < t < offset + 4],
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(f"Packaged {label}: {path.stat().st_size / 1e6:.2f} MB", flush=True)
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
