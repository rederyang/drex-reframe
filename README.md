# Drex Vertical Reframe

Turns a full landscape football match into vertical (9:16) highlight clips. Drex Max decides what each frame should show, and plain code turns those decisions into a smooth camera path.

## Portable Scene1 demo

This branch includes everything needed to demonstrate the main pipeline: **six four-second input clips**, four real recorded runs, exact model-input JPEGs, responses, probabilities, crop previews and rendered videos. The full match, API keys and machine-specific paths are not included. Git LFS is not required.

```bash
git clone --branch demo/portable-scene1 --single-branch https://github.com/rederyang/drex-reframe.git
cd drex-reframe
```

### Offline replay — Python only, no key or network

```bash
python3 scripts/serve_replay.py
```

Open **http://127.0.0.1:8504/replay/**. Choose a clip and recorded focus, inspect the request/response and probabilities, or replay the source and vertical video together. This mode clearly displays `REPLAY` and sends no model requests. It needs only Python 3's standard library, with no pip packages or FFmpeg installation.

### Live demo — custom goals and real NACE calls

```bash
conda env create -f demo/environment.yml
conda activate drex-reframe-demo
cp .env.example .env
# Edit .env and set your own NACE_API_KEY.
streamlit run demo/app.py --server.address 127.0.0.1 --server.port 8503 --server.runOnSave false
```

Open **http://127.0.0.1:8503**. The Conda environment includes Python 3.12 and FFmpeg. If using an existing Python environment instead, install `demo/requirements.txt` and make `ffmpeg` / `ffprobe` available on PATH.

The English interface offers six fixed clips, preset goals and free text. Eight frames are sampled at 0.5-second intervals. One NACE request asks seven Choice questions per frame: shot type, four region edges and focus x/y. It directly calls `prepare_keyframe_batch()` / `detect_keyframe_batch()`, the existing probability decoder, `plan_clip()` and `render()` to produce a 720 × 1280 video with audio. The no-goal request matches the main implementation at `ab86fd8`; custom goals add context to state without replacing the workflow's logic. Switching frames or views never queries NACE. Changing the clip or goal invalidates the current result.

The live app reads **only this clone's** `.env` or `NACE_API_KEY` environment variable. Each new run is saved under ignored `runs/region_demo/<id>/`; no credentials are saved. All media defaults to `demo/assets/`. Processing times are local to each four-second clip; the UI retains its original Scene1 timestamp through the asset manifest.

### Bundled files and verification

| Path | Contents |
|---|---|
| `demo/assets/manifest.json` | Clip metadata, original offsets, cut points and SHA-256 checksums |
| `demo/assets/scene1/` | Six four-second MP4 clips, each under 2 MB |
| `demo/assets/recordings/` | Four real runs: model images, previews, vertical and comparison videos |
| `demo/replay/` | Standalone replay interface and sanitized recorded request/response JSON |
| `demo/environment.yml` | Live-demo environment including FFmpeg |

Run `python -m unittest discover -s tests -v` in the live environment to verify assets, decoding, rendering and UI behavior. The parity check against the original main stage is skipped only when its baseline commit is absent from a source export or shallow clone.

Maintainers can regenerate input clips with `python scripts/package_demo_assets.py --source-scene /path/to/original/scene1`. This command needs the original video **only for regeneration**, not for running either demo.

## Pipeline (`reframe/pipeline.py`)

| Stage | What runs | Model calls |
|---|---|---|
| s0 structure | ffmpeg scene cuts, RMS loudness, transcript (NDI Parse, falls back to Whisper `base.en`) | NDI ×1 |
| s1 events | Recall candidates from commentary keywords and loudness peaks. Then 8 frames, plus nearby commentary, go to Drex as `choice` (event type) + `score` (excitement) + `choice` (peak frame) | Drex ×1 per candidate |
| s3 keyframes | Every 0.5 s inside each event clip. Each gridded frame gets 7 Choice questions: shot type, four must-keep region edges, and focus x/y. 8 consecutive frames go in one request | Drex ×1 per 4 s of clip |
| s4 plan | Must-keep region → smallest 9:16 box within the zoom limits. Each shot is smoothed with a weighted least-squares solve: weight = 1/σ² taken from the Drex `probabilities`, plus a dead zone and a speed cap. Shot boundaries are hard cuts | – |
| render | `vertical_*.mp4` (720×1280 with audio) and a `debug_*.mp4` side-by-side view (source with Drex region in yellow and smoothed crop in cyan, next to the vertical output) | – |

The design's Stage 2 (shot classification) is folded into the s3 requests.

```bash
python3 -m reframe.pipeline --video samples/football-keyframes/videos/el-clasico-2017.mp4 \
  --start 4350 --dur 1800 --name clasico
```

Each stage caches its output under `runs/<name>/s*.json`. Delete a cache file to rerun that stage. Every API call appends its usage and latency to `runs/usage.jsonl`.

## Grid feasibility experiment

```bash
python3 scripts/grid_experiment.py --variant g16      # also: g32, clean_g16
```

Sends one request per keyframe (30 frames). Output goes to `runs/grid_exp/<variant>/`: `results.jsonl`, per-frame overlays and `contact.jpg`.

**These frames have no ground truth yet.** To create it, open `tools/label.html`, drag a box around the must-keep region on each frame, export the result, and save it as `samples/football-keyframes/labels.json`. Then run `--rescore`, which reports center error and label coverage of the 9:16 crop.

## Setup

- Put the API key in `.env` as `NACE_API_KEY=...`. The file is gitignored.
- Requirements: `ffmpeg`, Python 3 with `requests`, `numpy`, `pillow`, `openai-whisper` (fallback only).
- Requests are throttled to one every 0.6 s. A burst of 6 parallel requests triggered the console's Vercel security checkpoint (HTTP 403).
