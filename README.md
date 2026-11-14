# Drex Vertical Reframe

Turns a full landscape football match into vertical (9:16) highlight clips. Drex Max decides what each frame should show, and plain code turns those decisions into a smooth camera path.

## Demo

Includes six short Scene1 clips and four recorded runs. The original match video is not required.

### Offline replay

Requires Python 3 only; no API key or network access.

```bash
python3 scripts/serve_replay.py
```

Open the [offline demo](http://127.0.0.1:8504/replay/) to explore recorded requests, responses and vertical videos.

### Live inference

```bash
conda env create -f demo/environment.yml
conda activate drex-reframe-demo
cp .env.example .env
# Edit .env and set your own NACE_API_KEY.
streamlit run demo/app.py --server.address 127.0.0.1 --server.port 8503 --server.runOnSave false
```

Open the [live demo](http://127.0.0.1:8503/) to run NACE with preset or custom viewing goals. The environment includes FFmpeg.

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
