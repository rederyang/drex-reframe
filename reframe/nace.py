"""Minimal plain-HTTPS clients for Nace Drex (systemone) and NDI (Perception Parse)."""
from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
from pathlib import Path

import requests

BASE = "https://console.nace.ai"
ROOT = Path(__file__).resolve().parents[1]
USAGE_LOG = ROOT / "runs" / "usage.jsonl"


def api_key() -> str:
    key = os.environ.get("NACE_API_KEY")
    if not key:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("NACE_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        raise RuntimeError("NACE_API_KEY missing (set env var or .env)")
    return key


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {api_key()}",
        "X-Nace-Client": "drex-reframe/0.1",
        "Content-Type": "application/json",
    }


def image_to_b64(img, fmt: str = "JPEG", quality: int = 88) -> str:
    """PIL image -> data URL. JPEG keeps inline bodies well under the 4 MiB cap."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=quality)
    mime = "jpeg" if fmt.upper() == "JPEG" else fmt.lower()
    return f"data:image/{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def _log_usage(tag: str, usage: dict | None, latency_s: float) -> None:
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with USAGE_LOG.open("a") as f:
        f.write(json.dumps({"ts": time.time(), "tag": tag, "latency_s": round(latency_s, 3),
                            "usage": usage}) + "\n")


_rate_lock = threading.Lock()
_last_call = [0.0]
MIN_INTERVAL_S = 0.6  # bursts trip the edge firewall (Vercel checkpoint 403)


def _throttle():
    with _rate_lock:
        wait = _last_call[0] + MIN_INTERVAL_S - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()


def drex(state: str, questions: dict, images: list | None = None,
         model: str = "drex-max-v1.1", tag: str = "drex", retries: int = 6) -> dict:
    """POST /v1/systemone. images: list of PIL images or data-URL strings."""
    body = {"model": model, "state": state, "questions": questions}
    if images:
        body["images"] = [im if isinstance(im, str) else image_to_b64(im) for im in images]
    payload = json.dumps(body)
    if len(payload) > 4 * 1024 * 1024:
        raise ValueError(f"request body {len(payload)/1e6:.2f} MB exceeds 4 MiB cap")
    last = None
    for attempt in range(retries):
        _throttle()
        t0 = time.time()
        r = requests.post(f"{BASE}/v1/systemone", headers=_headers(), data=payload, timeout=120)
        dt = time.time() - t0
        checkpoint = r.status_code == 403 and "Security Checkpoint" in r.text
        if r.status_code in (429, 500, 502, 503, 529) or checkpoint:
            last = r
            ra = r.headers.get("retry-after")
            try:
                ra = float(ra) if ra else 0.0
            except ValueError:
                ra = 0.0
            time.sleep(max(ra, 30.0 if checkpoint else 0.0, min(60, 3 * 2 ** attempt)))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"Drex {r.status_code}: {r.text[:800]}")
        data = r.json()
        data["_latency_s"] = dt
        _log_usage(tag, data.get("usage"), dt)
        return data
    raise RuntimeError(f"Drex retries exhausted: {last.status_code} {last.text[:400]}")


# ---------------- NDI (Perception) ----------------

def ndi_upload(path: Path) -> tuple[str, str]:
    """Upload a local file via upload grant; returns (workspace_id, file_id)."""
    h = _headers()
    g = requests.post(f"{BASE}/v1/documents/upload-grants", headers=h,
                      json={"path": path.name, "ttl_seconds": 900}, timeout=60)
    g.raise_for_status()
    grant = g.json()
    with path.open("rb") as f:
        r = requests.post(grant["upload_url"], headers={"X-Upload-Token": grant["token"]},
                          files={"file": (path.name, f)},
                          data={"metadata": json.dumps({"path": path.name,
                                                        "total_size_bytes": path.stat().st_size})},
                          timeout=600)
    if r.status_code >= 400:
        raise RuntimeError(f"NDI upload {r.status_code}: {r.text[:800]}")
    job = r.json()
    if job.get("status") not in (None, "succeeded"):
        job = ndi_wait(job["job_id"])
    file_id = job["result"]["file"]["file_id"]
    return grant["workspace_id"], file_id


def ndi_wait(job_id: str, timeout_s: int = 900) -> dict:
    delay, t0 = 1.0, time.time()
    while True:
        r = requests.get(f"{BASE}/v1/documents/jobs/{job_id}", headers=_headers(), timeout=60)
        r.raise_for_status()
        job = r.json()
        if job.get("status") in ("succeeded", "failed", "cancelled"):
            if job["status"] != "succeeded":
                raise RuntimeError(f"NDI job {job_id} {job['status']}: {job.get('error')}")
            return job
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"NDI job {job_id} still {job.get('status')}")
        time.sleep(delay)
        delay = min(delay * 1.6, 10)


def ndi_parse_file(path: Path) -> dict:
    ws, fid = ndi_upload(path)
    t0 = time.time()
    r = requests.post(f"{BASE}/v1/documents/parse?wait_seconds=60", headers=_headers(),
                      json={"source": {"type": "workspace_file", "workspace_id": ws, "file_id": fid}},
                      timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"NDI parse {r.status_code}: {r.text[:800]}")
    job = r.json()
    job = ndi_wait(job["job_id"])  # re-read by id so result.transcript is inline
    _log_usage("ndi_parse", {"cost": job.get("cost")}, time.time() - t0)
    return job
