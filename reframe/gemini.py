"""Gemini Files + Interactions REST client; credentials never enter run artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
HOST = "https://generativelanguage.googleapis.com"


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Gemini:
    def __init__(self, run: Path, model="gemini-4-hydrogen"):
        self.run, self.model = run, model
        self.run.mkdir(parents=True, exist_ok=True)
        env = {}
        if (ROOT / ".env").exists():
            for line in (ROOT / ".env").read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip("\"'")
        self.key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or env.get("GEMINI_API_KEY")
        if not self.key:
            raise RuntimeError("GEMINI_API_KEY missing; add it to the project .env")
        self.session = requests.Session()
        self.session.headers.update({"x-goog-api-key": self.key, "Api-Revision": "2026-05-20"})

    def log(self, value):
        with (self.run / "usage.jsonl").open("a") as f:
            f.write(json.dumps({"time": time.time(), **value}) + "\n")

    def request(self, method, path, **kwargs):
        start = time.monotonic()
        try:
            r = self.session.request(method, HOST + path, timeout=(30, 600), **kwargs)
        except requests.RequestException as e:
            # No automatic POST replay: an ambiguous timeout may already have been billed.
            self.log({"kind": "http_error", "method": method, "path": path,
                      "elapsed_s": time.monotonic() - start, "error_type": type(e).__name__})
            raise RuntimeError(f"Gemini {type(e).__name__}; request not automatically repeated") from None
        if not r.ok:
            self.log({"kind": "http_error", "method": method, "path": path, "status": r.status_code})
            raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text.replace(self.key, '[REDACTED]')[:800]}")
        return r

    def upload(self, path: Path, tag: str) -> dict:
        cache = self.run / f"file_{tag}.json"
        sha = digest(path)
        start = time.monotonic()
        file = None
        if cache.exists():
            prior = json.loads(cache.read_text())
            if prior["sha256"] == sha:
                r = self.session.get(HOST + "/v1beta/" + prior["file"]["name"], timeout=30)
                if r.ok:
                    file = r.json()
                elif r.status_code not in (404, 410):
                    raise RuntimeError(f"Cannot check uploaded file: HTTP {r.status_code}")
        if file is None:
            print(f"[upload] {tag}: {path.stat().st_size / 1e6:.1f} MB", flush=True)
            r = self.request("POST", "/upload/v1beta/files", headers={
                "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(path.stat().st_size),
                "X-Goog-Upload-Header-Content-Type": "video/mp4",
            }, json={"file": {"display_name": f"reframe-{tag}"}})
            upload_url = r.headers["X-Goog-Upload-URL"]
            with path.open("rb") as f:
                r = requests.post(upload_url, headers={"Content-Length": str(path.stat().st_size),
                    "X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"},
                    data=f, timeout=(30, 1800))
            if not r.ok:
                raise RuntimeError(f"File transfer failed: HTTP {r.status_code}")
            file = r.json()["file"]
            save(cache, {"sha256": sha, "file": file})
        deadline = time.monotonic() + 1800
        while file.get("state") != "ACTIVE":
            if file.get("state") == "FAILED":
                raise RuntimeError("Gemini video processing failed")
            if time.monotonic() > deadline:
                raise RuntimeError("Video still processing; rerun to resume")
            print(f"[processing] {tag}: {file.get('state')}", flush=True)
            time.sleep(10)
            file = self.request("GET", "/v1beta/" + file["name"]).json()
        save(cache, {"sha256": sha, "file": file})
        self.log({"kind": "upload_ready", "tag": tag, "bytes": path.stat().st_size,
                  "elapsed_s": time.monotonic() - start})
        return file

    def ask(self, file: dict, prompt: str, schema: dict, tag: str, processing="static"):
        payload = {"model": self.model, "input": [
            {"type": "video", "uri": file["uri"], "mime_type": "video/mp4", "processing": processing},
            {"type": "text", "text": prompt}],
            "response_format": {"type": "text", "mime_type": "application/json", "schema": schema},
            "background": True}
        signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        req, res = self.run / f"{tag}.request.json", self.run / f"{tag}.response.json"
        if req.exists() and json.loads(req.read_text())["signature"] != signature:
            raise ValueError(f"Configuration changed for {tag}; use a new run directory")
        save(req, {"signature": signature, "payload": payload})
        start = time.monotonic()
        if res.exists():
            result = json.loads(res.read_text())
        else:
            result = self.request("POST", "/v1beta/interactions", json=payload).json()
            save(res, result)
        deadline = time.monotonic() + 1800
        while result.get("status") in ("in_progress", "queued"):
            if time.monotonic() > deadline:
                raise RuntimeError(f"{tag} still running; rerun to resume saved interaction")
            print(f"[gemini] {tag}: {result['status']}", flush=True)
            time.sleep(10)
            result = self.request("GET", "/v1beta/interactions/" + result["id"]).json()
            save(res, result)
        if result.get("status") != "completed":
            raise RuntimeError(f"{tag} ended with {result.get('status')}; see saved response")
        self.log({"kind": "model_result", "tag": tag, "model": self.model,
                  "interaction_id": result.get("id"), "elapsed_s": time.monotonic() - start,
                  "usage": result.get("usage"), "processing": processing,
                  "processing_calls": sum(s.get("type") == "processing_call" for s in result.get("steps", []))})
        text = result.get("output_text")
        if not text:
            parts = result.get("outputs", [])
            if not parts:
                parts = [c for s in result.get("steps", []) if s.get("type") == "model_output"
                         for c in s.get("content", [])]
            text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        return json.loads(text)
