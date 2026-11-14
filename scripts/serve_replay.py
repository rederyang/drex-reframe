"""Serve the bundled offline replay. Python standard library only; no API key."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(302)
            self.send_header("Location", "/replay/")
            self.end_headers()
            return
        return super().do_GET()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8504)
    args = parser.parse_args()
    if not (ROOT / "demo" / "replay" / "manifest.json").is_file():
        parser.error("Bundled replay assets are missing; clone the portable demo branch.")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(ROOT / "demo")))
    print(f"Offline replay: http://127.0.0.1:{server.server_port}/replay/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
