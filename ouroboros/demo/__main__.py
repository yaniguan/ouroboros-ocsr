"""Run the notebook's local UI: python -m ouroboros.demo [--port 8765]."""

from __future__ import annotations

import argparse
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import torch

from ouroboros.demo.rotation import DEFAULT_SMILES, RotationLab


class LabServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int, lab: RotationLab):
        self.lab = lab
        self.prefix = f"/{secrets.token_urlsafe(24)}/"
        self.lock = threading.Lock()
        super().__init__(("127.0.0.1", port), LabHandler)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}{self.prefix}"


class LabHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass  # the URL contains a local session token; do not log requests

    def _send(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: dict):
        self._send(status, json.dumps(value, allow_nan=False).encode(), "application/json")

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == self.server.prefix:
            self._send(200, Path(__file__).with_name("index.html").read_bytes(), "text/html")
        elif path == self.server.prefix + "health":
            self._json(200, {"ready": True})
        else:
            self._json(404, {"error": "Use the session URL printed by the demo launcher."})

    def do_POST(self):
        path = urlsplit(self.path).path
        routes = {self.server.prefix + "evaluate", self.server.prefix + "model"}
        if path not in routes:
            self._json(404, {"error": "Unknown session or endpoint."})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "Expected application/json."})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 16384:
                raise ValueError("Request must contain 1–16384 bytes of JSON.")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object.")
            smiles = data.get("smiles", DEFAULT_SMILES)
            if not isinstance(smiles, str) or len(smiles) > 4000:
                raise ValueError("Reference SMILES must be text, at most 4000 characters.")
            with self.server.lock:
                lab = self.server.lab
                if path.endswith("/model"):
                    if data.get("checkpoint"):
                        lab = RotationLab.from_checkpoint(
                            data["checkpoint"], data.get("device", "cpu")
                        )
                    else:
                        lab = RotationLab.random(int(data.get("order", 4)))
                result = lab.evaluate(smiles, data.get("angle", 0))
                # A failed load/evaluation leaves the previously working session intact.
                self.server.lab = lab
            self._json(200, result)
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765, help="0 chooses an available port")
    ap.add_argument("--order", type=int, choices=(4, 8, 16), default=4)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--ready-file", type=Path, help="JSON connection info for the notebook")
    args = ap.parse_args()
    torch.set_num_threads(max(1, args.threads))
    lab = (
        RotationLab.from_checkpoint(args.checkpoint, args.device)
        if args.checkpoint
        else (RotationLab.random(args.order))
    )
    with LabServer(args.port, lab) as server:
        info = {"url": server.url, "port": server.server_port, "path": server.prefix}
        if args.ready_file:
            pending = args.ready_file.with_suffix(".tmp")
            pending.write_text(json.dumps(info))
            pending.replace(args.ready_file)
        print(f"Open the rotation lab: {server.url}", flush=True)
        print("Press Ctrl+C to stop. No dataset or trained weights needed for the default demo.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
