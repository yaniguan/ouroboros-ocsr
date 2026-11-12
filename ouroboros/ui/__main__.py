"""Launch the Ouroboros Interactive Workbench & Physics Compliance UI.

Usage:
    python -m ouroboros.ui --port 8787
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import torch

from ouroboros.ui.backend import WorkbenchEngine

MAX_JSON_BYTES = 12 * 1024 * 1024


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return obj


class WorkbenchServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        port: int,
        engine: WorkbenchEngine,
        *,
        host: str = "127.0.0.1",
        use_token: bool = False,
    ) -> None:
        self.engine = engine
        self.prefix = f"/{secrets.token_urlsafe(18)}/" if use_token else "/"
        super().__init__((host, port), WorkbenchHandler)

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}{self.prefix}"


class WorkbenchHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args: Any) -> None:
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        clean = _sanitize_json(payload)
        body = json.dumps(clean, allow_nan=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _subpath(self) -> str | None:
        raw_path = urlsplit(self.path).path
        pref = self.server.prefix  # type: ignore[attr-defined]
        if pref == "/":
            return raw_path.lstrip("/")
        if raw_path == pref.rstrip("/"):
            return ""
        if raw_path.startswith(pref):
            return raw_path[len(pref) :]
        return None

    def do_GET(self) -> None:
        sub = self._subpath()
        if sub is None:
            self._json(404, {"error": "Invalid session path."})
            return
        if sub in ("", "index.html"):
            html_bytes = Path(__file__).with_name("index.html").read_bytes()
            self._send(200, html_bytes, "text/html; charset=utf-8")
            return
        if sub == "api/health":
            self._json(200, {"ready": True})
            return
        if sub == "api/status":
            eng: WorkbenchEngine = self.server.engine  # type: ignore[attr-defined]
            with eng.lock:
                self._json(200, eng.status())
            return
        self._json(404, {"error": f"Unknown GET endpoint: {sub}"})

    def do_POST(self) -> None:
        sub = self._subpath()
        if sub is None:
            self._json(404, {"error": "Invalid session path."})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "Expected Content-Type: application/json."})
            return

        eng: WorkbenchEngine = self.server.engine  # type: ignore[attr-defined]
        routes = {
            "api/ocsr": eng.run_ocsr,
            "api/equivariance2d": eng.run_2d_equivariance,
            "api/compliance3d": eng.run_3d_compliance,
            "api/propagation": eng.run_error_propagation,
            "api/forcefield": eng.run_forcefield_builder,
            "api/evaluate": eng.run_batch_evaluation,
            "api/benchmark": lambda d: eng.load_repo_benchmark(str(d.get("path", ""))),
        }
        handler_fn = routes.get(sub)
        if handler_fn is None:
            self._json(404, {"error": f"Unknown POST endpoint: {sub}"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_JSON_BYTES:
                raise ValueError(f"JSON body must be between 1 and {MAX_JSON_BYTES} bytes.")
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Request payload must be a JSON object.")
            with eng.lock:
                result = handler_fn(data)
            self._json(200, result)
        except (ValueError, TypeError, KeyError, FileNotFoundError, OSError, RuntimeError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8787, help="Port to bind (0 chooses free port)")
    ap.add_argument("--host", default="127.0.0.1", help="Loopback address (default: 127.0.0.1)")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--token", action="store_true", help="Protect URL with random session token")
    ap.add_argument("--ready-file", type=Path, help="Optional JSON file written on startup")
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    engine = WorkbenchEngine(device=args.device)
    try:
        server = WorkbenchServer(args.port, engine, host=args.host, use_token=args.token)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            ap.error(f"Port {args.port} is already in use. Choose another --port or pass --port 0.")
        raise

    with server:
        info = {"url": server.url, "port": server.server_port, "prefix": server.prefix}
        if args.ready_file:
            tmp = args.ready_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(info), encoding="utf-8")
            tmp.replace(args.ready_file)
        print(f"Ouroboros Physics & Equivariance Workbench running at: {server.url}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
