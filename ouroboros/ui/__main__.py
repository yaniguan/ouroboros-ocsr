"""Run the Ouroboros local web app.

Example:
    python -m ouroboros.ui --port 8787
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import secrets
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import torch

from ouroboros.ui.backend import WorkbenchEngine

MAX_JSON_BYTES = 12 * 1024 * 1024


def _clean_numbers(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean_numbers(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_numbers(v) for v in value]
    return value


class WorkbenchServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = sys.platform != "win32"

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

    def server_bind(self) -> None:
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

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
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(_clean_numbers(payload), allow_nan=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _route(self) -> str | None:
        raw = urlsplit(self.path).path
        prefix = self.server.prefix  # type: ignore[attr-defined]
        if prefix == "/":
            return raw.lstrip("/")
        if raw == prefix.rstrip("/"):
            return ""
        if raw.startswith(prefix):
            return raw[len(prefix) :]
        return None

    def do_GET(self) -> None:
        route = self._route()
        if route is None:
            self._send_json(404, {"error": "Wrong session URL."})
            return
        if route in ("", "index.html"):
            html = Path(__file__).with_name("index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
            return
        if route == "api/health":
            self._send_json(200, {"ready": True})
            return
        if route == "api/status":
            eng: WorkbenchEngine = self.server.engine  # type: ignore[attr-defined]
            with eng.lock:
                self._send_json(200, eng.status())
            return
        self._send_json(404, {"error": f"Unknown GET route: {route}"})

    def do_POST(self) -> None:
        route = self._route()
        if route is None:
            self._send_json(404, {"error": "Wrong session URL."})
            return
        if self.headers.get_content_type() != "application/json":
            self._send_json(415, {"error": "Please send JSON (application/json)."})
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
        fn = routes.get(route)
        if fn is None:
            self._send_json(404, {"error": f"Unknown POST route: {route}"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_JSON_BYTES:
                raise ValueError("Request body is empty or too large.")
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object.")
            with eng.lock:
                self._send_json(200, fn(data))
        except (ValueError, TypeError, KeyError, FileNotFoundError, OSError, RuntimeError) as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8787, help="Port number (0 picks a free port)")
    parser.add_argument("--host", default="127.0.0.1", help="Address to listen on")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--token", action="store_true", help="Add a random token to the URL")
    parser.add_argument("--ready-file", type=Path, help="Write server URL info to this JSON file")
    args = parser.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    engine = WorkbenchEngine(device=args.device)
    try:
        server = WorkbenchServer(args.port, engine, host=args.host, use_token=args.token)
    except OSError as exc:
        if exc.errno in (errno.EADDRINUSE, errno.EACCES):
            parser.error(f"Port {args.port} is busy. Try another --port or use --port 0.")
        raise

    with server:
        if args.ready_file:
            info = {"url": server.url, "port": server.server_port, "prefix": server.prefix}
            tmp = args.ready_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(info), encoding="utf-8")
            tmp.replace(args.ready_file)
        print(f"Ouroboros UI running at: {server.url}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
