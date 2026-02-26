#!/usr/bin/env python3
"""
Horus local dev server.
  - Serves static files
  - Proxies /proxy/anthropic/* → api.anthropic.com  (avoids browser CORS)
  - POST /api/save-metrics      writes data/team-metrics.json
  - POST /api/run-fetch         spawns fetch_data.py in the background
  - GET  /api/fetch-status      returns {running, last_updated, size, log_tail}
"""
import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT         = Path(__file__).parent
PROXY_PREFIX = "/proxy/anthropic"
ANT_BASE     = "https://api.anthropic.com"
METRICS_FILE = ROOT / "data" / "team-metrics.json"
FETCH_LOG    = ROOT / "data" / "fetch.log"

_fetch_process: subprocess.Popen | None = None
_fetch_lock = threading.Lock()


def _is_running() -> bool:
    return _fetch_process is not None and _fetch_process.poll() is None


class HorusHandler(SimpleHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path.startswith(PROXY_PREFIX):
            self._proxy("GET")
        elif self.path.startswith("/api/fetch-status"):
            self._fetch_status()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.startswith(PROXY_PREFIX):
            self._proxy("POST")
        elif self.path == "/api/save-metrics":
            self._save_metrics()
        elif self.path == "/api/run-fetch":
            self._run_fetch()
        else:
            self.send_response(404)
            self.end_headers()

    # ── API handlers ──────────────────────────────────────────────────────────

    def _fetch_status(self):
        mtime = size = None
        if METRICS_FILE.exists():
            st    = METRICS_FILE.stat()
            mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
            size  = st.st_size
        # Return the last few lines of the log so the browser can show progress
        log_tail = ""
        if FETCH_LOG.exists():
            try:
                lines = FETCH_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
                log_tail = "\n".join(lines[-20:])
            except Exception:
                pass
        self._json_response(200, {
            "running":      _is_running(),
            "last_updated": mtime,
            "size":         size,
            "log_tail":     log_tail,
        })

    def _run_fetch(self):
        global _fetch_process
        with _fetch_lock:
            if _is_running():
                self._json_response(200, {"started": False, "message": "Already running"})
                return
            try:
                METRICS_FILE.parent.mkdir(exist_ok=True)
                log_fh = FETCH_LOG.open("w", encoding="utf-8")
                _fetch_process = subprocess.Popen(
                    [sys.executable, str(ROOT / "fetch_data.py"), "--days", "90"],
                    cwd=str(ROOT),
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )
                self._json_response(200, {"started": True})
            except Exception as e:
                self._json_response(500, {"error": str(e)})

    def _save_metrics(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            data   = json.loads(body)
            METRICS_FILE.parent.mkdir(exist_ok=True)
            tmp = METRICS_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.rename(METRICS_FILE)
            self._json_response(200, {"ok": True})
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _json_response(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ── Anthropic proxy ───────────────────────────────────────────────────────

    def _proxy(self, method):
        ant_path    = self.path[len(PROXY_PREFIX):]
        url         = ANT_BASE + ant_path
        api_key     = self.headers.get("x-api-key", "")
        ant_version = self.headers.get("anthropic-version", "2023-06-01")
        ct          = self.headers.get("Content-Type", "application/json")
        body = None
        if method == "POST":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length) if length else None
        req = urllib.request.Request(
            url, data=body, method=method,
            headers={"x-api-key": api_key, "anthropic-version": ant_version, "Content-Type": ct},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data, status = resp.read(), resp.status
        except urllib.error.HTTPError as e:
            data, status = e.read(), e.code
        except Exception as e:
            data, status = json.dumps({"error": str(e)}).encode(), 500
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "x-api-key, anthropic-version, Content-Type")

    def log_message(self, fmt, *args):
        if PROXY_PREFIX in self.path or "/api/" in self.path:
            super().log_message(fmt, *args)


if __name__ == "__main__":
    port   = 8080
    server = HTTPServer(("0.0.0.0", port), HorusHandler)
    print(f"Horus running at http://127.0.0.1:{port}")
    print("  /proxy/anthropic/*  → Anthropic API proxy")
    print("  POST /api/run-fetch → spawns fetch_data.py (background)")
    print("  GET  /api/fetch-status → poll fetch progress")
    server.serve_forever()
