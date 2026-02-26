#!/usr/bin/env python3
"""
Horus local dev server.
Serves static files + proxies /proxy/anthropic/* → api.anthropic.com
so the browser can call the Anthropic Admin API without CORS issues.
"""
import json
import urllib.request
import urllib.error
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


PROXY_PREFIX = "/proxy/anthropic"
ANT_BASE     = "https://api.anthropic.com"


class HorusHandler(SimpleHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path.startswith(PROXY_PREFIX):
            self._proxy("GET")
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.startswith(PROXY_PREFIX):
            self._proxy("POST")
        elif self.path == "/api/save-metrics":
            self._save_metrics()
        else:
            self.send_response(404)
            self.end_headers()

    def _save_metrics(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            data   = json.loads(body)
            out    = Path(__file__).parent / "data" / "team-metrics.json"
            out.parent.mkdir(exist_ok=True)
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.rename(out)
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        except Exception as e:
            self.send_response(500)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _proxy(self, method):
        ant_path = self.path[len(PROXY_PREFIX):]
        url      = ANT_BASE + ant_path

        api_key     = self.headers.get("x-api-key", "")
        ant_version = self.headers.get("anthropic-version", "2023-06-01")
        ct          = self.headers.get("Content-Type", "application/json")

        body = None
        if method == "POST":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length) if length else None

        req = urllib.request.Request(
            url, data=body, method=method,
            headers={
                "x-api-key":          api_key,
                "anthropic-version":  ant_version,
                "Content-Type":       ct,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data   = resp.read()
                status = resp.status
        except urllib.error.HTTPError as e:
            data   = e.read()
            status = e.code
        except Exception as e:
            data   = json.dumps({"error": str(e)}).encode()
            status = 500

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
        # Only log proxy and API calls, suppress noisy static-file logs
        if PROXY_PREFIX in self.path or "/api/" in self.path:
            super().log_message(fmt, *args)


if __name__ == "__main__":
    port   = 8080
    server = HTTPServer(("0.0.0.0", port), HorusHandler)
    print(f"Horus running at http://127.0.0.1:{port}")
    print("Static files + Anthropic proxy active (/proxy/anthropic/*)")
    server.serve_forever()
