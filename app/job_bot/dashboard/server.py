"""One-page tracker dashboard: a local, no-auth HTTP server bound to
localhost only (never 0.0.0.0) since it serves your application data with
no access control. See render.py for the HTML/escaping logic this wraps.
"""

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from job_bot.dashboard.render import render_page_html, render_rows_html
from job_bot.tracker.db import Tracker

DASHBOARD_HOST = "127.0.0.1"


def make_handler(db_path: Path) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - required name for BaseHTTPRequestHandler
            tracker = Tracker(db_path)
            if self.path == "/":
                jobs = tracker.list_jobs()
                self._send(200, "text/html; charset=utf-8", render_page_html(jobs).encode("utf-8"))
            elif self.path == "/api/rows":
                jobs = tracker.list_jobs()
                self._send(200, "text/html; charset=utf-8", render_rows_html(jobs).encode("utf-8"))
            elif self.path == "/api/jobs":
                jobs = tracker.list_jobs()
                self._send(200, "application/json", json.dumps(jobs, default=str).encode("utf-8"))
            else:
                self._send(404, "text/plain; charset=utf-8", b"Not found")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass  # quiet by default; the CLI prints the one line that matters

    return DashboardHandler


def run_dashboard(db_path: Path, port: int = 8765, open_browser: bool = True) -> None:
    handler = make_handler(db_path)
    server = ThreadingHTTPServer((DASHBOARD_HOST, port), handler)
    url = f"http://{DASHBOARD_HOST}:{server.server_port}/"
    print(f"Dashboard running at {url} (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
