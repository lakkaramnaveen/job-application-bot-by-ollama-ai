"""One-page tracker dashboard: a local, no-auth HTTP server bound to
localhost only (never 0.0.0.0) since it serves your application data with
no access control. See render.py for the HTML/escaping logic this wraps.

The dashboard also accepts one state-changing request (POST status update).
Because the server has no auth, any page open in the same browser could in
principle try to trigger it (a "drive-by localhost" request) - _is_same_origin
plus the browser's own CORS preflight (triggered by the required JSON
Content-Type) are what stand in for auth here. See _is_same_origin and
cmd_status_update below.
"""

import json
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlparse

from job_bot.dashboard.render import PAGE_SIZE, render_page_html, render_qa_html, render_rows_html
from job_bot.tracker.db import InvalidSort, InvalidStatus, Tracker

DASHBOARD_HOST = "127.0.0.1"

# A POST body larger than this is rejected outright - the only legitimate
# body is a tiny {"status": "..."} JSON object, so anything bigger is either
# a bug or abuse.
MAX_BODY_BYTES = 4096

_DEFAULT_SORT = "first_seen_at"
_DEFAULT_DIRECTION = "desc"


def _parse_list_params(query: dict[str, list[str]]) -> dict:
    """Shared query-string parsing for `/` and `/api/rows` - keeps the two
    handlers' filter/sort/page behavior identical.
    """
    status = (query.get("status", [""])[0] or "").strip()
    search = (query.get("q", [""])[0] or "").strip()
    sort = (query.get("sort", [_DEFAULT_SORT])[0] or _DEFAULT_SORT).strip()
    direction = (query.get("dir", [_DEFAULT_DIRECTION])[0] or _DEFAULT_DIRECTION).strip()
    try:
        page = max(1, int(query.get("page", ["1"])[0]))
    except ValueError:
        page = 1
    return {"status": status or None, "search": search or None, "sort": sort, "direction": direction, "page": page}


def make_handler(db_path: Path) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def _send(self, status: HTTPStatus | int, content_type: str, body: bytes, headers: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, status: HTTPStatus | int, text: str) -> None:
            self._send(status, "text/plain; charset=utf-8", text.encode("utf-8"))

        def _send_json(self, status: HTTPStatus | int, payload: dict) -> None:
            self._send(status, "application/json", json.dumps(payload).encode("utf-8"))

        def _is_same_origin(self) -> bool:
            """Reject a POST whose Origin header (sent by browsers on
            cross-origin fetches) doesn't match this dashboard's own
            host:port. A request with no Origin header (same-page navigation,
            curl, tests) is allowed through - Origin is specifically a
            cross-origin signal, so its absence isn't evidence of an attack.
            """
            origin = self.headers.get("Origin")
            if origin is None:
                return True
            port = cast(ThreadingHTTPServer, self.server).server_port
            expected = {f"http://{DASHBOARD_HOST}:{port}", f"http://localhost:{port}"}
            return origin in expected

        def _job_id_from_path(self, prefix: str, suffix: str) -> str | None:
            """Extract `{job_id}` from a path shaped like
            f"{prefix}{{job_id}}{suffix}", or None if it doesn't match.
            job_id is only ever used as a parameterized SQL value, so no
            further sanitization is needed here beyond "non-empty".
            """
            path = urlparse(self.path).path
            if not (path.startswith(prefix) and path.endswith(suffix)):
                return None
            job_id = path[len(prefix) : len(path) - len(suffix)]
            return job_id or None

        def do_GET(self) -> None:  # noqa: N802 - required name for BaseHTTPRequestHandler
            parsed = urlparse(self.path)
            tracker = Tracker(db_path)

            if parsed.path == "/":
                self._handle_index(tracker, parse_qs(parsed.query))
            elif parsed.path == "/api/rows":
                self._handle_rows(tracker, parse_qs(parsed.query))
            elif parsed.path == "/api/jobs":
                jobs = tracker.list_jobs()
                self._send(200, "application/json", json.dumps(jobs, default=str).encode("utf-8"))
            elif (job_id := self._job_id_from_path("/api/jobs/", "/qa")) is not None:
                self._handle_qa(tracker, job_id)
            else:
                self._send_text(404, "Not found")

        def do_POST(self) -> None:  # noqa: N802 - required name for BaseHTTPRequestHandler
            if (job_id := self._job_id_from_path("/api/jobs/", "/status")) is not None:
                self._handle_status_update(Tracker(db_path), job_id)
            else:
                self._send_text(404, "Not found")

        def _handle_index(self, tracker: Tracker, query: dict[str, list[str]]) -> None:
            params = _parse_list_params(query)
            try:
                total = tracker.count_jobs(status=params["status"], search=params["search"])
                jobs = tracker.list_jobs(
                    status=params["status"],
                    search=params["search"],
                    sort=params["sort"],
                    direction=params["direction"],
                    limit=PAGE_SIZE,
                    offset=(params["page"] - 1) * PAGE_SIZE,
                )
            except InvalidSort:
                params["sort"], params["direction"] = _DEFAULT_SORT, _DEFAULT_DIRECTION
                total = tracker.count_jobs(status=params["status"], search=params["search"])
                jobs = tracker.list_jobs(
                    status=params["status"], search=params["search"], limit=PAGE_SIZE, offset=0
                )
                params["page"] = 1
            body = render_page_html(
                jobs,
                total=total,
                page=params["page"],
                page_size=PAGE_SIZE,
                status=params["status"] or "",
                search=params["search"] or "",
                sort=params["sort"],
                direction=params["direction"],
            ).encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)

        def _handle_rows(self, tracker: Tracker, query: dict[str, list[str]]) -> None:
            params = _parse_list_params(query)
            try:
                total = tracker.count_jobs(status=params["status"], search=params["search"])
                jobs = tracker.list_jobs(
                    status=params["status"],
                    search=params["search"],
                    sort=params["sort"],
                    direction=params["direction"],
                    limit=PAGE_SIZE,
                    offset=(params["page"] - 1) * PAGE_SIZE,
                )
            except InvalidSort as e:
                self._send_text(400, str(e))
                return
            body = render_rows_html(jobs).encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body, headers={"X-Total-Jobs": str(total)})

        def _handle_qa(self, tracker: Tracker, job_id: str) -> None:
            if tracker.get_job(job_id) is None:
                self._send_text(404, "Job not found")
                return
            body = render_qa_html(tracker.list_qa(job_id)).encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)

        def _handle_status_update(self, tracker: Tracker, job_id: str) -> None:
            if not self._is_same_origin():
                self._send_text(403, "Cross-origin request rejected")
                return
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                self._send_text(400, "Content-Type must be application/json")
                return

            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send_text(400, "Request body missing or too large")
                return
            raw_body = self.rfile.read(length)

            try:
                payload = json.loads(raw_body)
                new_status = payload["status"]
                if not isinstance(new_status, str):
                    raise ValueError("status must be a string")
            except (json.JSONDecodeError, KeyError, ValueError):
                self._send_text(400, "Body must be JSON: {\"status\": \"<status>\"}")
                return

            try:
                tracker.update_status(job_id, new_status)
            except InvalidStatus as e:
                self._send_text(400, str(e))
                return
            except ValueError as e:
                self._send_text(404, str(e))
                return
            self._send_json(200, {"ok": True, "job_id": job_id, "status": new_status})

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
