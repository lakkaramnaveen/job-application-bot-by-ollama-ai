import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

from job_bot.dashboard.server import make_handler, run_dashboard
from job_bot.tracker.db import Tracker


@pytest.fixture
def live_server(tmp_path):
    db_path = tmp_path / "db.sqlite3"
    tracker = Tracker(db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme Corp", "https://example.com/job1", match_score=80)
    tracker.mark_applied("job1")
    # A job_id needing percent-encoding, to exercise the frontend's
    # encodeURIComponent(job_id) round-tripping through the server.
    tracker.upsert_job("job 2", "Frontend Engineer", "Acme Corp", "https://example.com/job2")

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(db_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_index_page_serves_html_with_job_data(live_server):
    with urllib.request.urlopen(f"{live_server}/") as resp:
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/html")
        body = resp.read().decode("utf-8")
    assert "Backend Engineer" in body
    assert "Acme Corp" in body


def test_api_rows_returns_table_rows_only(live_server):
    with urllib.request.urlopen(f"{live_server}/api/rows") as resp:
        body = resp.read().decode("utf-8")
    assert "<tr>" in body
    assert "<!doctype html>" not in body.lower()


def test_api_jobs_returns_json(live_server):
    with urllib.request.urlopen(f"{live_server}/api/jobs") as resp:
        assert resp.headers["Content-Type"] == "application/json"
        data = json.loads(resp.read().decode("utf-8"))
    by_id = {job["job_id"]: job for job in data}
    assert by_id["job1"]["status"] == "applied"


def test_unknown_path_returns_404(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{live_server}/does-not-exist")
    assert exc_info.value.code == 404


def _post_json(url: str, payload: dict, headers: dict | None = None, *, same_origin: bool = True):
    """POSTs `payload` as JSON. Real browsers attach an Origin header to
    every POST, same-origin included, so `same_origin=True` (the default)
    mimics that by setting Origin to the request's own origin - matching
    what the dashboard's own JS would send. Pass same_origin=False (or an
    explicit Origin in `headers`) to exercise a request without that.
    """
    body = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    if same_origin:
        split = urlsplit(url)
        request_headers["Origin"] = f"{split.scheme}://{split.netloc}"
    request_headers.update(headers or {})
    req = urllib.request.Request(url, data=body, method="POST", headers=request_headers)
    return urllib.request.urlopen(req)


def test_api_rows_supports_status_filter(live_server):
    with urllib.request.urlopen(f"{live_server}/api/rows?status=applied") as resp:
        body = resp.read().decode("utf-8")
    assert "Backend Engineer" in body


def test_api_rows_supports_search(live_server):
    with urllib.request.urlopen(f"{live_server}/api/rows?q=nonexistent") as resp:
        body = resp.read().decode("utf-8")
    assert "No jobs tracked yet" in body


def test_api_rows_reports_total_via_header(live_server):
    with urllib.request.urlopen(f"{live_server}/api/rows") as resp:
        assert resp.headers["X-Total-Jobs"] == "2"


def test_api_stats_returns_pill_fragment_reflecting_current_data(live_server):
    with urllib.request.urlopen(f"{live_server}/api/stats") as resp:
        assert resp.headers["Content-Type"].startswith("text/html")
        body = resp.read().decode("utf-8")
    assert "<!doctype html>" not in body.lower()
    assert 'All <span class="count">2</span>' in body
    assert 'data-status="applied"' in body
    assert 'data-status="seen"' in body


def test_api_stats_scopes_counts_to_search_term(live_server):
    with urllib.request.urlopen(f"{live_server}/api/stats?q=Backend") as resp:
        body = resp.read().decode("utf-8")
    assert 'All <span class="count">1</span>' in body
    assert 'data-status="applied"' in body
    assert 'data-status="seen"' not in body


def test_index_page_includes_stats_bar_reflecting_data(live_server):
    with urllib.request.urlopen(f"{live_server}/") as resp:
        body = resp.read().decode("utf-8")
    assert 'id="stats"' in body
    assert 'data-status="applied"' in body


def test_api_rows_rejects_invalid_sort_column(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{live_server}/api/rows?sort=job_id%3B+DROP+TABLE+jobs")
    assert exc_info.value.code == 400


def test_api_rows_falls_back_to_page_1_for_a_non_numeric_page(live_server):
    """A hand-edited or stale ?page= query param shouldn't 500 - it's not
    dangerous input like the sort column (which reaches raw SQL), just a
    display parameter, so it degrades to page 1 rather than erroring.
    """
    with urllib.request.urlopen(f"{live_server}/api/rows?page=not-a-number") as resp:
        assert resp.status == 200
        body = resp.read().decode("utf-8")
    assert "Backend Engineer" in body


def test_index_page_falls_back_to_default_sort_for_an_invalid_sort_column(live_server):
    """Unlike /api/rows (used by the frontend's own JS, which always sends
    a valid sort value from its own <select>), the index page can be
    reached with an arbitrary/stale query string typed or bookmarked by
    hand - it should render normally on the default sort rather than
    surfacing a raw 400 to someone who just mistyped a URL.
    """
    with urllib.request.urlopen(f"{live_server}/?sort=job_id%3B+DROP+TABLE+jobs") as resp:
        assert resp.status == 200
        body = resp.read().decode("utf-8")
    assert "Backend Engineer" in body


def test_post_to_unknown_path_returns_404(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(f"{live_server}/api/not-a-real-endpoint", {"status": "offer"})
    assert exc_info.value.code == 404


def test_post_status_rejects_a_non_string_status_value(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(f"{live_server}/api/jobs/job1/status", {"status": 123})
    assert exc_info.value.code == 400


def test_post_status_rejects_malformed_json_body(live_server):
    req = urllib.request.Request(
        f"{live_server}/api/jobs/job1/status",
        data=b"{not valid json",
        method="POST",
        headers={"Content-Type": "application/json", "Origin": live_server},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400


def test_api_jobs_qa_returns_fragment(live_server):
    with urllib.request.urlopen(f"{live_server}/api/jobs/job1/qa") as resp:
        assert resp.status == 200
        body = resp.read().decode("utf-8")
    assert "No answered questions" in body


def test_api_jobs_qa_returns_404_for_unknown_job(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{live_server}/api/jobs/does-not-exist/qa")
    assert exc_info.value.code == 404


def test_post_status_updates_job(live_server):
    resp = _post_json(f"{live_server}/api/jobs/job1/status", {"status": "interviewing"})
    assert resp.status == 200
    data = json.loads(resp.read().decode("utf-8"))
    assert data == {"ok": True, "job_id": "job1", "status": "interviewing"}

    with urllib.request.urlopen(f"{live_server}/api/jobs") as verify:
        jobs = json.loads(verify.read().decode("utf-8"))
    by_id = {job["job_id"]: job for job in jobs}
    assert by_id["job1"]["status"] == "interviewing"


def test_post_status_rejects_unknown_status(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(f"{live_server}/api/jobs/job1/status", {"status": "ghosted"})
    assert exc_info.value.code == 400


def test_post_status_rejects_unknown_job(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(f"{live_server}/api/jobs/does-not-exist/status", {"status": "offer"})
    assert exc_info.value.code == 404


def test_post_status_rejects_non_json_content_type(live_server):
    req = urllib.request.Request(
        f"{live_server}/api/jobs/job1/status",
        data=b"status=offer",
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": live_server,
        },
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400


def test_post_status_rejects_cross_origin_request(live_server):
    """No auth guards this server, so a cross-origin Origin header (which a
    browser attaches automatically and which a page can't spoof) is the only
    signal distinguishing the dashboard's own page from another site trying
    a drive-by localhost POST.
    """
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(
            f"{live_server}/api/jobs/job1/status",
            {"status": "offer"},
            headers={"Origin": "https://evil.example.com"},
        )
    assert exc_info.value.code == 403


def test_post_status_rejects_missing_origin(live_server):
    """A real browser attaches Origin to every POST, same-origin included -
    a request with none isn't a browser honoring same-origin semantics at
    all, so it's rejected rather than trusted by default.
    """
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(f"{live_server}/api/jobs/job1/status", {"status": "offer"}, same_origin=False)
    assert exc_info.value.code == 403


def test_post_status_allows_same_origin_request(live_server):
    resp = _post_json(f"{live_server}/api/jobs/job1/status", {"status": "offer"})
    assert resp.status == 200


def test_post_status_rejects_oversized_body(live_server):
    huge_payload = {"status": "offer", "padding": "x" * 10_000}
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(f"{live_server}/api/jobs/job1/status", huge_payload)
    assert exc_info.value.code == 400


def test_post_status_rejects_a_non_numeric_content_length(live_server):
    """Content-Length is client-supplied and needn't be a number. Parsing it
    unguarded raised ValueError out of do_POST, which dropped the connection
    with no HTTP response at all (and a traceback on the server console)
    instead of answering 400. Raw socket, since http.client refuses to send
    a malformed Content-Length in the first place.
    """
    parts = urlsplit(live_server)
    sock = socket.create_connection((parts.hostname, parts.port), timeout=5)
    try:
        sock.sendall(
            f"POST /api/jobs/job1/status HTTP/1.1\r\n"
            f"Host: {parts.hostname}:{parts.port}\r\n"
            f"Origin: {live_server}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: not-a-number\r\n\r\n".encode()
        )
        # Read to EOF rather than a single recv() - headers and body can
        # arrive in separate TCP segments. The server closes the connection
        # after responding (HTTP/1.0), so this terminates on its own.
        chunks = []
        while chunk := sock.recv(4096):
            chunks.append(chunk)
        response = b"".join(chunks).decode("utf-8", errors="replace")
    finally:
        sock.close()

    assert response.startswith("HTTP/1.0 400") or response.startswith("HTTP/1.1 400")
    assert "Invalid Content-Length" in response


def test_post_status_decodes_percent_encoded_job_id(live_server):
    """The frontend sends job_id via encodeURIComponent (see render.py) -
    the server must decode it back before matching against the stored
    value, or a job_id with a space never resolves.
    """
    resp = _post_json(f"{live_server}/api/jobs/job%202/status", {"status": "applied"})
    assert resp.status == 200
    data = json.loads(resp.read().decode("utf-8"))
    assert data == {"ok": True, "job_id": "job 2", "status": "applied"}


def test_run_dashboard_opens_browser_and_shuts_down_cleanly(tmp_path, monkeypatch):
    """run_dashboard() is the CLI's actual entry point (`job-bot dashboard`)
    - the rest of this file exercises the request handlers directly via
    make_handler(), bypassing it entirely. Ctrl+C (KeyboardInterrupt out of
    serve_forever()) is the normal way this function returns, so
    ThreadingHTTPServer.serve_forever is patched to raise it immediately
    instead of actually blocking, letting this run synchronously.
    """
    opened_urls = []
    monkeypatch.setattr(
        "job_bot.dashboard.server.webbrowser.open", lambda url: opened_urls.append(url)
    )

    def fake_serve_forever(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(ThreadingHTTPServer, "serve_forever", fake_serve_forever)

    run_dashboard(tmp_path / "db.sqlite3", port=0, open_browser=True)

    assert len(opened_urls) == 1
    assert opened_urls[0].startswith("http://127.0.0.1:")


def test_qa_endpoint_decodes_percent_encoded_job_id(live_server):
    with urllib.request.urlopen(f"{live_server}/api/jobs/job%202/qa") as resp:
        assert resp.status == 200
