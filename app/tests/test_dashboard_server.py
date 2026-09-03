import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

from job_bot.dashboard.server import make_handler
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


def test_api_rows_rejects_invalid_sort_column(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{live_server}/api/rows?sort=job_id%3B+DROP+TABLE+jobs")
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


def test_post_status_decodes_percent_encoded_job_id(live_server):
    """The frontend sends job_id via encodeURIComponent (see render.py) -
    the server must decode it back before matching against the stored
    value, or a job_id with a space never resolves.
    """
    resp = _post_json(f"{live_server}/api/jobs/job%202/status", {"status": "applied"})
    assert resp.status == 200
    data = json.loads(resp.read().decode("utf-8"))
    assert data == {"ok": True, "job_id": "job 2", "status": "applied"}


def test_qa_endpoint_decodes_percent_encoded_job_id(live_server):
    with urllib.request.urlopen(f"{live_server}/api/jobs/job%202/qa") as resp:
        assert resp.status == 200
