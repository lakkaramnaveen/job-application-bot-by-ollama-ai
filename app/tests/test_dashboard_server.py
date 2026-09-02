import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from job_bot.dashboard.server import make_handler
from job_bot.tracker.db import Tracker


@pytest.fixture
def live_server(tmp_path):
    db_path = tmp_path / "db.sqlite3"
    tracker = Tracker(db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme Corp", "https://example.com/job1", match_score=80)
    tracker.mark_applied("job1")

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
    assert len(data) == 1
    assert data[0]["job_id"] == "job1"
    assert data[0]["status"] == "applied"


def test_unknown_path_returns_404(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{live_server}/does-not-exist")
    assert exc_info.value.code == 404
