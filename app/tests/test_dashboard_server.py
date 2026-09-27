import json
import socket
import sqlite3
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

from job_bot.dashboard.server import DashboardPortInUse, make_handler, run_dashboard
from job_bot.safety.answer_gaps import AnswerGapStore
from job_bot.safety.audit_log import AuditLogger
from job_bot.safety.blacklist import CompanyBlacklist
from job_bot.tracker.db import Tracker


@pytest.fixture
def live_server(tmp_path):
    db_path = tmp_path / "db.sqlite3"
    # Read by test_blacklist_endpoint tests via tmp_path directly (the same
    # tmp_path instance this fixture and the test function both receive) -
    # not exposed on live_server's own return value, which stays a plain
    # URL string so the many existing f"{live_server}/..." call sites in
    # this file don't all need to change shape for one feature.
    blacklist_path = tmp_path / "blacklist.json"
    audit_log_path = tmp_path / "audit.log"
    failed_applications_log_path = tmp_path / "failed_applications.log"
    answer_gaps_path = tmp_path / "answer_gaps.json"
    tracker = Tracker(db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme Corp", "https://example.com/job1", match_score=80)
    tracker.mark_applied("job1")
    # A job_id needing percent-encoding, to exercise the frontend's
    # encodeURIComponent(job_id) round-tripping through the server.
    tracker.upsert_job("job 2", "Frontend Engineer", "Acme Corp", "https://example.com/job2")

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(db_path, blacklist_path, audit_log_path, failed_applications_log_path, answer_gaps_path),
    )
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


def test_export_csv_downloads_every_matching_job(live_server):
    with urllib.request.urlopen(f"{live_server}/api/export.csv") as resp:
        assert resp.headers["Content-Type"].startswith("text/csv")
        assert "attachment" in resp.headers["Content-Disposition"]
        body = resp.read().decode("utf-8")
    lines = body.strip().splitlines()
    assert lines[0].split(",")[0] == "job_id"
    assert any(line.startswith("job1,") for line in lines[1:])
    assert any(line.startswith("job 2,") for line in lines[1:])


def test_export_csv_respects_the_status_filter(live_server):
    with urllib.request.urlopen(f"{live_server}/api/export.csv?status=applied") as resp:
        body = resp.read().decode("utf-8")
    assert "job1," in body
    assert "job 2," not in body


def test_export_csv_respects_the_search_box(live_server):
    with urllib.request.urlopen(f"{live_server}/api/export.csv?q=Frontend") as resp:
        body = resp.read().decode("utf-8")
    assert "job 2," in body
    assert "job1," not in body


def test_export_json_downloads_every_matching_job(live_server):
    with urllib.request.urlopen(f"{live_server}/api/export.json") as resp:
        assert resp.headers["Content-Type"] == "application/json"
        assert "attachment" in resp.headers["Content-Disposition"]
        rows = json.loads(resp.read().decode("utf-8"))
    assert {r["job_id"] for r in rows} == {"job1", "job 2"}


def test_export_json_respects_the_status_filter(live_server):
    with urllib.request.urlopen(f"{live_server}/api/export.json?status=applied") as resp:
        rows = json.loads(resp.read().decode("utf-8"))
    assert [r["job_id"] for r in rows] == ["job1"]


def test_export_json_respects_the_search_box(live_server):
    with urllib.request.urlopen(f"{live_server}/api/export.json?q=Frontend") as resp:
        rows = json.loads(resp.read().decode("utf-8"))
    assert [r["job_id"] for r in rows] == ["job 2"]


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


def _backdate_applied_at(db_path, job_id: str, when: datetime) -> None:
    """Directly rewrite applied_at, since Tracker.mark_applied() always
    stamps "now" - a test needing a stale application has to backdate it
    after the fact rather than through the public Tracker API.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE jobs SET applied_at = ? WHERE job_id = ?", (when.isoformat(), job_id))
    conn.commit()
    conn.close()


def test_api_rows_marks_a_stale_application(live_server, tmp_path):
    """End-to-end proof that make_handler's stale_after_days (default 14,
    same as Settings.stale_after_days' own default) actually reaches
    render_rows_html - job1 in the live_server fixture was mark_applied()'d
    "just now", so backdating it here is what actually exercises the
    marker rather than just proving the wiring doesn't crash.
    """
    _backdate_applied_at(tmp_path / "db.sqlite3", "job1", datetime.now(UTC) - timedelta(days=20))

    with urllib.request.urlopen(f"{live_server}/api/rows") as resp:
        body = resp.read().decode("utf-8")

    assert "⏰" in body


def test_api_rows_supports_eligibility_filter(live_server, tmp_path):
    # Same tmp_path instance the live_server fixture used internally to
    # build its own db_path (see the fixture's own comment on the same
    # trick for blacklist_path) - lets this reach the same database
    # without live_server needing to expose db_path itself.
    tracker = Tracker(tmp_path / "db.sqlite3")
    tracker.record_score(
        "job3", "DevOps Engineer", "Acme Corp", "https://example.com/job3", score=20,
        should_apply=False, eligibility="fail",
    )

    with urllib.request.urlopen(f"{live_server}/api/rows?eligibility=fail") as resp:
        body = resp.read().decode("utf-8")

    assert "DevOps Engineer" in body
    assert "Backend Engineer" not in body


def test_api_rows_combines_status_and_eligibility_filters(live_server, tmp_path):
    """status and eligibility were each tested individually above (and
    their AND-not-OR combination was already proven at the Tracker level
    in a8d4a3d), but no test proved the *server* - _parse_list_params()
    parsing a real HTTP query string, not a Tracker call built directly in
    Python - actually reads and combines both params from one request.
    _parse_list_params is hand-written string parsing that a Tracker-level
    test bypasses entirely, so a bug there (e.g. a typo in a dict key)
    would go uncaught without a test that goes through do_GET itself.
    """
    tracker = Tracker(tmp_path / "db.sqlite3")
    # Matches both filters below.
    tracker.record_score(
        "job4", "SRE", "Acme Corp", "https://example.com/job4", score=20,
        should_apply=False, eligibility="fail",
    )
    tracker.update_status("job4", "applied")
    # Matches eligibility=fail but not status=applied (record_score's own
    # should_apply=False set it to "skipped").
    tracker.record_score(
        "job3", "DevOps Engineer", "Acme Corp", "https://example.com/job3", score=20,
        should_apply=False, eligibility="fail",
    )
    # job1 (from the live_server fixture) matches status=applied but has
    # no eligibility set at all (never scored, just mark_applied()'d).

    with urllib.request.urlopen(f"{live_server}/api/rows?status=applied&eligibility=fail") as resp:
        body = resp.read().decode("utf-8")

    assert "SRE" in body
    assert "DevOps Engineer" not in body
    assert "Backend Engineer" not in body


def test_export_csv_respects_the_eligibility_filter(live_server, tmp_path):
    tracker = Tracker(tmp_path / "db.sqlite3")
    tracker.record_score(
        "job3", "DevOps Engineer", "Acme Corp", "https://example.com/job3", score=20,
        should_apply=False, eligibility="fail",
    )

    with urllib.request.urlopen(f"{live_server}/api/export.csv?eligibility=fail") as resp:
        body = resp.read().decode("utf-8")

    assert "job3," in body
    assert "job1," not in body
    assert "job 2," not in body


def test_index_page_reflects_the_eligibility_filter_in_the_select(live_server):
    with urllib.request.urlopen(f"{live_server}/?eligibility=fail") as resp:
        body = resp.read().decode("utf-8")

    assert '<option value="fail" selected>' in body


def test_api_rows_search_also_matches_notes(live_server):
    _post_json(f"{live_server}/api/jobs/job1/note", {"note": "Referred by Jane."})

    with urllib.request.urlopen(f"{live_server}/api/rows?q=Jane") as resp:
        body = resp.read().decode("utf-8")

    assert "Backend Engineer" in body
    assert "Frontend Engineer" not in body


def test_api_rows_search_also_matches_missing_qualifications(live_server, tmp_path):
    """Verifies the search term actually round-trips end-to-end through the
    dashboard's own HTTP query-param parsing (_parse_list_params) into
    Tracker.list_jobs' search, not just that list_jobs itself supports it -
    a separate code path from a direct list_jobs() call, same reasoning as
    test_api_rows_search_also_matches_notes above.
    """
    tracker = Tracker(tmp_path / "db.sqlite3")
    tracker.record_score(
        "job1", "Backend Engineer", "Acme Corp", "https://example.com/job1", score=70,
        should_apply=True, missing_qualifications=["AWS certification"],
    )

    with urllib.request.urlopen(f"{live_server}/api/rows?q=certification") as resp:
        body = resp.read().decode("utf-8")

    assert "Backend Engineer" in body
    assert "Frontend Engineer" not in body


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


def test_api_stats_scopes_counts_to_the_eligibility_filter(tmp_path, live_server):
    """_handle_stats forwards eligibility to Tracker.status_counts() the
    same way it already forwards search (see the test above) - the line
    itself was covered by other tests exercising _handle_stats at all, but
    that never proved the filter actually narrows the pill counts, only
    that the wiring didn't crash.
    """
    tracker = Tracker(tmp_path / "db.sqlite3")
    tracker.record_score(
        "job3", "DevOps Engineer", "Acme Corp", "https://example.com/job3", score=20,
        should_apply=False, eligibility="fail",
    )

    with urllib.request.urlopen(f"{live_server}/api/stats?eligibility=fail") as resp:
        body = resp.read().decode("utf-8")

    assert 'All <span class="count">1</span>' in body
    assert 'data-status="skipped"' in body
    assert 'data-status="applied"' not in body
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


def test_api_jobs_resume_returns_fragment(live_server):
    with urllib.request.urlopen(f"{live_server}/api/jobs/job1/resume") as resp:
        assert resp.status == 200
        body = resp.read().decode("utf-8")
    assert "No tailored resume generated" in body


def test_api_jobs_resume_returns_the_recorded_generation(live_server, tmp_path):
    tracker = Tracker(tmp_path / "db.sqlite3")
    tracker.record_resume_generation(
        "job1", "Backend Engineer", "Acme Corp", "A tailored summary.", ["Python"], ["Did a thing."]
    )

    with urllib.request.urlopen(f"{live_server}/api/jobs/job1/resume") as resp:
        assert resp.status == 200
        body = resp.read().decode("utf-8")
    assert "A tailored summary." in body
    assert "Python" in body
    assert "Did a thing." in body


def test_api_jobs_resume_returns_404_for_unknown_job(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{live_server}/api/jobs/does-not-exist/resume")
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


def _post(url: str, *, same_origin: bool = True):
    """Same as _post_json above, but for the blacklist endpoint - which
    takes no request body at all, so a JSON Content-Type/payload would be
    misleading here.
    """
    headers = {}
    if same_origin:
        split = urlsplit(url)
        headers["Origin"] = f"{split.scheme}://{split.netloc}"
    req = urllib.request.Request(url, method="POST", headers=headers)
    return urllib.request.urlopen(req)


def test_post_blacklist_adds_the_jobs_company(live_server, tmp_path):
    """job1 (this fixture's own setup) is already status="applied" - an
    in-progress application at the very company being blacklisted here -
    so this also doubles as coverage for the in-progress-application
    warning (see Tracker.in_progress_jobs_at_company(), shared with
    cmd_blacklist's own CLI warning): blacklisting only stops future
    applications, so a company you're still actively in process with
    deserves a heads-up in the dashboard too, not just from the CLI.
    """
    resp = _post(f"{live_server}/api/jobs/job1/blacklist")

    assert resp.status == 200
    data = json.loads(resp.read().decode("utf-8"))
    assert data["ok"] is True
    assert data["job_id"] == "job1"
    assert data["company"] == "Acme Corp"
    assert "1 tracked application(s) at Acme Corp are still in progress (applied)" in data["warning"]
    assert CompanyBlacklist(tmp_path / "blacklist.json").is_blocked("Acme Corp")


def test_post_blacklist_rejects_a_cross_origin_request(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post(f"{live_server}/api/jobs/job1/blacklist", same_origin=False)
    assert exc_info.value.code == 403


def test_post_blacklist_on_unknown_job_id_returns_404(live_server, tmp_path):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post(f"{live_server}/api/jobs/does-not-exist/blacklist")
    assert exc_info.value.code == 404
    assert not CompanyBlacklist(tmp_path / "blacklist.json").is_blocked("Acme Corp")


def test_post_blacklist_decodes_percent_encoded_job_id(live_server):
    """"job 2" shares its company ("Acme Corp") with job1, which this
    fixture already marks "applied" - so this response also carries a
    warning, same as test_post_blacklist_adds_the_jobs_company. That's
    incidental to this test's actual purpose (job_id percent-decoding),
    so only the fields relevant to that are asserted here; see
    test_post_blacklist_no_warning_when_nothing_in_progress_at_that_company
    below for the genuinely warning-free case.
    """
    resp = _post(f"{live_server}/api/jobs/job%202/blacklist")

    assert resp.status == 200
    data = json.loads(resp.read().decode("utf-8"))
    assert data["ok"] is True
    assert data["job_id"] == "job 2"
    assert data["company"] == "Acme Corp"


def test_post_blacklist_no_warning_when_nothing_in_progress_at_that_company(live_server, tmp_path):
    tracker = Tracker(tmp_path / "db.sqlite3")
    tracker.upsert_job("job3", "DevOps Engineer", "Globex", "https://example.com/job3")  # status=seen

    resp = _post(f"{live_server}/api/jobs/job3/blacklist")

    assert resp.status == 200
    data = json.loads(resp.read().decode("utf-8"))
    assert data == {"ok": True, "job_id": "job3", "company": "Globex", "warning": None}


def test_get_note_returns_empty_string_when_none_set(live_server):
    with urllib.request.urlopen(f"{live_server}/api/jobs/job1/note") as resp:
        assert resp.headers["Content-Type"] == "application/json"
        data = json.loads(resp.read().decode("utf-8"))
    assert data == {"note": ""}


def test_get_note_on_unknown_job_id_returns_404(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{live_server}/api/jobs/does-not-exist/note")
    assert exc_info.value.code == 404


def test_post_note_sets_and_get_note_reflects_it(live_server):
    resp = _post_json(f"{live_server}/api/jobs/job1/note", {"note": "Recruiter said $150k base."})

    assert resp.status == 200
    data = json.loads(resp.read().decode("utf-8"))
    assert data == {"ok": True, "job_id": "job1", "note": "Recruiter said $150k base."}

    with urllib.request.urlopen(f"{live_server}/api/jobs/job1/note") as resp:
        assert json.loads(resp.read().decode("utf-8")) == {"note": "Recruiter said $150k base."}


def test_post_note_can_clear_an_existing_note(live_server):
    _post_json(f"{live_server}/api/jobs/job1/note", {"note": "First."})

    resp = _post_json(f"{live_server}/api/jobs/job1/note", {"note": ""})

    assert resp.status == 200
    with urllib.request.urlopen(f"{live_server}/api/jobs/job1/note") as resp:
        assert json.loads(resp.read().decode("utf-8")) == {"note": ""}


def test_post_note_rejects_a_cross_origin_request(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(f"{live_server}/api/jobs/job1/note", {"note": "x"}, same_origin=False)
    assert exc_info.value.code == 403


def test_post_note_rejects_non_json_content_type(live_server):
    req = urllib.request.Request(
        f"{live_server}/api/jobs/job1/note",
        data=b"note=x",
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded", "Origin": live_server},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400


def test_post_note_rejects_a_non_string_note_value(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(f"{live_server}/api/jobs/job1/note", {"note": 123})
    assert exc_info.value.code == 400


def test_post_note_rejects_malformed_json_body(live_server):
    req = urllib.request.Request(
        f"{live_server}/api/jobs/job1/note",
        data=b"{not valid json",
        method="POST",
        headers={"Content-Type": "application/json", "Origin": live_server},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400


def test_post_note_rejects_oversized_body(live_server):
    huge_payload = {"note": "x" * 10_000}
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(f"{live_server}/api/jobs/job1/note", huge_payload)
    assert exc_info.value.code == 400


def test_post_note_rejects_a_non_numeric_content_length(live_server):
    """Same real bug/fix as test_post_status_rejects_a_non_numeric_content_
    length above, for the /note endpoint's own, separately-implemented
    Content-Length parsing (server.py has no shared helper for this - see
    that test's docstring for why a raw socket is needed here).
    """
    parts = urlsplit(live_server)
    sock = socket.create_connection((parts.hostname, parts.port), timeout=5)
    try:
        sock.sendall(
            f"POST /api/jobs/job1/note HTTP/1.1\r\n"
            f"Host: {parts.hostname}:{parts.port}\r\n"
            f"Origin: {live_server}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: not-a-number\r\n\r\n".encode()
        )
        chunks = []
        while chunk := sock.recv(4096):
            chunks.append(chunk)
        response = b"".join(chunks).decode("utf-8", errors="replace")
    finally:
        sock.close()

    assert response.startswith("HTTP/1.0 400") or response.startswith("HTTP/1.1 400")
    assert "Invalid Content-Length" in response


def test_post_note_on_unknown_job_id_returns_404(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(f"{live_server}/api/jobs/does-not-exist/note", {"note": "x"})
    assert exc_info.value.code == 404


def test_post_note_decodes_percent_encoded_job_id(live_server):
    resp = _post_json(f"{live_server}/api/jobs/job%202/note", {"note": "x"})

    assert resp.status == 200
    data = json.loads(resp.read().decode("utf-8"))
    assert data == {"ok": True, "job_id": "job 2", "note": "x"}


def test_get_blacklist_is_empty_by_default(live_server):
    with urllib.request.urlopen(f"{live_server}/api/blacklist") as resp:
        assert resp.headers["Content-Type"].startswith("text/html")
        body = resp.read().decode("utf-8")
    assert "Blacklist is empty." in body


def test_get_blacklist_lists_added_companies(live_server, tmp_path):
    CompanyBlacklist(tmp_path / "blacklist.json").add("Acme Corp")

    with urllib.request.urlopen(f"{live_server}/api/blacklist") as resp:
        body = resp.read().decode("utf-8")

    assert "Acme Corp" in body
    assert 'data-company="Acme Corp"' in body


def test_get_blacklist_shows_the_reason_when_set(live_server, tmp_path):
    CompanyBlacklist(tmp_path / "blacklist.json").add("Acme Corp", reason="no H1B sponsorship")

    with urllib.request.urlopen(f"{live_server}/api/blacklist") as resp:
        body = resp.read().decode("utf-8")

    assert "no H1B sponsorship" in body


def test_get_missing_qualifications_is_empty_by_default(live_server):
    with urllib.request.urlopen(f"{live_server}/api/missing-qualifications") as resp:
        assert resp.headers["Content-Type"].startswith("text/html")
        body = resp.read().decode("utf-8")
    assert "No missing qualifications recorded yet." in body


def test_get_missing_qualifications_shows_the_most_common_gaps(live_server, tmp_path):
    tracker = Tracker(tmp_path / "db.sqlite3")
    tracker.record_score(
        "job1",
        "Backend Engineer",
        "Acme",
        "https://x/1",
        score=70,
        should_apply=True,
        missing_qualifications=["Kubernetes experience", "Docker"],
    )
    tracker.record_score(
        "job2",
        "SRE",
        "Beta",
        "https://x/2",
        score=65,
        should_apply=True,
        missing_qualifications=["Kubernetes experience"],
    )

    with urllib.request.urlopen(f"{live_server}/api/missing-qualifications") as resp:
        body = resp.read().decode("utf-8")

    assert body.index("Kubernetes experience") < body.index("Docker")


def test_get_answer_gaps_is_empty_by_default(live_server):
    with urllib.request.urlopen(f"{live_server}/api/answer-gaps") as resp:
        assert resp.headers["Content-Type"].startswith("text/html")
        body = resp.read().decode("utf-8")
    assert "No unanswered required questions recorded." in body


def test_get_answer_gaps_shows_the_most_frequently_seen_question_first(live_server, tmp_path):
    store = AnswerGapStore(tmp_path / "answer_gaps.json")
    store.record("Rare question", job_id="1", company="Acme", title="Backend Engineer")
    for job_id in ("2", "3"):
        store.record("Common question", job_id=job_id, company="Acme", title="Backend Engineer")

    with urllib.request.urlopen(f"{live_server}/api/answer-gaps") as resp:
        body = resp.read().decode("utf-8")

    assert body.index("Common question") < body.index("Rare question")
    assert "Backend Engineer at Acme" in body


def test_get_audit_log_is_empty_by_default(live_server):
    with urllib.request.urlopen(f"{live_server}/api/audit-log") as resp:
        assert resp.headers["Content-Type"].startswith("text/html")
        body = resp.read().decode("utf-8")
    assert "No audit log entries recorded yet." in body


def test_get_audit_log_shows_entries_most_recent_first(live_server, tmp_path):
    audit = AuditLogger(tmp_path / "audit.log")
    audit.log("search", keywords="backend engineer")
    audit.log("applied", job_id="1", company="Acme")

    with urllib.request.urlopen(f"{live_server}/api/audit-log") as resp:
        body = resp.read().decode("utf-8")

    assert body.index("applied") < body.index("search")


def test_get_audit_log_search_filters_entries(live_server, tmp_path):
    audit = AuditLogger(tmp_path / "audit.log")
    audit.log("applied", job_id="1", company="Acme Corp")
    audit.log("applied", job_id="2", company="Beta Inc")

    with urllib.request.urlopen(f"{live_server}/api/audit-log?q=acme") as resp:
        body = resp.read().decode("utf-8")

    assert "Acme Corp" in body
    assert "Beta Inc" not in body


def test_get_audit_log_failures_reads_the_failed_applications_log_instead(live_server, tmp_path):
    """`failures=1` is the dashboard's own switch for `job-bot audit-log
    --failures` - an entry only in one of the two files must never show up
    when reading the other.
    """
    AuditLogger(tmp_path / "audit.log").log("applied", job_id="1", company="Acme")
    AuditLogger(tmp_path / "failed_applications.log").log("prep_error", job_id="2", error="boom")

    with urllib.request.urlopen(f"{live_server}/api/audit-log?failures=1") as resp:
        body = resp.read().decode("utf-8")

    assert "prep_error" in body
    assert "applied" not in body


def test_get_audit_log_without_failures_reads_the_main_audit_log(live_server, tmp_path):
    AuditLogger(tmp_path / "audit.log").log("applied", job_id="1", company="Acme")
    AuditLogger(tmp_path / "failed_applications.log").log("prep_error", job_id="2", error="boom")

    with urllib.request.urlopen(f"{live_server}/api/audit-log") as resp:
        body = resp.read().decode("utf-8")

    assert "applied" in body
    assert "prep_error" not in body


def test_post_blacklist_remove_removes_the_company(live_server, tmp_path):
    CompanyBlacklist(tmp_path / "blacklist.json").add("Acme Corp")

    resp = _post_json(f"{live_server}/api/blacklist/remove", {"company": "Acme Corp"})

    assert resp.status == 200
    data = json.loads(resp.read().decode("utf-8"))
    assert data == {"ok": True, "company": "Acme Corp", "removed": True}
    assert not CompanyBlacklist(tmp_path / "blacklist.json").is_blocked("Acme Corp")


def test_post_blacklist_remove_of_absent_company_reports_not_removed(live_server, tmp_path):
    resp = _post_json(f"{live_server}/api/blacklist/remove", {"company": "Nobody Inc"})

    assert resp.status == 200
    data = json.loads(resp.read().decode("utf-8"))
    assert data == {"ok": True, "company": "Nobody Inc", "removed": False}


def test_post_blacklist_remove_rejects_a_cross_origin_request(live_server, tmp_path):
    CompanyBlacklist(tmp_path / "blacklist.json").add("Acme Corp")

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(f"{live_server}/api/blacklist/remove", {"company": "Acme Corp"}, same_origin=False)
    assert exc_info.value.code == 403
    assert CompanyBlacklist(tmp_path / "blacklist.json").is_blocked("Acme Corp")


def test_post_blacklist_remove_rejects_non_json_content_type(live_server):
    req = urllib.request.Request(
        f"{live_server}/api/blacklist/remove",
        data=b"company=x",
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded", "Origin": live_server},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400


def test_post_blacklist_remove_rejects_a_non_string_company_value(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(f"{live_server}/api/blacklist/remove", {"company": 123})
    assert exc_info.value.code == 400


def test_post_blacklist_remove_rejects_malformed_json_body(live_server):
    req = urllib.request.Request(
        f"{live_server}/api/blacklist/remove",
        data=b"{not valid json",
        method="POST",
        headers={"Content-Type": "application/json", "Origin": live_server},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400


def test_post_blacklist_remove_rejects_oversized_body(live_server):
    huge_payload = {"company": "x" * 10_000}
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(f"{live_server}/api/blacklist/remove", huge_payload)
    assert exc_info.value.code == 400


def test_post_blacklist_remove_rejects_a_non_numeric_content_length(live_server):
    """Same real bug/fix as test_post_status_rejects_a_non_numeric_content_
    length above, for /api/blacklist/remove's own, separately-implemented
    Content-Length parsing.
    """
    parts = urlsplit(live_server)
    sock = socket.create_connection((parts.hostname, parts.port), timeout=5)
    try:
        sock.sendall(
            f"POST /api/blacklist/remove HTTP/1.1\r\n"
            f"Host: {parts.hostname}:{parts.port}\r\n"
            f"Origin: {live_server}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: not-a-number\r\n\r\n".encode()
        )
        chunks = []
        while chunk := sock.recv(4096):
            chunks.append(chunk)
        response = b"".join(chunks).decode("utf-8", errors="replace")
    finally:
        sock.close()

    assert response.startswith("HTTP/1.0 400") or response.startswith("HTTP/1.1 400")
    assert "Invalid Content-Length" in response


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

    run_dashboard(
        tmp_path / "db.sqlite3",
        tmp_path / "blacklist.json",
        tmp_path / "audit.log",
        tmp_path / "failed_applications.log",
        tmp_path / "answer_gaps.json",
        port=0,
        open_browser=True,
    )

    assert len(opened_urls) == 1
    assert opened_urls[0].startswith("http://127.0.0.1:")


def test_run_dashboard_raises_a_clear_error_when_the_port_is_already_in_use(tmp_path):
    """Real failure this guards against: running `job-bot dashboard` while a
    previous instance is still running (a forgotten terminal tab is the
    common case) previously crashed with a raw OSError ("Address already in
    use") traceback instead of a clean, actionable message - see
    DashboardPortInUse.
    """
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    try:
        taken_port = blocker.getsockname()[1]
        with pytest.raises(DashboardPortInUse, match=r"already running"):
            run_dashboard(
                tmp_path / "db.sqlite3",
                tmp_path / "blacklist.json",
                tmp_path / "audit.log",
                tmp_path / "failed_applications.log",
                tmp_path / "answer_gaps.json",
                port=taken_port,
            )
    finally:
        blocker.close()


def test_run_dashboard_raises_a_clear_error_for_an_out_of_range_port(tmp_path):
    """--port/DASHBOARD_PORT has no range check of its own before reaching
    socket.bind() - a typo like an extra digit previously surfaced as a raw
    OverflowError traceback instead of DashboardPortInUse's clean message.
    """
    with pytest.raises(DashboardPortInUse):
        run_dashboard(
            tmp_path / "db.sqlite3",
            tmp_path / "blacklist.json",
            tmp_path / "audit.log",
            tmp_path / "failed_applications.log",
            tmp_path / "answer_gaps.json",
            port=99999999,
        )


def test_qa_endpoint_decodes_percent_encoded_job_id(live_server):
    with urllib.request.urlopen(f"{live_server}/api/jobs/job%202/qa") as resp:
        assert resp.status == 200
