from datetime import UTC, datetime, timedelta

from job_bot.dashboard.render import (
    render_blacklist_html,
    render_page_html,
    render_qa_html,
    render_rows_html,
    render_stats_html,
)


def make_job(**overrides):
    defaults = dict(
        job_id="job1",
        title="Backend Engineer",
        company="Acme Corp",
        url="https://example.com/job1",
        match_score=82,
        status="applied",
        first_seen_at="2026-01-01T00:00:00+00:00",
        applied_at="2026-01-02T00:00:00+00:00",
    )
    defaults.update(overrides)
    return defaults


def test_render_rows_html_empty_state():
    html = render_rows_html([])
    assert "No jobs tracked yet" in html


def test_render_rows_html_includes_job_fields():
    html = render_rows_html([make_job()])
    assert "Backend Engineer" in html
    assert "Acme Corp" in html
    assert "82" in html
    assert "job1" in html
    assert 'href="https://example.com/job1"' in html


def test_render_rows_html_escapes_company_name_to_prevent_xss():
    malicious = make_job(company="<script>alert(1)</script>")
    html = render_rows_html([malicious])
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_rows_html_includes_a_blacklist_button_per_row():
    html = render_rows_html([make_job()])
    assert 'class="blacklist-button" data-job-id="job1"' in html
    assert "Never apply to Acme Corp again" in html


def test_render_rows_html_escapes_company_name_in_the_blacklist_button_title():
    malicious = make_job(company='Acme"><script>alert(1)</script>')
    html = render_rows_html([malicious])
    assert "<script>alert(1)</script>" not in html


def test_render_rows_html_shows_match_reasoning_as_a_score_tooltip():
    html = render_rows_html([make_job(match_reasoning="Strong Python and AWS overlap with the posting.")])
    assert 'title="Strong Python and AWS overlap with the posting."' in html


def test_render_rows_html_omits_score_tooltip_when_no_reasoning_recorded():
    """A job upserted (not scored - see Tracker.upsert_job() vs.
    record_score()) has no match_reasoning at all, and one scored before
    this feature existed has it as "" (the migrated column's default) -
    neither should render a score cell with an empty title="" tooltip.
    Checks the <td> itself, not just absence of the word "title" anywhere
    in the row - the Note/Blacklist buttons legitimately have their own
    title attributes regardless of match_reasoning.
    """
    for reasoning in (None, ""):
        html = render_rows_html([make_job(match_reasoning=reasoning)])
        assert "<td title=" not in html


def test_render_rows_html_escapes_match_reasoning_in_the_score_tooltip():
    malicious = make_job(match_reasoning='Great fit"><script>alert(1)</script>')
    html = render_rows_html([malicious])
    assert "<script>alert(1)</script>" not in html


def test_render_rows_html_marks_a_failed_eligibility_verdict_in_the_score_cell():
    html = render_rows_html([make_job(eligibility="fail")])
    assert "⚠️ 82" in html


def test_render_rows_html_marks_a_flagged_eligibility_verdict_in_the_score_cell():
    html = render_rows_html([make_job(eligibility="flag")])
    assert "⚠️ 82" in html


def test_render_rows_html_does_not_mark_a_passing_eligibility_verdict():
    html = render_rows_html([make_job(eligibility="pass")])
    assert "⚠️" not in html


def test_render_rows_html_shows_the_eligibility_note_in_the_score_tooltip():
    html = render_rows_html(
        [make_job(eligibility="fail", eligibility_note="Requires active US security clearance.")]
    )
    assert "title=" in html
    assert "Eligibility: fail - Requires active US security clearance." in html


def test_render_rows_html_omits_the_note_suffix_when_eligibility_note_is_empty():
    html = render_rows_html([make_job(eligibility="flag", eligibility_note="")])
    assert "Eligibility: flag" in html
    assert "Eligibility: flag -" not in html


def test_render_rows_html_combines_eligibility_and_reasoning_in_one_tooltip():
    html = render_rows_html(
        [
            make_job(
                eligibility="fail",
                eligibility_note="Requires US citizenship.",
                match_reasoning="Otherwise a strong technical match.",
            )
        ]
    )
    assert "Eligibility: fail - Requires US citizenship.\nOtherwise a strong technical match." in html


def test_render_rows_html_escapes_the_eligibility_note_in_the_score_tooltip():
    malicious = make_job(eligibility="fail", eligibility_note='x"><script>alert(1)</script>')
    html = render_rows_html([malicious])
    assert "<script>alert(1)</script>" not in html


def test_render_rows_html_marks_a_stale_application():
    stale_job = make_job(
        status="applied", applied_at=(datetime.now(UTC) - timedelta(days=20)).isoformat()
    )
    html = render_rows_html([stale_job], stale_after_days=14)
    assert "⏰" in html
    assert "No reply after 14+ days" in html


def test_render_rows_html_does_not_mark_a_recent_application_as_stale():
    recent_job = make_job(status="applied", applied_at=datetime.now(UTC).isoformat())
    html = render_rows_html([recent_job], stale_after_days=14)
    assert "⏰" not in html


def test_render_rows_html_does_not_mark_a_non_applied_job_as_stale():
    """Only status="applied" jobs are eligible - the same status
    job-bot report --stale-days itself requires (Tracker._stale_
    applications() filters to status="applied" too), since a job that
    was never actually applied to has no "reply" to be waiting on.
    """
    old_but_not_applied = make_job(
        status="skipped", applied_at=(datetime.now(UTC) - timedelta(days=20)).isoformat()
    )
    html = render_rows_html([old_but_not_applied], stale_after_days=14)
    assert "⏰" not in html


def test_render_rows_html_omits_stale_marker_when_stale_after_days_not_given():
    stale_job = make_job(
        status="applied", applied_at=(datetime.now(UTC) - timedelta(days=20)).isoformat()
    )
    html = render_rows_html([stale_job])
    assert "⏰" not in html


def test_render_rows_html_includes_a_note_button_per_row():
    html = render_rows_html([make_job()])
    assert 'class="note-button" data-job-id="job1"' in html


def test_render_rows_html_marks_the_note_button_when_a_note_exists():
    html = render_rows_html([make_job(notes="Referred by Jane.")])
    assert 'class="note-button has-note"' in html
    assert 'title="Edit note"' in html


def test_render_rows_html_note_button_has_no_has_note_class_when_unset():
    html = render_rows_html([make_job(notes=None)])
    assert 'class="note-button"' in html
    assert "has-note" not in html
    assert 'title="Add a note"' in html


def test_render_rows_html_escapes_title_and_url():
    malicious = make_job(title='"><img src=x onerror=alert(1)>', url='javascript:alert(1)"')
    html = render_rows_html([malicious])
    assert "<img src=x onerror=alert(1)>" not in html


def test_render_rows_html_does_not_render_javascript_scheme_as_a_link():
    """javascript: (and any other non-http(s) scheme) must never reach an
    href - html.escape() alone doesn't neutralize it since it contains no
    HTML metacharacters. See app/SECURITY.md's dashboard XSS note.
    """
    malicious = make_job(title="Click me", url="javascript:alert(document.cookie)")
    html = render_rows_html([malicious])
    assert "javascript:" not in html
    assert "<a " not in html
    assert "Click me" in html  # still shown, just not as a link


def test_render_rows_html_does_not_render_data_scheme_as_a_link():
    malicious = make_job(url="data:text/html,<script>alert(1)</script>")
    html = render_rows_html([malicious])
    assert "data:text/html" not in html
    assert "<a " not in html


def test_render_rows_html_renders_http_and_https_as_links():
    html = render_rows_html([make_job(url="http://example.com/job")])
    assert 'href="http://example.com/job"' in html
    html = render_rows_html([make_job(url="https://example.com/job")])
    assert 'href="https://example.com/job"' in html


def test_render_rows_html_handles_empty_or_malformed_url_gracefully():
    html = render_rows_html([make_job(url="")])
    assert "<a " not in html
    html = render_rows_html([make_job(url="not a url at all ::::")])
    assert "<a " not in html


def test_render_rows_html_handles_a_url_urlparse_itself_rejects():
    """Most malformed strings just parse to an empty/unrecognized scheme
    (see the test above), but a few (a malformed IPv6 host, e.g.) make
    urlparse() itself raise ValueError rather than returning a harmless
    empty scheme - still must render as plain text, not a broken page.
    """
    html = render_rows_html([make_job(title="Click me", url="http://[::1")])
    assert "<a " not in html
    assert "Click me" in html


def test_render_rows_html_handles_missing_score_and_applied_at():
    job = make_job(match_score=None, applied_at=None)
    html = render_rows_html([job])
    assert ">-<" in html


def test_render_page_html_includes_title_and_refresh_script():
    html = render_page_html([make_job()], refresh_seconds=10)
    assert "job_bot tracker" in html
    assert "}, 10000);" in html
    assert "Backend Engineer" in html


def test_render_rows_html_includes_status_select_and_qa_button():
    html = render_rows_html([make_job(job_id="job1", status="applied")])
    assert 'data-job-id="job1"' in html
    assert '<option value="applied" selected>' in html
    assert 'class="qa-button"' in html


def test_render_rows_html_status_select_escapes_job_id():
    malicious = make_job(job_id='job1" onmouseover="alert(1)')
    html = render_rows_html([malicious])
    assert 'job1" onmouseover="alert(1)' not in html  # raw attribute breakout not present
    assert "&quot;" in html


def test_render_rows_html_includes_current_status_even_if_not_a_known_value():
    """Defensive: if the DB somehow held a status outside TRACKER_STATUSES,
    the select should still surface it rather than silently showing nothing
    selected or dropping the row.
    """
    html = render_rows_html([make_job(status="mystery_status")])
    assert "mystery status" in html


def test_render_rows_html_status_badge_is_escaped_exactly_once():
    """_badge() used to receive an already-html.escape()'d status string
    from render_rows_html and escape it again, producing double-escaped
    entities ("R&amp;amp;D" instead of "R&amp;D") - and since STATUS_COLORS
    is keyed by the raw status text, the escaped key always missed the
    lookup too, silently falling back to the default gray badge color for
    any status containing an HTML metacharacter.
    """
    html = render_rows_html([make_job(status="R&D")])
    assert "R&amp;D" in html
    assert "R&amp;amp;D" not in html


def test_render_qa_html_empty_state():
    html = render_qa_html([])
    assert "No answered questions" in html


def test_render_qa_html_escapes_question_and_answer():
    qa = [{"question": "<script>alert(1)</script>", "answer": "5 years"}]
    html = render_qa_html(qa)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "5 years" in html


def test_render_qa_html_renders_all_entries_in_order():
    qa = [
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"},
    ]
    html = render_qa_html(qa)
    assert html.index("Q1") < html.index("Q2")


def test_render_blacklist_html_empty_state():
    html = render_blacklist_html([])
    assert "Blacklist is empty." in html


def test_render_blacklist_html_renders_a_remove_button_per_company():
    html = render_blacklist_html(["Acme Corp", "Beta Inc"])
    assert "Acme Corp" in html
    assert "Beta Inc" in html
    assert html.count("blacklist-remove-button") == 2
    assert 'data-company="Acme Corp"' in html
    assert 'data-company="Beta Inc"' in html


def test_render_blacklist_html_escapes_company_name_to_prevent_xss():
    html = render_blacklist_html(['<script>alert(1)</script>Acme"'])
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_page_html_includes_the_manage_blacklist_button_and_dialog():
    html = render_page_html([make_job()])
    assert 'id="manageBlacklist"' in html
    assert 'id="blacklistDialog"' in html
    assert 'id="blacklistContent"' in html


def test_render_page_html_includes_the_note_dialog():
    html = render_page_html([make_job()])
    assert 'id="noteDialog"' in html
    assert 'id="noteTextarea"' in html
    assert 'id="noteSave"' in html


def test_render_page_html_does_not_leak_search_term_into_script_context():
    """A search term containing "</script>" must never appear unescaped in
    the inline <script> block - the HTML parser would close the tag on that
    literal text regardless of any JS-string escaping. See render.py's note
    on reading initial state from the DOM instead of interpolating it.
    """
    html = render_page_html([make_job()], search="</script><script>alert(1)</script>")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;/script&gt;" in html


def test_render_page_html_includes_filter_and_sort_controls():
    html = render_page_html([make_job()], status="applied", search="engineer", sort="company", direction="asc")
    assert 'id="q"' in html
    assert 'id="status"' in html
    assert 'id="sort"' in html
    assert 'value="engineer"' in html
    assert '<option value="applied" selected>' in html
    assert '<option value="company:asc" selected>' in html


def test_render_page_html_includes_the_eligibility_filter_control():
    html = render_page_html([make_job()], eligibility="fail")
    assert 'id="eligibility"' in html
    assert '<option value="fail" selected>' in html


def test_render_page_html_eligibility_filter_defaults_to_any():
    html = render_page_html([make_job()])
    assert '<option value="" selected>Any eligibility</option>' in html


def test_render_page_html_includes_pager():
    html = render_page_html([make_job()], total=100, page=2, page_size=25)
    assert 'id="prevPage"' in html
    assert 'id="nextPage"' in html
    assert "page: 2," in html


def test_render_page_html_includes_export_csv_link():
    html = render_page_html([make_job()])
    assert 'id="exportCsv"' in html
    assert 'href="/api/export.csv"' in html
    assert "exportCsv').href" in html  # kept in sync with filters by refresh()


def test_render_page_html_includes_export_json_link():
    html = render_page_html([make_job()])
    assert 'id="exportJson"' in html
    assert 'href="/api/export.json"' in html
    assert "exportJson').href" in html  # kept in sync with filters by refresh()


def test_render_stats_html_shows_all_pill_with_summed_total():
    html = render_stats_html({"applied": 2, "seen": 3}, selected_status="")
    assert "All <span class=\"count\">5</span>" in html
    assert 'data-status=""' in html


def test_render_stats_html_marks_selected_status_active():
    html = render_stats_html({"applied": 2, "seen": 3}, selected_status="applied")
    assert '<button type="button" class="stat-pill active" data-status="applied"' in html
    assert '<button type="button" class="stat-pill" data-status="seen"' in html


def test_render_stats_html_hides_zero_count_statuses_unless_selected():
    html = render_stats_html({"applied": 2}, selected_status="")
    assert "seen" not in html
    assert "offer" not in html

    # A status can legitimately have zero matches right now (e.g. the last
    # "offer" job was just moved to "interviewing") while still being the
    # active filter - it must stay visible so the user can click off of it.
    html = render_stats_html({"applied": 2}, selected_status="offer")
    assert 'data-status="offer"' in html


def test_render_stats_html_escapes_unknown_status_value():
    html = render_stats_html({"<script>alert(1)</script>": 1}, selected_status="")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_page_html_includes_stats_bar():
    html = render_page_html([make_job()], counts={"applied": 2, "seen": 1}, status="applied")
    assert 'id="stats"' in html
    assert 'class="stat-pill active" data-status="applied"' in html
    assert "/api/stats" in html


def test_render_page_html_defaults_to_empty_stats_when_counts_omitted():
    html = render_page_html([make_job()])
    assert 'id="stats"' in html
    assert 'All <span class="count">0</span>' in html
