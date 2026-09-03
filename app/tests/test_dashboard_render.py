from job_bot.dashboard.render import render_page_html, render_qa_html, render_rows_html


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


def test_render_page_html_includes_pager():
    html = render_page_html([make_job()], total=100, page=2, page_size=25)
    assert 'id="prevPage"' in html
    assert 'id="nextPage"' in html
    assert "page: 2," in html
