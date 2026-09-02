from job_bot.dashboard.render import render_page_html, render_rows_html


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


def test_render_rows_html_handles_missing_score_and_applied_at():
    job = make_job(match_score=None, applied_at=None)
    html = render_rows_html([job])
    assert ">-<" in html


def test_render_page_html_includes_title_and_refresh_script():
    html = render_page_html([make_job()], refresh_seconds=10)
    assert "job_bot tracker" in html
    assert "setInterval(refresh, 10000)" in html
    assert "Backend Engineer" in html
