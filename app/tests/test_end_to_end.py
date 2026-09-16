"""End-to-end workflow test: search -> score -> generate -> apply -> track ->
report/export -> dashboard -> gmail-sync -> re-run.

Unlike test_cli_run.py (which fakes the browser adapter itself to test
cmd_run's orchestration), this drives the *real* LinkedInAdapter with a real
Playwright browser against local file:// fixtures. Everything else is real
too: the SQLite tracker, rate limiter, blacklist, audit log, artifact
writing, the dashboard's HTTP server, and the Gmail sync logic. Only two
things are faked, because they're the only ones that would reach the
network: the LLM provider and the Gmail API client.

What this covers that the per-module tests can't: the seams between them -
that a scraped relative href survives into a URL the dashboard can link to,
that an applied job is recorded once and never re-applied to on a second
run, and that an email arriving later moves that same row forward.
"""

import json
import threading
import urllib.request
from datetime import date
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pytest
from playwright.sync_api import sync_playwright

from job_bot.cli import cmd_export, cmd_report, cmd_run
from job_bot.config import Settings
from job_bot.dashboard.server import make_handler
from job_bot.integrations.gmail_client import EmailMessage
from job_bot.integrations.gmail_sync import sync_gmail
from job_bot.llm.base import LLMProvider
from job_bot.models.schemas import (
    ApplicationAnswer,
    CoverLetter,
    EmailClassification,
    JobMatchScore,
    TailoredResume,
)
from job_bot.safety.audit_log import AuditLogger
from job_bot.tracker.db import Tracker

FIXTURES = Path(__file__).parent / "fixtures"
SEARCH_FIXTURE = FIXTURES / "search_results_e2e.html"
POSTING_FIXTURE = FIXTURES / "job_posting_page.html"

APPLICABLE_JOB_ID = "901"
LOW_SCORE_JOB_ID = "902"
BLACKLISTED_JOB_ID = "903"

# Each posting serves its own page, so the description the scorer sees
# actually differs per job the way it would in a real run.
POSTING_FIXTURES = {
    APPLICABLE_JOB_ID: POSTING_FIXTURE,
    LOW_SCORE_JOB_ID: FIXTURES / "job_posting_page_data_engineer.html",
}


class FakeProvider(LLMProvider):
    """Scores job 901 as a strong match and job 902 as a weak one, so one
    run exercises both the apply path and the skip path. Records every
    schema it was asked for, which is how the re-run assertions detect that
    an already-scored job wasn't sent to the LLM a second time.
    """

    def __init__(self):
        self.schemas_requested: list[type] = []
        self.email_classification = EmailClassification(
            is_job_related=True,
            category="offer",
            company_guess="Acme Corp",
            role_guess="Backend Engineer",
            confidence=0.95,
        )

    def generate_structured(self, *, system, prompt, schema):
        self.schemas_requested.append(schema)
        if schema is JobMatchScore:
            # Stands in for a real scorer reading the posting text: the Scala
            # data-engineering role is a poor fit for this Python resume.
            weak = "Scala" in prompt
            return JobMatchScore(
                eligibility="pass",
                technical_fit=40 if weak else 92,
                experience_fit=40 if weak else 90,
                culture_fit=50 if weak else 88,
                score=42 if weak else 91,
                reasoning="Weak match" if weak else "Strong match",
                should_apply=not weak,
            )
        if schema is TailoredResume:
            return TailoredResume(
                summary="Backend engineer with 6 years of Python experience.",
                highlighted_skills=["Python", "PostgreSQL", "AWS"],
                bullet_points=["Built and operated data services in Python."],
            )
        if schema is CoverLetter:
            return CoverLetter(body="Dear Acme Corp, I would love to join your team.")
        if schema is ApplicationAnswer:
            answer = "Yes" if "authorized to work" in prompt else "6"
            return ApplicationAnswer(answer=answer, confidence=0.95, based_on_resume=True)
        if schema is EmailClassification:
            return self.email_classification
        raise AssertionError(f"Unexpected schema requested: {schema}")


class FakeGmailClient:
    def __init__(self, emails: list[EmailMessage]):
        self._emails = emails

    def search_messages(self, query: str, max_results: int = 50) -> list[EmailMessage]:
        return self._emails[:max_results]


@pytest.fixture
def browser_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        yield page
        browser.close()


@pytest.fixture
def settings(tmp_path) -> Settings:
    resume_path = tmp_path / "resume.txt"
    resume_path.write_text(
        "Jane Doe - Senior Backend Engineer\n"
        "6 years of professional Python experience. FastAPI, PostgreSQL, AWS.\n"
        "Authorized to work in the US.",
        encoding="utf-8",
    )
    blacklist_path = tmp_path / "blacklist.json"
    blacklist_path.write_text(json.dumps(["Blocked Corp"]), encoding="utf-8")
    return Settings(
        _env_file=None,
        llm_provider="claude",
        anthropic_api_key="sk-ant-fake",
        resume_path=resume_path,
        faq_path=tmp_path / "faq.json",
        blacklist_path=blacklist_path,
        db_path=tmp_path / "job_bot.sqlite3",
        browser_profile_dir=tmp_path / "profile",
        audit_log_path=tmp_path / "audit.log",
        failed_applications_log_path=tmp_path / "failed_applications.log",
        applications_dir=tmp_path / "applications",
        daily_application_cap=5,
        require_confirm_before_submit=False,
    )


@pytest.fixture
def wired_run(browser_page, monkeypatch):
    """Points the real adapter's navigation at local fixtures and hands back
    a callable that runs cmd_run end to end.
    """
    import contextlib

    real_goto = browser_page.goto

    def fake_goto(url, **kwargs):
        parsed = urlparse(url)
        if parsed.path.rstrip("/").endswith("/jobs/search"):
            # Every search page serves the same results, which is how
            # LinkedIn behaves once you page past the last result - the
            # adapter sees only already-seen job ids on page 2 and stops.
            return real_goto(f"file://{SEARCH_FIXTURE}")
        job_id = parsed.path.rstrip("/").rsplit("/", 1)[-1]
        return real_goto(f"file://{POSTING_FIXTURES.get(job_id, POSTING_FIXTURE)}")

    monkeypatch.setattr(browser_page, "goto", fake_goto)
    # The adapter's human-scale UI pauses aren't useful in a test.
    monkeypatch.setattr("job_bot.browser.linkedin_adapter.ACTION_DELAY_SECONDS", 0)

    class FakeContext:
        def new_page(self):
            return browser_page

    @contextlib.contextmanager
    def fake_browser_session(profile_dir, headless=False, cdp_url=None):
        yield FakeContext()

    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)

    def run(settings, provider, **arg_overrides):
        monkeypatch.setattr("job_bot.cli.get_provider", lambda _settings: provider)
        args = run_args(**arg_overrides)
        cmd_run(settings, args)

    return run


def run_args(**overrides):
    import argparse

    defaults = dict(
        keywords="backend engineer",
        location="Remote",
        max_apps=5,
        search_pool=25,
        dry_run=False,
        headless=True,
        provider=None,
        model=None,
        yes_i_understand_the_risk=True,
        min_score=None,
        exclude_title_keywords=None,
        experience_level=None,
        loop=False,
        loop_interval_minutes=20,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def report_args(**overrides):
    import argparse

    defaults = dict(stale_days=None, by_score=False)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _audit_actions(settings) -> list[str]:
    lines = settings.audit_log_path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line)["action"] for line in lines]


def test_full_run_applies_scores_and_records_everything(settings, wired_run):
    provider = FakeProvider()

    wired_run(settings, provider)

    tracker = Tracker(settings.db_path)

    # The strong match was applied to for real (the fixture's Submit button
    # was clicked), and recorded exactly once.
    applied = tracker.get_job(APPLICABLE_JOB_ID)
    assert applied["status"] == "applied"
    assert applied["applied_at"] is not None
    assert applied["match_score"] == 91
    assert tracker.has_applied(APPLICABLE_JOB_ID) is True

    # The weak match was scored and skipped without applying.
    skipped = tracker.get_job(LOW_SCORE_JOB_ID)
    assert skipped["status"] == "skipped"
    assert tracker.has_applied(LOW_SCORE_JOB_ID) is False

    # The blacklisted company was never even scored or tracked.
    assert tracker.get_job(BLACKLISTED_JOB_ID) is None

    # The scraped relative href resolved to a real, absolute LinkedIn URL.
    assert applied["url"] == "https://www.linkedin.com/jobs/view/901/?refId=e2e"

    # Generated material landed on disk for the job actually applied to,
    # under today's date folder (see generation/artifacts.py's _job_dir()).
    job_dir = settings.applications_dir / date.today().isoformat() / "901 - Acme Corp - Backend Engineer"
    assert (job_dir / "tailored_resume.txt").read_text(encoding="utf-8").startswith("SUMMARY")
    assert "Acme Corp" in (job_dir / "cover_letter.txt").read_text(encoding="utf-8")
    dated_dir = settings.applications_dir / date.today().isoformat()
    assert list(dated_dir.glob(f"{LOW_SCORE_JOB_ID}*")) == []

    # Every form question the adapter couldn't fill from context went to the
    # LLM, and each answer was both recorded against the job and cached for
    # reuse on future applications.
    answers_by_question = {entry["question"]: entry["answer"] for entry in tracker.list_qa(APPLICABLE_JOB_ID)}
    assert answers_by_question == {
        "Years of Python experience": "6",
        "Are you authorized to work in the US?": "Yes",
    }
    assert json.loads(settings.faq_path.read_text(encoding="utf-8")) == answers_by_question

    # The audit trail tells the whole story, in order.
    actions = _audit_actions(settings)
    assert actions[0] == "search"
    assert "skip_blacklisted" in actions
    assert "generated_materials" in actions
    assert "applied" in actions


def test_second_run_neither_reapplies_nor_rescores(settings, wired_run):
    """The dedup guarantee, across two full runs against the same postings:
    the applied job must not be submitted again (a duplicate real
    application), and neither already-decided job should cost another LLM
    scoring call.
    """
    first_provider = FakeProvider()
    wired_run(settings, first_provider)

    second_provider = FakeProvider()
    wired_run(settings, second_provider)

    tracker = Tracker(settings.db_path)
    assert tracker.get_job(APPLICABLE_JOB_ID)["status"] == "applied"
    # Nothing was sent to the LLM at all on the second pass: job 901 is
    # already applied, 902 is already skipped, 903 is blacklisted.
    assert second_provider.schemas_requested == []

    applied_events = [action for action in _audit_actions(settings) if action == "applied"]
    assert len(applied_events) == 1


def test_run_then_report_and_export_reflect_the_same_state(settings, wired_run, capsys):
    provider = FakeProvider()
    wired_run(settings, provider)
    capsys.readouterr()  # discard the run's own output

    cmd_report(settings, report_args(by_score=True))
    report_out = capsys.readouterr().out
    assert "applied" in report_out
    assert "skipped" in report_out
    assert "Outcomes by match score:" in report_out
    assert "90-100" in report_out  # the applied job's bucket

    cmd_export(settings, run_args(status="applied", out=None))
    csv_out = capsys.readouterr().out
    assert APPLICABLE_JOB_ID in csv_out
    assert "https://www.linkedin.com/jobs/view/901/?refId=e2e" in csv_out
    assert LOW_SCORE_JOB_ID not in csv_out


def test_dashboard_serves_the_run_result_and_accepts_a_status_change(settings, wired_run):
    provider = FakeProvider()
    wired_run(settings, provider)

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(settings.db_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(f"{base}/") as resp:
            page = resp.read().decode("utf-8")
        # The posting is rendered as a real, clickable LinkedIn link - the
        # relative-href bug would have produced an href pointing back at the
        # dashboard's own origin instead.
        assert 'href="https://www.linkedin.com/jobs/view/901/?refId=e2e"' in page
        assert "Backend Engineer" in page

        with urllib.request.urlopen(f"{base}/api/jobs/{APPLICABLE_JOB_ID}/qa") as resp:
            qa_html = resp.read().decode("utf-8")
        assert "Years of Python experience" in qa_html

        request = urllib.request.Request(
            f"{base}/api/jobs/{APPLICABLE_JOB_ID}/status",
            data=json.dumps({"status": "interviewing"}).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", "Origin": base},
        )
        with urllib.request.urlopen(request) as resp:
            assert resp.status == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert Tracker(settings.db_path).get_job(APPLICABLE_JOB_ID)["status"] == "interviewing"


def test_gmail_sync_moves_the_applied_job_forward_after_the_run(settings, wired_run):
    provider = FakeProvider()
    wired_run(settings, provider)

    tracker = Tracker(settings.db_path)
    gmail_client = FakeGmailClient(
        [
            EmailMessage(
                id="m1",
                subject="Your offer from Acme Corp",
                sender="recruiting@acme.example",
                date="2026-09-10",
                snippet="We are delighted to extend an offer",
                body_text="We are delighted to extend an offer for the Backend Engineer role.",
            )
        ]
    )

    result = sync_gmail(
        provider,
        gmail_client,
        tracker,
        days=14,
        max_emails=10,
        confidence_threshold=0.6,
        dry_run=False,
        audit=AuditLogger(settings.audit_log_path),
    )

    assert result.updated == [(APPLICABLE_JOB_ID, "Acme Corp", "offer")]
    assert tracker.get_job(APPLICABLE_JOB_ID)["status"] == "offer"
    # has_applied() keys off applied_at, so the outcome change doesn't undo
    # the record that a real application was submitted.
    assert tracker.has_applied(APPLICABLE_JOB_ID) is True


def test_gmail_sync_never_touches_a_skipped_job(settings, wired_run):
    """The bot chose not to apply to job 902, so there's no application
    behind it for an email to be about - a loosely-matched email must not
    resurrect it as though one had been submitted.
    """
    provider = FakeProvider()
    provider.email_classification = EmailClassification(
        is_job_related=True,
        category="interview_invite",
        company_guess="Beta Industries",
        role_guess="Data Engineer",
        confidence=0.95,
    )
    wired_run(settings, provider)

    tracker = Tracker(settings.db_path)
    gmail_client = FakeGmailClient(
        [
            EmailMessage(
                id="m2",
                subject="Interview with Beta Industries",
                sender="recruiting@beta.example",
                date="2026-09-10",
                snippet="Let's schedule a call",
                body_text="Let's schedule a call about the Data Engineer role.",
            )
        ]
    )

    result = sync_gmail(provider, gmail_client, tracker, dry_run=False)

    assert result.updated == []
    assert tracker.get_job(LOW_SCORE_JOB_ID)["status"] == "skipped"


def test_dry_run_stops_before_submitting_and_records_nothing_as_applied(settings, wired_run):
    provider = FakeProvider()

    wired_run(settings, provider, dry_run=True)

    tracker = Tracker(settings.db_path)
    job = tracker.get_job(APPLICABLE_JOB_ID)
    assert job["status"] == "seen"
    assert job["applied_at"] is None
    assert tracker.has_applied(APPLICABLE_JOB_ID) is False
    assert "dry_run_stopped" in _audit_actions(settings)
    assert "applied" not in _audit_actions(settings)
