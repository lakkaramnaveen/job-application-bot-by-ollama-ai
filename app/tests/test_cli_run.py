"""End-to-end test of `job-bot run`'s orchestration in cli.cmd_run, with the
browser and LLM provider faked out. This is deliberately an integration test
of the wiring itself (which module calls which, with what) rather than a
unit test of any one piece - it's exactly the kind of test that would have
caught resume tailoring being generated but never called (job_bot.generation
.resume_tailor existed, fully implemented and tested in isolation, but
cmd_run never invoked it until this was fixed).
"""

import argparse
import json
from contextlib import contextmanager

from job_bot.browser.base_adapter import JobPosting
from job_bot.cli import cmd_run
from job_bot.config import Settings
from job_bot.llm.base import LLMProvider
from job_bot.models.schemas import ApplicationAnswer, CoverLetter, JobMatchScore, TailoredResume
from job_bot.safety.rate_limiter import DailyCapReached
from job_bot.tracker.db import Tracker

JOB = JobPosting(
    job_id="job1", title="Backend Engineer", company="Acme Corp", url="https://x/1", description=""
)
JOB2 = JobPosting(
    job_id="job2", title="Platform Engineer", company="Acme Corp", url="https://x/2", description=""
)
JOB3 = JobPosting(
    job_id="job3", title="Infra Engineer", company="Acme Corp", url="https://x/3", description=""
)


class FakeProvider(LLMProvider):
    def __init__(self, application_answer: ApplicationAnswer | None = None):
        self.schemas_requested: list[type] = []
        self._application_answer = application_answer or ApplicationAnswer(
            answer="5 years", confidence=0.9, based_on_resume=True
        )

    def generate_structured(self, *, system, prompt, schema):
        self.schemas_requested.append(schema)
        if schema is JobMatchScore:
            return JobMatchScore(
                eligibility="pass",
                technical_fit=90,
                experience_fit=90,
                culture_fit=90,
                score=90,
                reasoning="Great fit",
                should_apply=True,
            )
        if schema is TailoredResume:
            return TailoredResume(
                summary="Tailored summary for Acme.",
                highlighted_skills=["Python"],
                bullet_points=["Shipped feature X"],
            )
        if schema is CoverLetter:
            return CoverLetter(body="Dear Acme, I would love to join your team.")
        if schema is ApplicationAnswer:
            return self._application_answer
        raise AssertionError(f"Unexpected schema requested: {schema}")


class FakeAdapter:
    def __init__(self, page):
        self.page = page
        self.fill_and_submit_calls: list[dict] = []

    def search(self, keywords, location, max_results=25):
        return [JOB]

    def load_description(self, posting):
        return "We need a backend engineer with Python experience."

    def fill_and_submit(self, posting, *, answer_question, resume_path, cover_letter_text, dry_run):
        answered = answer_question("Years of experience?")
        self.fill_and_submit_calls.append(
            {
                "posting": posting,
                "resume_path": resume_path,
                "cover_letter_text": cover_letter_text,
                "dry_run": dry_run,
                "answered": answered,
            }
        )
        return not dry_run


class MultiJobAdapter(FakeAdapter):
    """Same as FakeAdapter but with more than one Easy-Apply result, for
    tests that need to exercise more than one loop iteration of cmd_run.
    """

    def search(self, keywords, location, max_results=25):
        return [JOB, JOB2, JOB3]


class FakePage:
    pass


class FakeContext:
    def new_page(self):
        return FakePage()


@contextmanager
def fake_browser_session(profile_dir, headless=False):
    yield FakeContext()


def make_settings(tmp_path) -> Settings:
    resume_path = tmp_path / "resume.txt"
    resume_path.write_text("Experienced backend engineer skilled in Python.", encoding="utf-8")
    return Settings(
        _env_file=None,
        llm_provider="claude",
        anthropic_api_key="sk-ant-fake",
        resume_path=resume_path,
        faq_path=tmp_path / "faq.json",
        blacklist_path=tmp_path / "blacklist.json",
        db_path=tmp_path / "db.sqlite3",
        browser_profile_dir=tmp_path / "profile",
        audit_log_path=tmp_path / "audit.log",
        applications_dir=tmp_path / "applications",
        require_confirm_before_submit=True,
    )


def make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        keywords="backend",
        location="Remote",
        max_apps=5,
        search_pool=25,
        dry_run=False,
        headless=True,
        provider=None,
        model=None,
        yes_i_understand_the_risk=True,  # skip the interactive confirm() prompt in tests
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_run_generates_and_persists_tailored_resume_and_cover_letter(tmp_path, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args())

    assert TailoredResume in provider.schemas_requested
    assert CoverLetter in provider.schemas_requested

    job_dir = settings.applications_dir / "job1"
    assert (job_dir / "tailored_resume.txt").exists()
    assert "Tailored summary for Acme" in (job_dir / "tailored_resume.txt").read_text()
    assert (job_dir / "cover_letter.txt").read_text() == "Dear Acme, I would love to join your team."


def test_run_submits_and_records_application(tmp_path, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args())

    tracker = Tracker(settings.db_path)
    assert tracker.has_applied("job1") is True


def test_run_dry_run_does_not_mark_applied(tmp_path, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args(dry_run=True))

    tracker = Tracker(settings.db_path)
    assert tracker.has_applied("job1") is False
    # Materials are still generated even in a dry run, so the user can review them.
    assert (settings.applications_dir / "job1" / "cover_letter.txt").exists()


def test_run_uploads_the_original_resume_file_not_a_generated_one(tmp_path, monkeypatch):
    """The tailored resume is a reference artifact only - fill_and_submit
    must still receive the user's own verified resume_path. See
    job_bot/generation/artifacts.py's module docstring for why.
    """
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)

    captured_adapter = {}

    class CapturingAdapter(FakeAdapter):
        def __init__(self, page):
            super().__init__(page)
            captured_adapter["adapter"] = self

    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", CapturingAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args())

    call = captured_adapter["adapter"].fill_and_submit_calls[0]
    assert call["resume_path"] == str(settings.resume_path)


def test_run_caches_high_confidence_answers_to_faq(tmp_path, monkeypatch):
    provider = FakeProvider(
        application_answer=ApplicationAnswer(answer="5 years", confidence=0.9, based_on_resume=True)
    )
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args())

    faq = json.loads(settings.faq_path.read_text())
    assert faq == {"Years of experience?": "5 years"}


def test_run_does_not_cache_low_confidence_answers_to_faq(tmp_path, monkeypatch):
    provider = FakeProvider(
        application_answer=ApplicationAnswer(answer="Not sure", confidence=0.2, based_on_resume=False)
    )
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args())

    assert not settings.faq_path.exists()


class FakeRateLimiterHittingCapOnSecondCall:
    """Simulates the daily cap being reached mid-loop by something other
    than cmd_run's own top-of-loop check - e.g. a second concurrent
    `job-bot run` process racing the same SQLite DB. remaining_today()
    always reports room (like a stale read would), so the loop's own guard
    never trips; only record_application() raises, on its second call.
    """

    def __init__(self, db_path, daily_cap):
        self.record_calls = 0

    def remaining_today(self):
        return 99

    def record_application(self):
        self.record_calls += 1
        if self.record_calls == 2:
            raise DailyCapReached("cap reached by a concurrent process")


class LoadDescriptionFailsForFirstJobAdapter(FakeAdapter):
    """job1's load_description() raises (as e.g. a network hiccup, a
    provider error during scoring, or artifacts.py's UnsafeJobId could) -
    job2 and job3 must still be processed rather than the whole run
    aborting on one bad posting.
    """

    def search(self, keywords, location, max_results=25):
        return [JOB, JOB2, JOB3]

    def load_description(self, posting):
        if posting.job_id == "job1":
            raise RuntimeError("simulated failure loading job1's description")
        return "We need a backend engineer with Python experience."


def test_run_continues_past_a_posting_that_fails_during_prep(tmp_path, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", LoadDescriptionFailsForFirstJobAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args(max_apps=10))

    tracker = Tracker(settings.db_path)
    # job1 never got far enough to be tracked at all (it failed before the
    # upsert_job() call in the scoring step).
    assert tracker.get_job("job1") is None
    # job2 and job3 were still processed and applied to.
    assert tracker.has_applied("job2") is True
    assert tracker.has_applied("job3") is True


def test_run_still_marks_applied_when_rate_limiter_raises_after_a_real_submission(tmp_path, monkeypatch):
    """A submission the browser already clicked through must be recorded in
    the tracker even if record_application() then raises - otherwise the
    job would look un-applied and a future run could apply to it again for
    real. See the comment above tracker.mark_applied() in cli.cmd_run.
    """
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", MultiJobAdapter)
    monkeypatch.setattr("job_bot.cli.RateLimiter", FakeRateLimiterHittingCapOnSecondCall)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args(max_apps=10))  # would process all 3 jobs if not stopped by the cap

    tracker = Tracker(settings.db_path)
    # job1's record_application() succeeded (call #1); job2's raised on
    # call #2, but mark_applied() for job2 must still have run first.
    assert tracker.has_applied("job1") is True
    assert tracker.has_applied("job2") is True
    # The loop must have stopped cleanly after job2 rather than crashing -
    # job3 was never reached.
    assert tracker.get_job("job3") is None
