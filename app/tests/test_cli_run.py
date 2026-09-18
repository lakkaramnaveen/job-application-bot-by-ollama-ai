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
from datetime import date

import pytest

from job_bot.browser.base_adapter import JobPosting
from job_bot.browser.linkedin_adapter import UnansweredRequiredQuestion
from job_bot.cli import cmd_run
from job_bot.config import Settings
from job_bot.llm.base import LLMProvider
from job_bot.models.schemas import ApplicationAnswer, CoverLetter, JobMatchScore, TailoredResume
from job_bot.safety.answer_gaps import AnswerGapStore
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

# Materials for JOB land under <applications_dir>/<today>/<this folder name>/
# - see generation/artifacts.py's _job_dir().
JOB_MATERIALS_DIR_NAME = "job1 - Acme Corp - Backend Engineer"


class FakeProvider(LLMProvider):
    def __init__(self, application_answer: ApplicationAnswer | None = None):
        self.schemas_requested: list[type] = []
        self.tailor_resume_prompts: list[str] = []
        self.application_answer_prompts: list[str] = []
        self.job_match_system_prompts: list[str] = []
        self._application_answer = application_answer or ApplicationAnswer(
            answer="5 years", confidence=0.9, based_on_resume=True
        )

    def generate_structured(self, *, system, prompt, schema):
        self.schemas_requested.append(schema)
        if schema is TailoredResume:
            self.tailor_resume_prompts.append(prompt)
        if schema is ApplicationAnswer:
            self.application_answer_prompts.append(prompt)
        if schema is JobMatchScore:
            self.job_match_system_prompts.append(system)
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

    def search(self, keywords, location, max_results=25, experience_levels=None, include_external=False):
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

    def search(self, keywords, location, max_results=25, experience_levels=None, include_external=False):
        return [JOB, JOB2, JOB3]


class LoopFakeAdapter(FakeAdapter):
    """Returns one brand-new, never-before-seen posting each search() call -
    for --loop tests, since a repeated posting would just get skipped by
    tracker.has_applied() on the second cycle and never exercise the "cap
    reached mid-loop" path search() alone can't reach.
    """

    def __init__(self, page):
        super().__init__(page)
        self.search_calls = 0

    def search(self, keywords, location, max_results=25, experience_levels=None, include_external=False):
        self.search_calls += 1
        job_id = f"loop-job-{self.search_calls}"
        return [JobPosting(job_id=job_id, title="Backend Engineer", company="Acme Corp", url=f"https://x/{job_id}", description="")]


class UnansweredQuestionFakeAdapter(FakeAdapter):
    """fill_and_submit() raises UnansweredRequiredQuestion, as the real
    LinkedInAdapter does for a required text/radio/select field it
    deliberately left unanswered rather than guess.
    """

    def fill_and_submit(self, posting, *, answer_question, resume_path, cover_letter_text, dry_run):
        answer_question("Years of experience?")
        raise UnansweredRequiredQuestion(
            posting.job_id, "Are you comfortable commuting to this job's location?", "reason"
        )


class FakePage:
    def __init__(self):
        self._closed = False

    def is_closed(self):
        return self._closed

    def close(self):
        self._closed = True


class FakeContext:
    def new_page(self):
        return FakePage()


@contextmanager
def fake_browser_session(profile_dir, headless=False, cdp_url=None):
    yield FakeContext()


def make_settings(tmp_path, **overrides) -> Settings:
    resume_path = tmp_path / "resume.txt"
    resume_path.write_text("Experienced backend engineer skilled in Python.", encoding="utf-8")
    kwargs = dict(
        _env_file=None,
        llm_provider="claude",
        anthropic_api_key="sk-ant-fake",
        resume_path=resume_path,
        faq_path=tmp_path / "faq.json",
        blacklist_path=tmp_path / "blacklist.json",
        db_path=tmp_path / "db.sqlite3",
        browser_profile_dir=tmp_path / "profile",
        audit_log_path=tmp_path / "audit.log",
        failed_applications_log_path=tmp_path / "failed_applications.log",
        answer_gaps_path=tmp_path / "answer_gaps.json",
        applications_dir=tmp_path / "applications",
        require_confirm_before_submit=True,
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


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
        min_score=None,
        exclude_title_keywords=None,
        experience_level=None,
        max_years_experience=None,
        require_w2=False,
        loop=False,
        loop_interval_minutes=20,
        include_external_apply=False,
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

    job_dir = settings.applications_dir / date.today().isoformat() / JOB_MATERIALS_DIR_NAME
    assert (job_dir / "tailored_resume.txt").exists()
    assert "Tailored summary for Acme" in (job_dir / "tailored_resume.txt").read_text()
    assert (job_dir / "cover_letter.txt").read_text() == "Dear Acme, I would love to join your team."


def test_run_passes_settings_max_years_experience_and_require_w2_to_the_scorer(tmp_path, monkeypatch):
    """cmd_run must wire Settings.max_years_experience/require_w2 through to
    score_job_match() - the eligibility check itself is scorer.py's
    responsibility (see test_scorer.py), this only guards the wiring gap
    that would otherwise silently leave the setting inert.
    """
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path, max_years_experience=6, require_w2=True)
    cmd_run(settings, make_args())

    assert len(provider.job_match_system_prompts) == 1
    system = provider.job_match_system_prompts[0]
    assert "more than 6 years" in system
    assert "Corp-to-Corp" in system


def test_run_max_years_experience_flag_overrides_settings(tmp_path, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path, max_years_experience=6)
    cmd_run(settings, make_args(max_years_experience=3))

    system = provider.job_match_system_prompts[0]
    assert "more than 3 years" in system


def test_run_require_w2_flag_turns_the_check_on_even_when_settings_has_it_off(tmp_path, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path, require_w2=False)
    cmd_run(settings, make_args(require_w2=True))

    system = provider.job_match_system_prompts[0]
    assert "Corp-to-Corp" in system


def test_run_max_years_experience_setting_used_when_flag_not_given(tmp_path, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path, max_years_experience=None, require_w2=False)
    cmd_run(settings, make_args())  # neither flag passed on the CLI

    system = provider.job_match_system_prompts[0]
    assert "Seniority" not in system
    assert "Corp-to-Corp" not in system


def test_run_records_the_resume_generation_for_future_reuse(tmp_path, monkeypatch):
    """cmd_run must log each tailor_resume() output to the tracker, or
    best_resume_examples() (fed back as few-shot context on the next run -
    see resume_tailor.py's tailor_resume() docstring) would silently never
    see anything - the same wiring-gap risk this file's own module
    docstring calls out for tailored-resume generation itself.
    """
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args())

    tracker = Tracker(settings.db_path)
    examples = tracker.best_resume_examples()
    assert len(examples) == 1
    assert examples[0]["job_id"] == "job1"
    assert examples[0]["summary"] == "Tailored summary for Acme."


def test_run_feeds_a_past_resume_example_into_the_next_tailor_call(tmp_path, monkeypatch):
    """A resume generation recorded by an earlier run must actually reach
    the model as a few-shot example on a later one - proving the loop
    genuinely closes end-to-end, not just that the recording half works.
    """
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path)
    Tracker(settings.db_path).record_resume_generation(
        "earlier-job",
        "Backend Engineer",
        "Other Co",
        "A summary from a resume that landed an interview.",
        ["Python"],
        ["Shipped an earlier feature."],
    )

    cmd_run(settings, make_args())

    assert len(provider.tailor_resume_prompts) == 1
    assert "A summary from a resume that landed an interview." in provider.tailor_resume_prompts[0]


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
    job_dir = settings.applications_dir / date.today().isoformat() / JOB_MATERIALS_DIR_NAME
    assert (job_dir / "cover_letter.txt").exists()


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


def test_run_min_score_skips_a_posting_the_model_said_yes_to(tmp_path, monkeypatch):
    """FakeProvider's JobMatchScore always has should_apply=True, score=90 -
    --min-score is an extra floor on top of that verdict, not a replacement
    for it, so a floor above 90 must still skip the posting even though the
    model itself said apply.
    """
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args(min_score=95))

    tracker = Tracker(settings.db_path)
    job = tracker.get_job("job1")
    assert job["status"] == "skipped"
    assert job["match_score"] == 90  # the real score is still recorded, just not acted on
    assert tracker.has_applied("job1") is False


def test_run_min_score_setting_used_when_flag_not_given(tmp_path, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path)
    settings.min_match_score = 95
    cmd_run(settings, make_args())  # min_score not passed on the CLI

    tracker = Tracker(settings.db_path)
    assert tracker.get_job("job1")["status"] == "skipped"


def test_run_reapplies_a_raised_min_score_to_an_already_seen_job(tmp_path, monkeypatch):
    """A job scored and marked "seen" under a looser (or absent) min_score
    must not slip through forever once the floor is raised in a later run -
    unlike the model's own should_apply verdict, min_score is user config
    that can change between runs, so the reused-score path has to re-check
    it against today's floor rather than trusting the old "seen" status.
    """
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    # Simulates an earlier run under a looser floor: scored 90 (matches
    # FakeProvider's JobMatchScore) and marked seen/worth-applying.
    tracker.record_score(JOB.job_id, JOB.title, JOB.company, JOB.url, score=90, should_apply=True)

    cmd_run(settings, make_args(min_score=95))

    assert JobMatchScore not in provider.schemas_requested  # reused the score, didn't re-score
    job = tracker.get_job(JOB.job_id)
    assert job["status"] == "skipped"
    assert job["match_score"] == 90  # the real score is preserved, just no longer acted on
    assert tracker.has_applied(JOB.job_id) is False


def test_run_still_reuses_a_seen_job_that_clears_a_raised_min_score(tmp_path, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score(JOB.job_id, JOB.title, JOB.company, JOB.url, score=90, should_apply=True)

    cmd_run(settings, make_args(min_score=80))

    assert JobMatchScore not in provider.schemas_requested
    assert tracker.has_applied(JOB.job_id) is True


def test_run_below_default_min_score_of_zero_still_applies(tmp_path, monkeypatch):
    """The default (0) must not change existing behavior: should_apply alone
    still decides, since any real score clears a floor of 0.
    """
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args())

    tracker = Tracker(settings.db_path)
    assert tracker.has_applied("job1") is True


EXTERNAL_JOB = JobPosting(
    job_id="job1",
    title="Backend Engineer",
    company="Acme Corp",
    url="https://x/1",
    description="",
    easy_apply=False,
)


class FakeExternalPage:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class ExternalApplyFakeAdapter(FakeAdapter):
    """Same shape as FakeAdapter, but the one posting returned is
    easy_apply=False - exercises cmd_run's external-apply routing
    (open_external_application() instead of calling fill_and_submit()
    directly) rather than the ordinary Easy Apply path.
    """

    def __init__(self, page):
        super().__init__(page)
        self.external_page = FakeExternalPage()
        self.opened_for: list[str] = []

    def search(self, keywords, location, max_results=25, experience_levels=None, include_external=False):
        return [EXTERNAL_JOB]

    def open_external_application(self, posting):
        self.opened_for.append(posting.job_id)
        return self.external_page


class ExternalApplyMissingButtonAdapter(ExternalApplyFakeAdapter):
    """open_external_application() returns None - the posting turned out
    not to have the external-apply button after all (see the real
    LinkedInAdapter's docstring for when that happens).
    """

    def open_external_application(self, posting):
        self.opened_for.append(posting.job_id)
        return None


class FakeExternalApplyAdapter:
    """Stands in for job_bot.browser.external_apply_adapter.ExternalApplyAdapter."""

    instances: list["FakeExternalApplyAdapter"] = []

    def __init__(self, page):
        self.page = page
        self.fill_and_submit_calls: list[dict] = []
        FakeExternalApplyAdapter.instances.append(self)

    def fill_and_submit(self, *, answer_question, resume_path, cover_letter_text, dry_run):
        answered = answer_question("Years of experience?")
        self.fill_and_submit_calls.append(
            {
                "resume_path": resume_path,
                "cover_letter_text": cover_letter_text,
                "dry_run": dry_run,
                "answered": answered,
            }
        )
        return not dry_run


class FailingFakeExternalApplyAdapter(FakeExternalApplyAdapter):
    def fill_and_submit(self, *, answer_question, resume_path, cover_letter_text, dry_run):
        raise RuntimeError("This application requires solving a CAPTCHA - job-bot never attempts this.")


@pytest.fixture(autouse=True)
def _reset_fake_external_apply_adapter_instances():
    FakeExternalApplyAdapter.instances = []
    yield
    FakeExternalApplyAdapter.instances = []


def test_run_applies_via_external_apply_adapter_for_a_non_easy_apply_posting(tmp_path, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", ExternalApplyFakeAdapter)
    monkeypatch.setattr("job_bot.cli.ExternalApplyAdapter", FakeExternalApplyAdapter)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")  # confirm the external-apply prompt

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args(yes_i_understand_the_risk=True, include_external_apply=True))

    assert len(FakeExternalApplyAdapter.instances) == 1
    adapter = FakeExternalApplyAdapter.instances[0]
    assert adapter.fill_and_submit_calls[0]["dry_run"] is False
    tracker = Tracker(settings.db_path)
    assert tracker.has_applied(EXTERNAL_JOB.job_id) is True


def test_run_closes_the_external_page_once_handled(tmp_path, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    linkedin_adapter = ExternalApplyFakeAdapter
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", linkedin_adapter)
    monkeypatch.setattr("job_bot.cli.ExternalApplyAdapter", FakeExternalApplyAdapter)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    captured_adapter = {}
    real_init = linkedin_adapter.__init__

    def capturing_init(self, page):
        real_init(self, page)
        captured_adapter["adapter"] = self

    monkeypatch.setattr(linkedin_adapter, "__init__", capturing_init)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args(yes_i_understand_the_risk=True, include_external_apply=True))

    assert captured_adapter["adapter"].external_page.closed is True


def test_run_always_confirms_external_apply_even_with_yes_i_understand_the_risk(tmp_path, monkeypatch):
    """external_confirmer ignores --yes-i-understand-the-risk entirely -
    with no stdin attached (as in this test), the confirmation prompt's
    input() call raises EOFError, which SubmitConfirmer treats as declined
    (see safety/confirm.py) rather than assuming yes. That's exactly what
    must happen here: the external posting is skipped, not silently
    applied to without ever really confirming.
    """
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", ExternalApplyFakeAdapter)
    monkeypatch.setattr("job_bot.cli.ExternalApplyAdapter", FakeExternalApplyAdapter)

    def raise_eof(prompt=""):
        raise EOFError  # no terminal attached to answer from, as in a real unattended run

    monkeypatch.setattr("builtins.input", raise_eof)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args(yes_i_understand_the_risk=True, include_external_apply=True))

    assert FakeExternalApplyAdapter.instances == []  # never even reached fill_and_submit
    tracker = Tracker(settings.db_path)
    assert tracker.has_applied(EXTERNAL_JOB.job_id) is False


def test_run_reports_a_clean_error_when_the_external_apply_button_is_gone(tmp_path, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", ExternalApplyMissingButtonAdapter)
    monkeypatch.setattr("job_bot.cli.ExternalApplyAdapter", FakeExternalApplyAdapter)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args(yes_i_understand_the_risk=True, include_external_apply=True))

    assert FakeExternalApplyAdapter.instances == []
    tracker = Tracker(settings.db_path)
    assert tracker.has_applied(EXTERNAL_JOB.job_id) is False


def test_run_closes_the_external_page_even_when_fill_and_submit_raises(tmp_path, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    linkedin_adapter = ExternalApplyFakeAdapter
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", linkedin_adapter)
    monkeypatch.setattr("job_bot.cli.ExternalApplyAdapter", FailingFakeExternalApplyAdapter)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    captured_adapter = {}
    real_init = linkedin_adapter.__init__

    def capturing_init(self, page):
        real_init(self, page)
        captured_adapter["adapter"] = self

    monkeypatch.setattr(linkedin_adapter, "__init__", capturing_init)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args(yes_i_understand_the_risk=True, include_external_apply=True))

    assert captured_adapter["adapter"].external_page.closed is True
    entries = [json.loads(line) for line in settings.failed_applications_log_path.read_text().splitlines()]
    assert "CAPTCHA" in entries[0]["details"]["error"]


def test_run_skips_a_posting_whose_title_matches_an_exclude_keyword(tmp_path, monkeypatch):
    """JOB2's title is "Platform Engineer" - excluding "platform" must skip
    it before it's ever scored (job2 never even gets tracked), while JOB and
    JOB3 (unaffected titles) are processed normally.
    """
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", MultiJobAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args(max_apps=10, exclude_title_keywords="platform"))

    tracker = Tracker(settings.db_path)
    assert tracker.get_job("job2") is None
    assert tracker.has_applied("job1") is True
    assert tracker.has_applied("job3") is True


def test_run_exclude_keyword_match_is_case_insensitive(tmp_path, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", MultiJobAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args(max_apps=10, exclude_title_keywords="PLATFORM"))

    tracker = Tracker(settings.db_path)
    assert tracker.get_job("job2") is None


def test_run_rejects_an_unknown_experience_level_before_opening_a_browser(tmp_path, monkeypatch, capsys):
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    browser_session_calls = []
    monkeypatch.setattr(
        "job_bot.cli.browser_session",
        lambda *a, **k: browser_session_calls.append(1) or fake_browser_session(*a, **k),
    )

    settings = make_settings(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        cmd_run(settings, make_args(experience_level="mid-senior,not-a-real-level"))

    assert exc_info.value.code == 1
    assert "not-a-real-level" in capsys.readouterr().err
    assert browser_session_calls == []  # never got as far as opening a browser


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

    def search(self, keywords, location, max_results=25, experience_levels=None, include_external=False):
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

    # A dedicated, focused log of just what needs fixing - not the full
    # audit log's interleaved search/scored/applied noise - records the
    # failure with enough context to act on it without re-running anything.
    entries = [json.loads(line) for line in settings.failed_applications_log_path.read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["action"] == "prep_error"
    assert entries[0]["details"]["job_id"] == "job1"
    assert entries[0]["details"]["title"] == JOB.title
    assert entries[0]["details"]["company"] == JOB.company
    assert "simulated failure loading job1's description" in entries[0]["details"]["error"]


def test_run_prints_a_summary_line_pointing_at_the_failure_log(tmp_path, monkeypatch, capsys):
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", LoadDescriptionFailsForFirstJobAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args(max_apps=10))

    out = capsys.readouterr().out
    assert "1 posting(s) could not be completed" in out
    assert str(settings.failed_applications_log_path) in out


def test_run_with_no_failures_writes_nothing_to_the_failure_log(tmp_path, monkeypatch, capsys):
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args())

    assert not settings.failed_applications_log_path.exists()
    assert "could not be completed" not in capsys.readouterr().out


class PrepClosesTheBrowserForFirstJobAdapter(FakeAdapter):
    """job1's load_description() fails because the browser itself died mid
    -call (the window was closed, crashed, or the process was killed) -
    distinct from LoadDescriptionFailsForFirstJobAdapter's ordinary failure
    above. Every remaining posting shares the same page, so the whole run
    must stop rather than churn through job2/job3 against a dead page.
    """

    def search(self, keywords, location, max_results=25, experience_levels=None, include_external=False):
        return [JOB, JOB2, JOB3]

    def load_description(self, posting):
        if posting.job_id == "job1":
            self.page.close()
            raise RuntimeError("Target page, context or browser has been closed")
        return "We need a backend engineer with Python experience."


def test_run_stops_the_whole_run_when_the_browser_closes_during_prep(tmp_path, monkeypatch, capsys):
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", PrepClosesTheBrowserForFirstJobAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args(max_apps=10))

    tracker = Tracker(settings.db_path)
    assert tracker.get_job("job1") is None
    # job2 and job3 were never reached at all - the run stopped as soon as
    # it noticed the browser was gone, rather than repeating the same
    # failure for the rest of the search pool.
    assert tracker.get_job("job2") is None
    assert tracker.get_job("job3") is None
    assert "Browser window was closed" in capsys.readouterr().out


class ApplyClosesTheBrowserForFirstJobAdapter(FakeAdapter):
    """Same idea as PrepClosesTheBrowserForFirstJobAdapter, but the browser
    dies during fill_and_submit() (the apply step) instead of during prep -
    the two are separate except blocks in cmd_run, each needing its own
    is_closed() check.
    """

    def search(self, keywords, location, max_results=25, experience_levels=None, include_external=False):
        return [JOB, JOB2, JOB3]

    def fill_and_submit(self, posting, *, answer_question, resume_path, cover_letter_text, dry_run):
        if posting.job_id == "job1":
            self.page.close()
            raise RuntimeError("Target page, context or browser has been closed")
        return super().fill_and_submit(
            posting,
            answer_question=answer_question,
            resume_path=resume_path,
            cover_letter_text=cover_letter_text,
            dry_run=dry_run,
        )


def test_run_stops_the_whole_run_when_the_browser_closes_during_apply(tmp_path, monkeypatch, capsys):
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", ApplyClosesTheBrowserForFirstJobAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args(max_apps=10))

    tracker = Tracker(settings.db_path)
    # job1 was scored (worth applying to) before the browser died, but the
    # submission itself never went through.
    assert tracker.has_applied("job1") is False
    assert tracker.get_job("job2") is None
    assert tracker.get_job("job3") is None
    assert "Browser window was closed" in capsys.readouterr().out


def test_run_reuses_an_earlier_runs_score_instead_of_rescoring(tmp_path, monkeypatch):
    """Simulates resuming after a crash: JOB was already scored (and judged
    worth applying to) by an earlier `job-bot run` that didn't get as far as
    submitting. This run must apply to it without spending another
    JobMatchScore call re-judging it.
    """
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score(JOB.job_id, JOB.title, JOB.company, JOB.url, score=90, should_apply=True)

    cmd_run(settings, make_args())

    assert JobMatchScore not in provider.schemas_requested
    assert tracker.has_applied(JOB.job_id) is True


def test_run_skips_a_posting_already_skipped_in_an_earlier_run(tmp_path, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score(JOB.job_id, JOB.title, JOB.company, JOB.url, score=20, should_apply=False)

    cmd_run(settings, make_args())

    assert provider.schemas_requested == []
    assert tracker.get_job(JOB.job_id)["status"] == "skipped"


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


def test_loop_runs_multiple_cycles_and_stops_once_the_daily_cap_is_reached(tmp_path, monkeypatch):
    """--loop must keep re-searching (picking up newly-posted jobs each
    cycle, via search_calls growing) rather than stopping after the first
    batch like a plain run does - but it's still bounded by the real daily
    application cap, not truly infinite.
    """
    provider = FakeProvider()
    adapter = LoopFakeAdapter(page=None)
    sleep_calls: list[float] = []
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", lambda page: adapter)
    monkeypatch.setattr("job_bot.cli.time.sleep", lambda seconds: sleep_calls.append(seconds))

    settings = make_settings(tmp_path, daily_application_cap=2)
    cmd_run(settings, make_args(loop=True, loop_interval_minutes=7, max_apps=1))

    tracker = Tracker(settings.db_path)
    assert tracker.has_applied("loop-job-1") is True
    assert tracker.has_applied("loop-job-2") is True
    # A third cycle never happened - the cap was reached after cycle 2.
    assert adapter.search_calls == 2
    assert tracker.get_job("loop-job-3") is None
    # Slept once, between cycle 1 and cycle 2 - not before cycle 1, and not
    # again after cycle 2 since the cap check stops the loop first.
    assert sleep_calls == [7 * 60]


def test_run_quits_ollama_once_the_daily_cap_is_reached_when_configured(tmp_path, monkeypatch):
    """Settings.quit_ollama_when_done must actually reach cmd_run - proving
    the wiring, not just that quit_ollama() itself works (see
    test_ollama_provider.py for that).
    """
    provider = FakeProvider()
    quit_calls: list[None] = []
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)
    monkeypatch.setattr("job_bot.cli.quit_ollama", lambda: quit_calls.append(None) or True)

    settings = make_settings(
        tmp_path, llm_provider="ollama", anthropic_api_key=None, daily_application_cap=1, quit_ollama_when_done=True
    )
    cmd_run(settings, make_args())

    assert quit_calls == [None]


def test_run_does_not_quit_ollama_when_the_setting_is_off(tmp_path, monkeypatch):
    provider = FakeProvider()
    quit_calls: list[None] = []
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)
    monkeypatch.setattr("job_bot.cli.quit_ollama", lambda: quit_calls.append(None) or True)

    settings = make_settings(
        tmp_path, llm_provider="ollama", anthropic_api_key=None, daily_application_cap=1, quit_ollama_when_done=False
    )
    cmd_run(settings, make_args())

    assert quit_calls == []


def test_run_does_not_quit_ollama_for_the_claude_provider_even_if_configured(tmp_path, monkeypatch):
    """quit_ollama_when_done is meaningless with LLM_PROVIDER=claude - must
    stay inert rather than kill an Ollama the run never even used.
    """
    provider = FakeProvider()
    quit_calls: list[None] = []
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)
    monkeypatch.setattr("job_bot.cli.quit_ollama", lambda: quit_calls.append(None) or True)

    settings = make_settings(tmp_path, daily_application_cap=1, quit_ollama_when_done=True)
    cmd_run(settings, make_args())

    assert quit_calls == []


def test_run_does_not_quit_ollama_when_the_daily_cap_is_not_yet_reached(tmp_path, monkeypatch):
    provider = FakeProvider()
    quit_calls: list[None] = []
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)
    monkeypatch.setattr("job_bot.cli.quit_ollama", lambda: quit_calls.append(None) or True)

    settings = make_settings(
        tmp_path,
        llm_provider="ollama",
        anthropic_api_key=None,
        daily_application_cap=100,
        quit_ollama_when_done=True,
    )
    cmd_run(settings, make_args())

    assert quit_calls == []


def test_loop_quits_ollama_once_the_daily_cap_is_reached_when_configured(tmp_path, monkeypatch):
    provider = FakeProvider()
    adapter = LoopFakeAdapter(page=None)
    quit_calls: list[None] = []
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", lambda page: adapter)
    monkeypatch.setattr("job_bot.cli.time.sleep", lambda seconds: None)
    monkeypatch.setattr("job_bot.cli.quit_ollama", lambda: quit_calls.append(None) or True)

    settings = make_settings(
        tmp_path,
        llm_provider="ollama",
        anthropic_api_key=None,
        daily_application_cap=2,
        quit_ollama_when_done=True,
    )
    cmd_run(settings, make_args(loop=True, max_apps=1))

    assert quit_calls == [None]


def test_loop_stops_cleanly_on_keyboard_interrupt(tmp_path, monkeypatch):
    provider = FakeProvider()
    adapter = LoopFakeAdapter(page=None)

    def raise_interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", lambda page: adapter)
    monkeypatch.setattr("job_bot.cli.time.sleep", raise_interrupt)

    settings = make_settings(tmp_path, daily_application_cap=100)
    # Must not raise - a Ctrl+C mid-loop is a normal, expected way to stop.
    cmd_run(settings, make_args(loop=True, max_apps=1))

    tracker = Tracker(settings.db_path)
    assert tracker.has_applied("loop-job-1") is True


def test_run_records_an_unanswered_required_question_as_an_answer_gap(tmp_path, monkeypatch):
    """The whole point of the "learn from its mistakes" loop: a required
    question the LLM couldn't confidently answer must be recorded to
    answer_gaps_path (not just logged as a generic apply_error), or
    `job-bot review-answers` would have nothing to show the user and the
    same question would keep failing the same way on every future posting.
    """
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", UnansweredQuestionFakeAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args())

    gaps = AnswerGapStore(settings.answer_gaps_path).list_unanswered()
    question = "Are you comfortable commuting to this job's location?"
    assert question in gaps
    assert gaps[question]["example_job_id"] == JOB.job_id
    assert gaps[question]["example_company"] == JOB.company


def test_run_feeds_past_qa_answers_into_the_next_question(tmp_path, monkeypatch):
    """Real wiring gap this guards against: recent_qa_pairs() existing in
    isolation proves nothing if cmd_run never actually calls it - this is
    exactly the kind of test that would have caught tailor_resume() being
    generated but never invoked (see this file's own module docstring).
    Every past answer, not just the curated FAQ subset, must reach the
    prompt for the next question asked - that's the whole point of
    recent_qa_pairs() over relying on FAQ_PATH alone.
    """
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path)
    Tracker(settings.db_path).record_qa("earlier-job", "Willing to relocate?", "No")

    cmd_run(settings, make_args())

    assert len(provider.application_answer_prompts) == 1
    prompt = provider.application_answer_prompts[0]
    assert "Willing to relocate?" in prompt
    assert "No" in prompt
