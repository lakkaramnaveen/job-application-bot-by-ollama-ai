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
from pathlib import Path

import pytest

from job_bot.browser.base_adapter import JobPosting
from job_bot.browser.linkedin_adapter import LinkedInSignedOut, UnansweredRequiredQuestion
from job_bot.cli import cmd_run
from job_bot.config import Settings
from job_bot.llm.base import LLMProvider
from job_bot.llm.claude_provider import ClaudeProviderError
from job_bot.llm.ollama_provider import OllamaProviderError
from job_bot.models.schemas import ApplicationAnswer, CoverLetter, JobMatchScore, TailoredResume
from job_bot.safety.answer_gaps import AnswerGapStore
from job_bot.safety.audit_log import AuditLogger
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


def test_run_prints_every_non_blocking_settings_warning(tmp_path, monkeypatch, capsys):
    """validate_ready() itself is unit-tested in test_config.py to return
    the right warning text (e.g. a DAILY_APPLICATION_CAP above the hard
    ceiling) - this covers the other half, that cmd_run actually prints
    what it returns rather than silently discarding it.
    """
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path, daily_application_cap=500)
    cmd_run(settings, make_args())

    assert "Warning: DAILY_APPLICATION_CAP=500 exceeds the hard ceiling" in capsys.readouterr().out


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


def test_run_persists_the_scorers_reasoning_to_the_tracker(tmp_path, monkeypatch):
    """cmd_run must actually thread JobMatchScore.reasoning and the
    eligibility verdict/note through to Tracker.record_score() - the
    fields existing on the schema, or record_score() knowing how to store
    them (see test_tracker.py), proves neither the LLM's answer nor the
    storage works end to end without this wiring, the same class of gap
    test_run_passes_settings_max_years_experience_and_require_w2_to_the_
    scorer above guards for a different field.
    """
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args())

    job = Tracker(settings.db_path).get_job(JOB.job_id)
    assert job["match_reasoning"] == "Great fit"
    assert job["eligibility"] == "pass"


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


def test_run_uploads_the_original_resume_file_when_a_tailored_docx_cant_be_built(tmp_path, monkeypatch):
    """make_settings()'s resume fixture is a single plain sentence with no
    SUMMARY/SKILLS/EXPERIENCE section headers - build_tailored_resume_docx()
    can't confidently locate them, so write_tailored_resume_docx() returns
    None and generate_materials() must fall back to the user's own
    unmodified resume_path rather than upload nothing or guess. See
    test_run_uploads_a_freshly_tailored_docx_resume_when_one_can_be_built()
    for the case where a tailored .docx is built and used instead.
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


def test_run_uploads_a_freshly_tailored_docx_resume_when_one_can_be_built(tmp_path, monkeypatch):
    """When resume_text has clearly-labeled SUMMARY/SKILLS/EXPERIENCE
    sections, generate_materials() must build and use a per-job tailored
    .docx instead of falling back to the static resume_path - this is the
    user-requested behavior change (the actual uploaded resume now varies
    by job description, unlike test_run_uploads_the_original_resume_file_
    when_a_tailored_docx_cant_be_built()'s no-structure-detected case).
    """
    structured_resume_path = tmp_path / "structured_resume.txt"
    structured_resume_path.write_text(
        "Jane Doe\njane@example.com\n\n"
        "SUMMARY\nExperienced backend engineer skilled in Python.\n\n"
        "SKILLS\nPython, Django, PostgreSQL\n\n"
        "EXPERIENCE\nSoftware Engineer, Acme Corp, 2020-Present\n- Built things.\n",
        encoding="utf-8",
    )
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path, resume_path=structured_resume_path)
    adapter_instances = []
    original_fake_adapter = FakeAdapter

    class CapturingAdapter(original_fake_adapter):
        def __init__(self, page):
            super().__init__(page)
            adapter_instances.append(self)

    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", CapturingAdapter)

    cmd_run(settings, make_args())

    resume_path_used = adapter_instances[0].fill_and_submit_calls[0]["resume_path"]
    assert resume_path_used.endswith(".docx")
    assert resume_path_used != str(structured_resume_path)
    assert Path(resume_path_used).exists()


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


def test_run_treats_an_answer_that_echoes_the_question_as_no_answer(tmp_path, monkeypatch):
    """Real bug (data/faq_answers.json): for a bare field label like "Phone
    country code", the model returned the label itself, which was then
    recorded, cached to FAQ_PATH at high confidence, and replayed on every
    later posting - never matching any option. It must come back as ""
    (unanswered, so the adapter's own answer-gap flow takes over) and be
    neither recorded nor cached.
    """
    provider = FakeProvider(ApplicationAnswer(answer="years of experience", confidence=0.95, based_on_resume=True))
    adapter = FakeAdapter(page=None)
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", lambda page: adapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args())

    assert adapter.fill_and_submit_calls[0]["answered"] == ""
    assert not settings.faq_path.exists()
    assert Tracker(settings.db_path).recent_qa_pairs() == []


def test_run_still_applies_when_the_faq_file_is_unreadable_and_leaves_it_untouched(tmp_path, monkeypatch, capsys):
    """Caching an answer is an optimization - an unreadable FAQ_PATH must
    neither be overwritten (losing its contents) nor fail the application.
    """
    provider = FakeProvider()
    adapter = FakeAdapter(page=None)
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", lambda page: adapter)

    settings = make_settings(tmp_path)
    settings.faq_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt = b'{"Willing to relocate?": "No"'
    settings.faq_path.write_bytes(corrupt)

    cmd_run(settings, make_args())

    assert adapter.fill_and_submit_calls[0]["answered"] == "5 years"
    assert settings.faq_path.read_bytes() == corrupt
    assert "Warning: answer not cached" in capsys.readouterr().out


def test_run_refuses_to_start_when_the_blacklist_file_is_unreadable(tmp_path, monkeypatch, capsys):
    """An unreadable blacklist loads as empty, which for a run would mean
    silently applying to every company the user blocked - the run must stop
    before searching anything, with the file left untouched.
    """
    adapter = FakeAdapter(page=None)
    searched = []
    adapter.search = lambda *a, **kw: searched.append(True) or [JOB]
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: FakeProvider())
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", lambda page: adapter)

    settings = make_settings(tmp_path)
    settings.blacklist_path.parent.mkdir(parents=True, exist_ok=True)
    settings.blacklist_path.write_bytes(b'["Acme"')

    with pytest.raises(SystemExit) as exc_info:
        cmd_run(settings, make_args())

    assert exc_info.value.code == 1
    assert "no company would be blocked this run" in capsys.readouterr().err
    assert searched == []
    assert adapter.fill_and_submit_calls == []
    assert settings.blacklist_path.read_bytes() == b'["Acme"'


def test_run_refuses_cleanly_when_the_blacklist_path_is_a_directory(tmp_path, monkeypatch, capsys):
    """An OSError, not just bad JSON - must hit the same clean refusal, not
    a raw IsADirectoryError traceback.
    """
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: FakeProvider())
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path)
    settings.blacklist_path.mkdir(parents=True)

    with pytest.raises(SystemExit) as exc_info:
        cmd_run(settings, make_args())

    assert exc_info.value.code == 1
    assert "no company would be blocked this run" in capsys.readouterr().err


def test_run_still_answers_questions_when_the_faq_path_is_unreadable(tmp_path, monkeypatch, capsys):
    """faq_answers() is called for every question - an OSError there used
    to fail every posting that asked one.
    """
    adapter = FakeAdapter(page=None)
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: FakeProvider())
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", lambda page: adapter)

    settings = make_settings(tmp_path)
    settings.faq_path.mkdir(parents=True)

    cmd_run(settings, make_args())

    assert adapter.fill_and_submit_calls[0]["answered"] == "5 years"
    assert "Warning: answer not cached" in capsys.readouterr().out


@pytest.mark.parametrize("which", ["audit_log_path", "failed_applications_log_path"])
def test_run_refuses_to_start_when_a_log_it_writes_to_is_unwritable(tmp_path, monkeypatch, capsys, which):
    """The first audit write happens outside any try/except - an unwritable
    log used to crash the run with a raw IsADirectoryError traceback. Must
    stop cleanly before searching, naming the file.
    """
    adapter = FakeAdapter(page=None)
    searched = []
    adapter.search = lambda *a, **kw: searched.append(True) or [JOB]
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: FakeProvider())
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", lambda page: adapter)

    settings = make_settings(tmp_path)
    getattr(settings, which).mkdir(parents=True)  # a directory where the log file should be

    with pytest.raises(SystemExit) as exc_info:
        cmd_run(settings, make_args())

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "refusing to start a run" in err
    assert str(getattr(settings, which)) in err
    assert searched == []


def test_run_starts_normally_with_no_blacklist_file_at_all(tmp_path, monkeypatch):
    """A fresh install has no blacklist file - that's not corruption."""
    adapter = FakeAdapter(page=None)
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: FakeProvider())
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", lambda page: adapter)

    settings = make_settings(tmp_path)
    assert not settings.blacklist_path.exists()

    cmd_run(settings, make_args())

    assert len(adapter.fill_and_submit_calls) == 1


class WorkHistoryDateAdapter(FakeAdapter):
    def fill_and_submit(self, posting, *, answer_question, resume_path, cover_letter_text, dry_run):
        self.fill_and_submit_calls.append({"answered": answer_question("Year of From")})
        return not dry_run


def test_run_leaves_a_per_position_date_field_unanswered_without_asking_the_llm(tmp_path, monkeypatch):
    """The LLM sees only the bare label, so it can't know which position is
    meant, and even a cached FAQ answer would be the same date for every
    position - "" without an LLM call, recorded nowhere.
    """
    provider = FakeProvider(ApplicationAnswer(answer="2019", confidence=0.95, based_on_resume=True))
    adapter = WorkHistoryDateAdapter(page=None)
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", lambda page: adapter)

    settings = make_settings(tmp_path)
    settings.faq_path.parent.mkdir(parents=True, exist_ok=True)
    settings.faq_path.write_text(json.dumps({"Year of From": "2019"}), encoding="utf-8")

    cmd_run(settings, make_args())

    assert adapter.fill_and_submit_calls[0]["answered"] == ""
    assert ApplicationAnswer not in provider.schemas_requested
    assert Tracker(settings.db_path).recent_qa_pairs() == []


class UnanswerableAdapter(FakeAdapter):
    def fill_and_submit(self, posting, *, answer_question, resume_path, cover_letter_text, dry_run):
        raise UnansweredRequiredQuestion(posting.job_id, "Security clearance level?", "No answer")


def test_run_warns_but_keeps_going_when_the_answer_gaps_file_is_unreadable(tmp_path, monkeypatch, capsys):
    """An unreadable answer_gaps.json refuses the write (a8febaa) - the run
    must report the posting's failure normally and warn, not crash, and the
    file must be left as it was.
    """
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: FakeProvider())
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", UnanswerableAdapter)

    settings = make_settings(tmp_path)
    settings.answer_gaps_path.parent.mkdir(parents=True, exist_ok=True)
    settings.answer_gaps_path.write_bytes(b'{"Old question?": {"count": 3}')

    cmd_run(settings, make_args())  # must not raise

    out = capsys.readouterr().out
    assert "Warning: unanswered question not recorded" in out
    assert "Error applying to" in out
    assert settings.answer_gaps_path.read_bytes() == b'{"Old question?": {"count": 3}'


def test_run_ignores_an_echoed_answer_already_cached_in_the_faq(tmp_path, monkeypatch):
    """Entries cached before the echo check existed are still in real users'
    FAQ files - they must fall through to the LLM, not be replayed.
    """
    provider = FakeProvider()
    adapter = FakeAdapter(page=None)
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", lambda page: adapter)

    settings = make_settings(tmp_path)
    settings.faq_path.parent.mkdir(parents=True, exist_ok=True)
    settings.faq_path.write_text(json.dumps({"Years of experience?": "Years of experience"}), encoding="utf-8")

    cmd_run(settings, make_args())

    assert ApplicationAnswer in provider.schemas_requested
    assert adapter.fill_and_submit_calls[0]["answered"] == "5 years"


def test_run_reuses_an_exact_faq_match_without_calling_the_llm(tmp_path, monkeypatch):
    """A question whose exact text is already a FAQ_PATH key is a curated,
    confident, resume-grounded answer (that's the whole promotion bar in
    save_faq_answer()) - asking the LLM to re-derive it is a guaranteed-
    redundant round trip, and a slow one on a local model. This proves
    that exact match short-circuits ApplicationAnswer generation entirely,
    not just that the cached text happens to be returned.
    """
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)

    settings = make_settings(tmp_path)
    settings.faq_path.parent.mkdir(parents=True, exist_ok=True)
    settings.faq_path.write_text(json.dumps({"Years of experience?": "7"}), encoding="utf-8")

    cmd_run(settings, make_args())

    assert ApplicationAnswer not in provider.schemas_requested
    tracker = Tracker(settings.db_path)
    pairs = tracker.recent_qa_pairs()
    assert {"question": "Years of experience?", "answer": "7"} in pairs


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


class OllamaUnreachableProvider(LLMProvider):
    """Simulates Ollama being completely down for the whole run - every
    call raises the exact OllamaProviderError ollama_provider.py raises on
    httpx.ConnectError, not just some generic failure.
    """

    def __init__(self):
        self.calls = 0

    def generate_structured(self, *, system, prompt, schema):
        self.calls += 1
        raise OllamaProviderError(
            "Could not reach Ollama at http://localhost:11434. Is it running? "
            "Try `ollama serve` in another terminal."
        )


def test_run_stops_the_whole_run_when_ollama_is_unreachable(tmp_path, monkeypatch, capsys):
    """Real bug this guards against: with Ollama down, every remaining
    posting otherwise repeats the exact same guaranteed-to-fail LLM call
    and prints an identical error - confirmed live (the same "Could not
    reach Ollama" prep_error 6 times in a row before this fix) instead of
    recognizing after the first failure that nothing downstream can
    succeed either.
    """
    provider = OllamaUnreachableProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", MultiJobAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args(max_apps=10))

    assert provider.calls == 1
    out = capsys.readouterr().out
    assert "Ollama is unreachable" in out
    assert out.count("Error preparing application") == 1
    tracker = Tracker(settings.db_path)
    assert tracker.get_job("job2") is None
    assert tracker.get_job("job3") is None


def test_loop_stops_instead_of_retrying_when_ollama_is_unreachable(tmp_path, monkeypatch, capsys):
    """Real bug this guards against: _run_apply_cycle already stops early
    within *that* cycle when Ollama is unreachable (see
    test_run_stops_the_whole_run_when_ollama_is_unreachable above) - but
    --loop's own outer loop had no way to tell that apart from an ordinary
    "nothing to apply this cycle" one, so applied == 0 looked identical
    either way. Confirmed live: a run with Ollama down correctly stopped
    the first cycle's posting loop early, then --loop printed "Nothing to
    apply to this cycle - sleeping 20 minute(s)..." and retried the
    identical, guaranteed-to-fail cycle every 20 minutes until manually
    interrupted. time.sleep is monkeypatched to fail the test outright if
    called at all - the loop must stop, not sleep-and-retry.
    """
    provider = OllamaUnreachableProvider()

    def fail_if_called(seconds):
        raise AssertionError("must not sleep/retry once the provider is unreachable - the loop should stop instead")

    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", MultiJobAdapter)
    monkeypatch.setattr("job_bot.cli.time.sleep", fail_if_called)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args(loop=True, max_apps=10))  # must not raise

    out = capsys.readouterr().out
    assert "Ollama is unreachable" in out
    assert "Stopping the loop" in out
    assert provider.calls == 1


class ClaudeMisconfiguredProvider(LLMProvider):
    """Simulates a revoked/invalid Claude API key - every call raises the
    exact ClaudeProviderError claude_provider.py raises on
    anthropic.AuthenticationError, not just some generic failure.
    """

    def __init__(self):
        self.calls = 0

    def generate_structured(self, *, system, prompt, schema):
        self.calls += 1
        raise ClaudeProviderError("Invalid ANTHROPIC_API_KEY.")


def test_run_stops_the_whole_run_when_claude_is_misconfigured(tmp_path, monkeypatch, capsys):
    """Same class of bug as test_run_stops_the_whole_run_when_ollama_is_unreachable
    above, for the Claude provider: an invalid/revoked API key fails
    identically on every remaining posting, so the run should recognize
    that after the first failure instead of repeating the identical,
    guaranteed-to-fail LLM call (and an identical printed error) once per
    remaining posting.
    """
    provider = ClaudeMisconfiguredProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", MultiJobAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args(max_apps=10))

    assert provider.calls == 1
    out = capsys.readouterr().out
    assert "Claude provider is misconfigured" in out
    assert out.count("Error preparing application") == 1
    tracker = Tracker(settings.db_path)
    assert tracker.get_job("job2") is None
    assert tracker.get_job("job3") is None


def test_run_stops_processing_further_postings_once_the_daily_cap_is_reached_mid_cycle(
    tmp_path, monkeypatch, capsys
):
    """The daily cap can be hit partway through a single cycle's list of
    postings, not just once per cycle at the loop level (see the --loop
    tests below for that separate check) - --max-apps alone doesn't guard
    this, since it's set well above what the cap allows here. This
    exercises _run_apply_cycle's own per-posting remaining_today() check.
    """
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", MultiJobAdapter)

    settings = make_settings(tmp_path, daily_application_cap=2)
    cmd_run(settings, make_args(max_apps=10))

    tracker = Tracker(settings.db_path)
    assert tracker.has_applied("job1") is True
    assert tracker.has_applied("job2") is True
    assert tracker.get_job("job3") is None
    assert "Daily application cap reached." in capsys.readouterr().out


def test_run_stops_processing_further_postings_once_max_apps_is_reached_mid_cycle(tmp_path, monkeypatch):
    """--max-apps can also be hit partway through a single cycle's list of
    postings, with more still left in the pool - every other test setting
    max_apps this low (search-results size 1) never leaves a posting
    unprocessed behind the limit, so this is the shape needed to actually
    reach the loop's own `applied >= args.max_apps` check with postings
    still remaining, rather than the for-loop simply running out on its
    own at the same moment.
    """
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", MultiJobAdapter)

    settings = make_settings(tmp_path, daily_application_cap=100)
    cmd_run(settings, make_args(max_apps=2))

    tracker = Tracker(settings.db_path)
    assert tracker.has_applied("job1") is True
    assert tracker.has_applied("job2") is True
    assert tracker.get_job("job3") is None


class OllamaUnreachableDuringApplyProvider(LLMProvider):
    """Scoring and tailoring succeed normally - only answering a form
    question (which happens inside apply_to(), during the *apply* phase)
    raises. This is the shape needed to reach the apply phase's own
    is-Ollama-unreachable check (cli.py's apply_to() except block),
    distinct from the prep-phase check already covered by
    test_run_stops_the_whole_run_when_ollama_is_unreachable above, which
    only ever fails during scoring/tailoring.
    """

    def __init__(self):
        self.answer_calls = 0

    def generate_structured(self, *, system, prompt, schema):
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
            self.answer_calls += 1
            raise OllamaProviderError(
                "Could not reach Ollama at http://localhost:11434. Is it running? "
                "Try `ollama serve` in another terminal."
            )
        raise AssertionError(f"Unexpected schema requested: {schema}")


def test_run_stops_the_whole_run_when_ollama_becomes_unreachable_mid_apply(tmp_path, monkeypatch, capsys):
    provider = OllamaUnreachableDuringApplyProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", MultiJobAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args(max_apps=10))

    assert provider.answer_calls == 1
    out = capsys.readouterr().out
    assert "Ollama is unreachable - stopping the run instead of repeating this for every posting." in out
    tracker = Tracker(settings.db_path)
    assert tracker.get_job("job2") is None
    assert tracker.get_job("job3") is None


class ClaudeMisconfiguredDuringApplyProvider(LLMProvider):
    """Same idea as OllamaUnreachableDuringApplyProvider above, for the
    apply phase's separate is-Claude-misconfigured check.
    """

    def __init__(self):
        self.answer_calls = 0

    def generate_structured(self, *, system, prompt, schema):
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
            self.answer_calls += 1
            raise ClaudeProviderError("Invalid ANTHROPIC_API_KEY.")
        raise AssertionError(f"Unexpected schema requested: {schema}")


def test_run_stops_the_whole_run_when_claude_becomes_misconfigured_mid_apply(tmp_path, monkeypatch, capsys):
    provider = ClaudeMisconfiguredDuringApplyProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", MultiJobAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args(max_apps=10))

    assert provider.answer_calls == 1
    out = capsys.readouterr().out
    assert "Claude provider is misconfigured" in out
    assert "stopping the run instead of repeating this for every posting" in out
    tracker = Tracker(settings.db_path)
    assert tracker.get_job("job2") is None
    assert tracker.get_job("job3") is None


class SearchFailsOnceAdapter(FakeAdapter):
    """search() raises on its first call - simulating a transient
    net::ERR_HTTP_RESPONSE_CODE_FAILURE that survives _goto_with_retry's
    own retries and still fails - then succeeds on the next one, proving
    a search failure costs only that cycle rather than crashing the run.
    """

    def __init__(self, page):
        super().__init__(page)
        self.search_calls = 0

    def search(self, keywords, location, max_results=25, experience_levels=None, include_external=False):
        self.search_calls += 1
        if self.search_calls == 1:
            raise RuntimeError("Failed to load https://www.linkedin.com/jobs/search/... after 3 attempts")
        return [JOB]


def test_run_search_failure_is_logged_and_does_not_crash(tmp_path, monkeypatch, capsys):
    """Real bug this guards against: adapter.search() itself failing (e.g.
    LinkedIn erroring on the search results page after its own retries are
    exhausted) propagated straight out of _run_apply_cycle() uncaught,
    crashing the whole process with a raw traceback instead of failing
    this one cycle cleanly - confirmed live.
    """
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", SearchFailsOnceAdapter)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args())  # must not raise

    assert "Error searching for postings" in capsys.readouterr().out
    failure_lines = settings.failed_applications_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(failure_lines) == 1
    assert json.loads(failure_lines[0])["action"] == "search_error"


class SignedOutAdapter(FakeAdapter):
    def __init__(self, page):
        super().__init__(page)
        self.search_calls = 0

    def search(self, keywords, location, max_results=25, experience_levels=None, include_external=False):
        self.search_calls += 1
        raise LinkedInSignedOut("https://www.linkedin.com/authwall?trk=bf")


def test_loop_stops_instead_of_retrying_when_linkedin_is_signed_out(tmp_path, monkeypatch, capsys):
    """A signed-out session fails every search identically until the user
    runs `job-bot login` - --loop must stop and say so, not sleep and retry
    forever the way an ordinary empty cycle does. time.sleep fails the
    test outright if called.
    """
    adapter = SignedOutAdapter(page=None)

    def fail_if_called(seconds):
        raise AssertionError("must not sleep/retry once LinkedIn is signed out - the loop should stop instead")

    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: FakeProvider())
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", lambda page: adapter)
    monkeypatch.setattr("job_bot.cli.time.sleep", fail_if_called)

    settings = make_settings(tmp_path)
    cmd_run(settings, make_args(loop=True, max_apps=10))  # must not raise

    out = capsys.readouterr().out
    assert "job-bot login" in out
    assert "Stopping the loop" in out
    assert adapter.search_calls == 1
    # the structured flag `job-bot doctor` reads back (not the message text)
    entry = AuditLogger(settings.audit_log_path).read_entries(action="search_error")[0]
    assert entry["details"]["signed_out"] is True


def test_loop_recovers_from_a_search_failure_on_the_next_cycle(tmp_path, monkeypatch):
    provider = FakeProvider()
    adapter = SearchFailsOnceAdapter(page=None)
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", lambda page: adapter)
    monkeypatch.setattr("job_bot.cli.time.sleep", lambda seconds: None)

    settings = make_settings(tmp_path, daily_application_cap=1)
    cmd_run(settings, make_args(loop=True, max_apps=1))  # must not raise

    tracker = Tracker(settings.db_path)
    assert tracker.has_applied("job1") is True
    # Cycle 1's search failed and applied nothing; cycle 2 recovered and
    # applied the one job the cap allowed, which is also what stopped the
    # loop - proving cycle 1's failure didn't end the run early.
    assert adapter.search_calls == 2


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
    application cap, not truly infinite. Every cycle here applies to
    something, so it must never sleep - each cycle starts immediately after
    the last, only the cap itself stops it (see the next test for the
    "nothing applied this cycle" case, which does sleep).
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
    # Never slept: cycle 1 applied to something, so cycle 2 started right
    # away, and the cap check stops the loop before a third cycle's sleep
    # would ever be reached.
    assert sleep_calls == []


class SleepThenReturnsAJobAdapter(FakeAdapter):
    """search() returns nothing on its first call (no eligible postings
    right now), then a fresh, never-before-seen posting on the next -
    exercises the one case where --loop still pauses between cycles: one
    that applied to nothing, as opposed to test_loop_runs_multiple_cycles_
    and_stops_once_the_daily_cap_is_reached above, where every cycle
    applies to something and must never pause.
    """

    def __init__(self, page):
        super().__init__(page)
        self.search_calls = 0

    def search(self, keywords, location, max_results=25, experience_levels=None, include_external=False):
        self.search_calls += 1
        if self.search_calls == 1:
            return []
        return [JobPosting(job_id="loop-job-1", title="Backend Engineer", company="Acme Corp", url="https://x/1", description="")]


def test_loop_only_sleeps_between_cycles_that_applied_to_nothing(tmp_path, monkeypatch):
    """Real bug this guards against: --loop used to pause
    --loop-interval-minutes between EVERY cycle, even one that just
    successfully applied to as many jobs as --max-apps allowed - needlessly
    slowing progress toward the daily cap. It must now only pause after a
    cycle that applied to nothing (here, cycle 1's empty search result),
    and go straight into the next cycle once postings are actually found.
    """
    provider = FakeProvider()
    adapter = SleepThenReturnsAJobAdapter(page=None)
    sleep_calls: list[float] = []
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", lambda page: adapter)
    monkeypatch.setattr("job_bot.cli.time.sleep", lambda seconds: sleep_calls.append(seconds))

    settings = make_settings(tmp_path, daily_application_cap=1)
    cmd_run(settings, make_args(loop=True, loop_interval_minutes=7, max_apps=1))

    tracker = Tracker(settings.db_path)
    assert tracker.has_applied("loop-job-1") is True
    # Slept once, after cycle 1's empty result - not after cycle 2, since
    # the cap check stops the loop first.
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


def test_run_reports_when_quit_ollama_itself_fails(tmp_path, monkeypatch, capsys):
    """_quit_ollama_if_configured prints a different message depending on
    whether quit_ollama() actually succeeded - every other quit_ollama test
    above stubs it to always return True, so the "could not quit" message
    (the real ollama_provider.quit_ollama() return value when every OS
    command it tries fails, e.g. Ollama already stopped) has never actually
    been printed by any test.
    """
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", FakeAdapter)
    monkeypatch.setattr("job_bot.cli.quit_ollama", lambda: False)

    settings = make_settings(
        tmp_path,
        llm_provider="ollama",
        anthropic_api_key=None,
        daily_application_cap=1,
        quit_ollama_when_done=True,
    )
    cmd_run(settings, make_args())

    assert "Daily cap reached - could not quit Ollama (it may already be stopped)." in capsys.readouterr().out


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


class AppliesOnceThenFindsNothingAdapter(FakeAdapter):
    """First search() returns a fresh posting (so cycle 1 applies to
    something); every call after returns nothing. --loop only ever calls
    time.sleep() - the one place a Ctrl+C can land mid-loop - when a cycle
    applied to nothing (see test_loop_only_sleeps_between_cycles_that_
    applied_to_nothing), so this is the shape needed to actually reach that
    call, rather than one where sleep is never invoked at all.
    """

    def __init__(self, page):
        super().__init__(page)
        self.search_calls = 0

    def search(self, keywords, location, max_results=25, experience_levels=None, include_external=False):
        self.search_calls += 1
        if self.search_calls == 1:
            return [
                JobPosting(
                    job_id="loop-job-1",
                    title="Backend Engineer",
                    company="Acme Corp",
                    url="https://x/loop-job-1",
                    description="",
                )
            ]
        return []


def test_loop_stops_cleanly_on_keyboard_interrupt(tmp_path, monkeypatch, capsys):
    """Real bug this test itself had: the original version used
    LoopFakeAdapter, which returns a fresh posting on every single search()
    call - every cycle then applies to something, and --loop only ever
    calls time.sleep() (where the interrupt is injected below) between
    cycles that applied to nothing. time.sleep() was therefore never
    called, and the `except KeyboardInterrupt` branch this test claims to
    cover was never reached - the test passed anyway, because cmd_run
    happened to stop on its own once the (deliberately high) daily cap was
    exhausted after 100 fast, no-op cycles. This version forces cycle 2 to
    find nothing, so time.sleep() - and the interrupt - are actually hit.
    """
    provider = FakeProvider()
    adapter = AppliesOnceThenFindsNothingAdapter(page=None)

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
    # Cycle 1 applied; cycle 2 found nothing, slept (raising the
    # interrupt), and the loop stopped there - a third cycle never ran.
    assert adapter.search_calls == 2
    assert "\nStopped." in capsys.readouterr().out


class ClosesThePageDuringApplyAdapter(FakeAdapter):
    """--loop has its own page.is_closed() check once per cycle, after
    run_one_cycle() returns - separate from the is_closed() guards inside
    _run_apply_cycle itself, which only catch the browser dying *during*
    fill_and_submit for a plain (non --loop) run (see
    ApplyClosesTheBrowserForFirstJobAdapter above). This simulates the
    window closing right as a cycle finishes applying successfully - the
    one shape only --loop's own top-level check catches.
    """

    def fill_and_submit(self, posting, *, answer_question, resume_path, cover_letter_text, dry_run):
        result = super().fill_and_submit(
            posting,
            answer_question=answer_question,
            resume_path=resume_path,
            cover_letter_text=cover_letter_text,
            dry_run=dry_run,
        )
        self.page.close()
        return result


def test_loop_stops_when_the_browser_window_closes_between_cycles(tmp_path, monkeypatch, capsys):
    provider = FakeProvider()
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: provider)
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session)
    monkeypatch.setattr("job_bot.cli.LinkedInAdapter", ClosesThePageDuringApplyAdapter)

    settings = make_settings(tmp_path, daily_application_cap=100)
    cmd_run(settings, make_args(loop=True, max_apps=1))

    assert "Browser window was closed - stopping." in capsys.readouterr().out
    tracker = Tracker(settings.db_path)
    assert tracker.has_applied(JOB.job_id) is True


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
