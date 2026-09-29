"""Tests for the small, previously-uncovered CLI commands: report, status,
blacklist, and export. Unlike test_cli_run.py these don't need a fake
browser/LLM provider - each command only touches the tracker DB, the
blacklist file, or stdout/a file.
"""

import argparse
import csv
import io
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from job_bot.cli import (
    EXPECTED_ERRORS,
    _apply_provider_overrides,
    _score_bucket_label,
    build_parser,
    cmd_audit_log,
    cmd_blacklist,
    cmd_dashboard,
    cmd_doctor,
    cmd_export,
    cmd_faq,
    cmd_gmail_sync,
    cmd_qa_history,
    cmd_report,
    cmd_resume_history,
    cmd_review_answers,
    cmd_status,
    cmd_test_provider,
    main,
)
from job_bot.config import Settings
from job_bot.llm.base import LLMProvider
from job_bot.models.schemas import JobMatchScore
from job_bot.resume.store import ResumeStore
from job_bot.safety.answer_gaps import AnswerGapStore
from job_bot.safety.audit_log import AuditLogger
from job_bot.safety.blacklist import CompanyBlacklist
from job_bot.safety.rate_limiter import RateLimiter
from job_bot.tracker.db import Tracker


class FakeScoreProvider(LLMProvider):
    """Returns a fixed passing JobMatchScore regardless of the prompt -
    enough for cmd_test_provider, which only cares that a real call
    round-trips and can be printed, not that the score itself is realistic.
    """

    def generate_structured(self, *, system, prompt, schema):
        return JobMatchScore(
            eligibility="pass",
            technical_fit=90,
            experience_fit=90,
            culture_fit=90,
            score=90,
            reasoning="Strong match",
            should_apply=True,
            missing_qualifications=[],
        )


class FakeGmailClientForCli:
    """Stands in for GmailClient in cmd_gmail_sync tests - cmd_gmail_sync
    constructs GmailClient itself from settings paths, so the class (not an
    instance) is what test_cli_commands.py's tests monkeypatch.
    """

    def __init__(self, *args, **kwargs):
        pass

    def search_messages(self, query, max_results=50):
        return []


def make_settings(tmp_path, **overrides) -> Settings:
    defaults = dict(
        _env_file=None,
        llm_provider="claude",
        anthropic_api_key="sk-ant-fake",
        resume_path=tmp_path / "resume.txt",
        faq_path=tmp_path / "faq.json",
        blacklist_path=tmp_path / "blacklist.json",
        db_path=tmp_path / "db.sqlite3",
        browser_profile_dir=tmp_path / "profile",
        audit_log_path=tmp_path / "audit.log",
        failed_applications_log_path=tmp_path / "failed_applications.log",
        answer_gaps_path=tmp_path / "answer_gaps.json",
        applications_dir=tmp_path / "applications",
        # Without these, any test checking a Gmail-related doctor result
        # (or leaving it at Settings' own default) was silently reading the
        # real, working-directory data/gmail_credentials.json/gmail_token.json
        # instead of an isolated fixture - happened to pass in this repo
        # checkout only because neither file exists here, not because the
        # test was actually isolated the way every other path above is.
        gmail_credentials_path=tmp_path / "gmail_credentials.json",
        gmail_token_path=tmp_path / "gmail_token.json",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def report_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        stale_days=None,
        by_score=False,
        by_eligibility=False,
        by_company=False,
        by_week=False,
        by_missing_qualifications=False,
        missing_qualifications_limit=None,
        format="text",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def doctor_args(**overrides) -> argparse.Namespace:
    defaults = dict(format="text")
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def review_answers_args(**overrides) -> argparse.Namespace:
    defaults = dict(search=None, format="text", dismiss=None)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def qa_history_args(**overrides) -> argparse.Namespace:
    defaults = dict(search=None, company=None, format="text")
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def resume_history_args(**overrides) -> argparse.Namespace:
    defaults = dict(search=None, company=None, format="text")
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def audit_log_args(**overrides) -> argparse.Namespace:
    defaults = dict(search=None, action=None, failures=False, format="text")
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def status_args(**overrides) -> argparse.Namespace:
    defaults = dict(job_id="job1", status=None, note=None, format="text")
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def gmail_sync_args(**overrides) -> argparse.Namespace:
    defaults = dict(days=None, max_emails=50, dry_run=False, format="text")
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def export_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        status=None, search=None, eligibility=None, company=None, stale_days=None, out=None, format="csv"
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _backdate_applied_at(db_path, job_id: str, when: datetime) -> None:
    """Directly rewrite applied_at, since mark_applied() always stamps
    "now" - tests that need a stale application have to backdate it after
    the fact rather than through the public Tracker API.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE jobs SET applied_at = ? WHERE job_id = ?", (when.isoformat(), job_id))
    conn.commit()
    conn.close()


# --- status ---


def test_status_updates_a_tracked_job(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")

    cmd_status(settings, status_args(job_id="job1", status="interviewing", note=None))

    assert tracker.get_job("job1")["status"] == "interviewing"
    assert "job1 -> interviewing" in capsys.readouterr().out


def test_status_format_json_is_ignored_when_a_status_change_is_given(tmp_path, capsys):
    """--format json only affects the no-<status>/--note view (see
    cmd_status's docstring) - an actual status change already prints a
    trivially-parseable one-line result and isn't meant to grow a second,
    JSON-shaped output for the same action.
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")

    cmd_status(settings, status_args(job_id="job1", status="interviewing", format="json"))

    assert "job1 -> interviewing" in capsys.readouterr().out


def test_status_on_unknown_job_id_exits_with_error(tmp_path, capsys):
    settings = make_settings(tmp_path)
    Tracker(settings.db_path)  # create the (empty) DB

    with pytest.raises(SystemExit) as exc_info:
        cmd_status(settings, status_args(job_id="does-not-exist", status="offer", note=None))

    assert exc_info.value.code == 1
    assert "Error" in capsys.readouterr().err


def test_status_with_no_status_arg_prints_the_job_record_and_does_not_change_it(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1", match_score=80)
    tracker.mark_applied("job1")

    cmd_status(settings, status_args(job_id="job1", status=None, note=None))

    out = capsys.readouterr().out
    assert "Backend Engineer" in out
    assert "Acme" in out
    assert "applied" in out
    assert "80" in out
    assert tracker.get_job("job1")["status"] == "applied"  # unchanged


def test_status_with_no_status_arg_and_format_json_prints_the_full_record(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score(
        "job1",
        "Backend Engineer",
        "Acme",
        "https://x/1",
        score=85,
        should_apply=True,
        reasoning="Great fit",
        eligibility="pass",
        eligibility_note="",
    )
    tracker.set_note("job1", "Referred by Jane.")
    tracker.record_qa("job1", "Years of Python?", "5")
    tracker.record_resume_generation(
        "job1", "Backend Engineer", "Acme", "A tailored summary.", ["Python", "AWS"], ["Did a thing."]
    )

    cmd_status(settings, status_args(job_id="job1", format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert payload["job_id"] == "job1"
    assert payload["status"] == "seen"
    assert payload["match_score"] == 85
    assert payload["match_reasoning"] == "Great fit"
    assert payload["eligibility"] == "pass"
    assert payload["notes"] == "Referred by Jane."
    assert payload["is_blacklisted"] is False
    assert len(payload["qa_history"]) == 1
    assert payload["qa_history"][0]["question"] == "Years of Python?"
    assert payload["qa_history"][0]["answer"] == "5"
    assert payload["resume_generation"]["summary"] == "A tailored summary."
    assert tracker.get_job("job1")["status"] == "seen"  # unchanged


def test_status_with_no_status_arg_and_format_json_flags_a_blacklisted_company(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme Corp", "https://x/1")
    CompanyBlacklist(settings.blacklist_path).add("Acme Corp")

    cmd_status(settings, status_args(job_id="job1", format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert payload["is_blacklisted"] is True


def test_status_with_no_status_arg_and_format_json_omits_optional_sections_when_absent(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")

    cmd_status(settings, status_args(job_id="job1", format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert payload["match_reasoning"] is None
    assert payload["notes"] is None
    assert payload["resume_generation"] is None
    assert payload["qa_history"] == []


def test_status_with_no_status_arg_and_format_json_on_unknown_job_id_exits_with_error(tmp_path, capsys):
    settings = make_settings(tmp_path)
    Tracker(settings.db_path)  # create the (empty) DB

    with pytest.raises(SystemExit) as exc_info:
        cmd_status(settings, status_args(job_id="does-not-exist", format="json"))

    assert exc_info.value.code == 1
    err = json.loads(capsys.readouterr().err)
    assert "does-not-exist" in err["error"]


def test_status_with_no_status_arg_includes_qa_history(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.record_qa("job1", "Years of Python?", "5")

    cmd_status(settings, status_args(job_id="job1", status=None, note=None))

    out = capsys.readouterr().out
    assert "Q&A history (1):" in out
    assert "Years of Python?" in out
    assert "5" in out


def test_status_with_no_status_arg_includes_the_tailored_resume_generation(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.record_resume_generation(
        "job1", "Backend Engineer", "Acme", "A tailored summary.", ["Python", "AWS"], ["Did a thing."]
    )

    cmd_status(settings, status_args(job_id="job1", status=None, note=None))

    out = capsys.readouterr().out
    assert "Tailored resume generated" in out
    assert "A tailored summary." in out
    assert "Python, AWS" in out


def test_status_with_no_status_arg_omits_resume_generation_section_when_never_generated(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")

    cmd_status(settings, status_args(job_id="job1", status=None, note=None))

    assert "Tailored resume generated" not in capsys.readouterr().out


def test_status_with_no_status_arg_on_unknown_job_id_exits_with_error(tmp_path, capsys):
    settings = make_settings(tmp_path)
    Tracker(settings.db_path)  # create the (empty) DB

    with pytest.raises(SystemExit) as exc_info:
        cmd_status(settings, status_args(job_id="does-not-exist", status=None, note=None))

    assert exc_info.value.code == 1
    assert "No tracked job" in capsys.readouterr().err


def test_status_note_flag_sets_a_note_without_changing_status(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.mark_applied("job1")

    cmd_status(
        settings, status_args(job_id="job1", status=None, note="Recruiter mentioned $150k base.")
    )

    assert "Note set for job1." in capsys.readouterr().out
    assert tracker.get_job("job1")["notes"] == "Recruiter mentioned $150k base."
    assert tracker.get_job("job1")["status"] == "applied"  # unchanged


def test_status_note_flag_combined_with_a_status_change_does_both(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")

    cmd_status(settings, status_args(job_id="job1", status="interviewing", note="Second round."))

    out = capsys.readouterr().out
    assert "Note set for job1." in out
    assert "job1 -> interviewing" in out
    job = tracker.get_job("job1")
    assert job["notes"] == "Second round."
    assert job["status"] == "interviewing"


def test_status_note_flag_on_unknown_job_id_exits_with_error(tmp_path, capsys):
    settings = make_settings(tmp_path)
    Tracker(settings.db_path)  # create the (empty) DB

    with pytest.raises(SystemExit) as exc_info:
        cmd_status(settings, status_args(job_id="does-not-exist", status=None, note="A note."))

    assert exc_info.value.code == 1
    assert "Error" in capsys.readouterr().err


def test_status_with_no_status_arg_shows_the_note_when_present(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.set_note("job1", "Referred by Jane.")

    cmd_status(settings, status_args(job_id="job1", status=None, note=None))

    assert "Note: Referred by Jane." in capsys.readouterr().out


def test_status_with_no_status_arg_omits_note_section_when_none_set(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")

    cmd_status(settings, status_args(job_id="job1", status=None, note=None))

    assert "Note:" not in capsys.readouterr().out


def test_status_with_no_status_arg_shows_match_reasoning_when_present(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score(
        "job1", "Backend Engineer", "Acme", "https://x/1", score=85, should_apply=True, reasoning="Great fit"
    )

    cmd_status(settings, status_args(job_id="job1", status=None, note=None))

    assert "Match reasoning: Great fit" in capsys.readouterr().out


def test_status_with_no_status_arg_omits_match_reasoning_section_when_never_scored(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")

    cmd_status(settings, status_args(job_id="job1", status=None, note=None))

    assert "Match reasoning:" not in capsys.readouterr().out


def test_status_with_no_status_arg_shows_missing_qualifications_when_present(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score(
        "job1",
        "Backend Engineer",
        "Acme",
        "https://x/1",
        score=70,
        should_apply=True,
        missing_qualifications=["AWS certification", "5+ years of Go"],
    )

    cmd_status(settings, status_args(job_id="job1", status=None, note=None))

    assert "Missing qualifications: AWS certification, 5+ years of Go" in capsys.readouterr().out


def test_status_with_no_status_arg_omits_missing_qualifications_section_when_none(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score("job1", "Backend Engineer", "Acme", "https://x/1", score=90, should_apply=True)

    cmd_status(settings, status_args(job_id="job1", status=None, note=None))

    assert "Missing qualifications:" not in capsys.readouterr().out


def test_status_json_includes_missing_qualifications_as_a_real_list(tmp_path, capsys):
    """The stored column is a JSON string (see Tracker.record_score()'s
    docstring) - the --format json payload must decode it back into a real
    array, not leak the raw stored string to a script consumer.
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score(
        "job1",
        "Backend Engineer",
        "Acme",
        "https://x/1",
        score=70,
        should_apply=True,
        missing_qualifications=["AWS certification"],
    )

    cmd_status(settings, status_args(job_id="job1", status=None, note=None, format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert payload["missing_qualifications"] == ["AWS certification"]


def test_status_with_no_status_arg_flags_a_failed_eligibility_verdict(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score(
        "job1",
        "Backend Engineer",
        "Acme",
        "https://x/1",
        score=20,
        should_apply=False,
        eligibility="fail",
        eligibility_note="Requires active US security clearance.",
    )

    cmd_status(settings, status_args(job_id="job1", status=None, note=None))

    out = capsys.readouterr().out
    assert "[!!] Eligibility: fail - Requires active US security clearance." in out


def test_status_with_no_status_arg_flags_a_flagged_eligibility_verdict(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score(
        "job1", "Backend Engineer", "Acme", "https://x/1", score=70, should_apply=True, eligibility="flag"
    )

    cmd_status(settings, status_args(job_id="job1", status=None, note=None))

    assert "[!!] Eligibility: flag" in capsys.readouterr().out


def test_status_with_no_status_arg_omits_eligibility_warning_when_eligibility_passes(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score(
        "job1", "Backend Engineer", "Acme", "https://x/1", score=85, should_apply=True, eligibility="pass"
    )

    cmd_status(settings, status_args(job_id="job1", status=None, note=None))

    assert "Eligibility:" not in capsys.readouterr().out


def test_status_with_no_status_arg_omits_eligibility_warning_when_never_scored(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")

    cmd_status(settings, status_args(job_id="job1", status=None, note=None))

    assert "Eligibility:" not in capsys.readouterr().out


def test_status_with_no_status_arg_warns_when_the_company_is_blacklisted(tmp_path, capsys):
    """Real gap this guards against: a job tracked/applied to before its
    company was blacklisted (or blacklisted afterward for an unrelated
    reason) had nothing surfacing the inconsistency - job-bot run's own
    blacklist check only ever runs at search time against new postings,
    never retroactively against what's already tracked.
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme Corp", "https://x/1")
    CompanyBlacklist(settings.blacklist_path).add("Acme Corp")

    cmd_status(settings, status_args(job_id="job1", status=None, note=None))

    assert "Acme Corp is on your blacklist." in capsys.readouterr().out


def test_status_with_no_status_arg_omits_blacklist_warning_when_not_blacklisted(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme Corp", "https://x/1")

    cmd_status(settings, status_args(job_id="job1", status=None, note=None))

    assert "blacklist" not in capsys.readouterr().out


# --- report ---


def test_report_prints_status_counts(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.upsert_job("job2", "Frontend Engineer", "Acme", "https://x/2")
    tracker.mark_applied("job2")

    cmd_report(settings, report_args())

    out = capsys.readouterr().out
    assert "seen" in out
    assert "applied" in out
    assert "total" in out
    assert "2" in out


def test_report_on_empty_tracker_says_so(tmp_path, capsys):
    settings = make_settings(tmp_path)
    Tracker(settings.db_path)

    cmd_report(settings, report_args())

    assert "No jobs tracked yet." in capsys.readouterr().out


def test_report_flags_stale_applications_past_the_threshold(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.mark_applied("job1")
    _backdate_applied_at(settings.db_path, "job1", datetime.now(UTC) - timedelta(days=20))
    tracker.upsert_job("job2", "Frontend Engineer", "Beta", "https://x/2")
    tracker.mark_applied("job2")  # applied just now - not stale

    cmd_report(settings, report_args(stale_days=14))

    out = capsys.readouterr().out
    assert "Applied 14+ days ago with no reply (1):" in out
    assert "job1" in out
    assert "job2" not in out.split("Applied 14+")[1]


def test_report_stale_days_defaults_to_settings(tmp_path, capsys):
    settings = make_settings(tmp_path, stale_after_days=5)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.mark_applied("job1")
    _backdate_applied_at(settings.db_path, "job1", datetime.now(UTC) - timedelta(days=10))

    cmd_report(settings, report_args())

    assert "Applied 5+ days ago with no reply (1):" in capsys.readouterr().out


def test_report_omits_stale_section_when_nothing_is_stale(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.mark_applied("job1")

    cmd_report(settings, report_args(stale_days=14))

    assert "no reply" not in capsys.readouterr().out


def test_report_by_score_breaks_down_outcomes_by_score_bucket(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score("job1", "Backend Engineer", "Acme", "https://x/1", score=92, should_apply=True)
    tracker.update_status("job1", "offer")
    tracker.record_score("job2", "Frontend Engineer", "Beta", "https://x/2", score=40, should_apply=False)
    tracker.upsert_job("job3", "Unscored Role", "Gamma", "https://x/3")  # no match_score yet

    cmd_report(settings, report_args(by_score=True))

    out = capsys.readouterr().out
    assert "Outcomes by match score:" in out
    assert "90-100" in out
    assert "0-59" in out
    # job3 has no match_score and must not appear in the breakdown at all.
    assert out.count("job3") == 0


def test_report_by_score_prints_nothing_when_no_job_has_been_scored(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")  # no match_score yet

    cmd_report(settings, report_args(by_score=True))

    assert "Outcomes by match score:" not in capsys.readouterr().out


def test_report_by_eligibility_breaks_down_outcomes_by_verdict(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score(
        "job1", "Backend Engineer", "Acme", "https://x/1", score=90, should_apply=True, eligibility="pass"
    )
    tracker.record_score(
        "job2", "Frontend Engineer", "Beta", "https://x/2", score=20, should_apply=False, eligibility="fail"
    )
    tracker.record_score(
        "job3", "DevOps Engineer", "Gamma", "https://x/3", score=70, should_apply=True, eligibility="flag"
    )
    tracker.upsert_job("job4", "Unscored Role", "Delta", "https://x/4")  # never scored at all

    cmd_report(settings, report_args(by_eligibility=True))

    out = capsys.readouterr().out
    assert "Outcomes by eligibility verdict:" in out
    lines = [line for line in out.splitlines() if line.strip()]
    # pass, then flag, then fail, then not scored - not alphabetical.
    verdict_lines = [line for line in lines if line.split()[0] in ("pass", "flag", "fail", "not")]
    assert [line.split()[0] for line in verdict_lines] == ["pass", "flag", "fail", "not"]
    assert "pass" in out and "1" in out
    assert "fail" in out
    assert "flag" in out
    assert "not scored" in out


def test_report_by_eligibility_prints_nothing_when_no_job_is_tracked(tmp_path, capsys):
    settings = make_settings(tmp_path)
    Tracker(settings.db_path)  # create the (empty) DB

    cmd_report(settings, report_args(by_eligibility=True))

    assert "Outcomes by eligibility verdict:" not in capsys.readouterr().out


def test_report_format_json_includes_by_eligibility_when_requested(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score(
        "job1", "Backend Engineer", "Acme", "https://x/1", score=90, should_apply=True, eligibility="pass"
    )

    cmd_report(settings, report_args(by_eligibility=True, format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert payload["by_eligibility"] == {"pass": 1}


def test_report_format_json_omits_by_eligibility_when_not_requested(tmp_path, capsys):
    settings = make_settings(tmp_path)
    Tracker(settings.db_path)  # create the (empty) DB

    cmd_report(settings, report_args(format="json"))

    assert "by_eligibility" not in json.loads(capsys.readouterr().out)


def test_report_by_company_breaks_down_outcomes_most_applied_first(tmp_path, capsys):
    """Zeta has more tracked jobs than Acme, deliberately the opposite of
    alphabetical order - so this only passes if the breakdown is genuinely
    sorted by total count, not coincidentally by name.
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.mark_applied("job1")
    tracker.upsert_job("job2", "SRE", "Zeta", "https://x/2")
    tracker.mark_applied("job2")
    tracker.update_status("job2", "interviewing")
    tracker.upsert_job("job3", "Platform Engineer", "Zeta", "https://x/3")

    cmd_report(settings, report_args(by_company=True))

    out = capsys.readouterr().out
    assert "Outcomes by company:" in out
    lines = [line for line in out.splitlines() if line.strip()]
    company_lines = [line for line in lines if line.split()[0] in ("Acme", "Zeta")]
    # Zeta has 2 tracked jobs, Acme has 1 - most-applied first, not alphabetical.
    assert [line.split()[0] for line in company_lines] == ["Zeta", "Acme"]


def test_report_by_company_prints_nothing_when_no_job_is_tracked(tmp_path, capsys):
    settings = make_settings(tmp_path)
    Tracker(settings.db_path)  # create the (empty) DB

    cmd_report(settings, report_args(by_company=True))

    assert "Outcomes by company:" not in capsys.readouterr().out


def test_report_by_company_omitted_without_the_flag(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")

    cmd_report(settings, report_args())

    assert "Outcomes by company" not in capsys.readouterr().out


def test_report_json_includes_by_company_only_when_requested(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")

    cmd_report(settings, report_args(by_company=True, format="json"))
    payload = json.loads(capsys.readouterr().out)
    assert payload["by_company"] == {"Acme": {"seen": 1}}

    cmd_report(settings, report_args(format="json"))
    payload = json.loads(capsys.readouterr().out)
    assert "by_company" not in payload


@pytest.fixture
def chicago_tz(monkeypatch):
    """Pin the local timezone so _week_start()'s local-time bucketing is
    deterministic regardless of the machine running the suite.
    """
    import time

    monkeypatch.setenv("TZ", "America/Chicago")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_report_by_week_counts_applications_per_week_most_recent_first(tmp_path, capsys, chicago_tz):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    for job_id in ("job1", "job2", "job3"):
        tracker.upsert_job(job_id, "Engineer", "Acme", f"https://x/{job_id}")
        tracker.mark_applied(job_id)
    tracker.update_status("job2", "interviewing")
    # Wednesdays at midday: job1 in the week of 2026-09-07, job2/job3 in 2026-09-14.
    _backdate_applied_at(settings.db_path, "job1", datetime(2026, 9, 9, 17, tzinfo=UTC))
    _backdate_applied_at(settings.db_path, "job2", datetime(2026, 9, 16, 17, tzinfo=UTC))
    _backdate_applied_at(settings.db_path, "job3", datetime(2026, 9, 16, 18, tzinfo=UTC))

    cmd_report(settings, report_args(by_week=True))

    out = capsys.readouterr().out
    assert "Applications by week:" in out
    week_lines = [line.split() for line in out.splitlines() if line.startswith("2026-")]
    # columns: week, applied, interviewing, total - most recent week first
    assert week_lines == [["2026-09-14", "1", "1", "2"], ["2026-09-07", "1", "0", "1"]]


def test_report_by_week_leaves_out_jobs_never_applied_to(tmp_path, capsys, chicago_tz):
    """A seen/skipped job has no applied_at - it was never an application,
    so it must not show up in any week (and with nothing applied, the
    section is omitted entirely rather than printing an empty table).
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Engineer", "Acme", "https://x/1")

    cmd_report(settings, report_args(by_week=True))

    assert "Applications by week:" not in capsys.readouterr().out


def test_report_by_week_buckets_by_local_time_not_utc(tmp_path, capsys, chicago_tz):
    """Sunday 2026-09-13 at 8pm in Chicago is already Monday 01:00 UTC -
    it belongs to the week the user actually sent it (week of 2026-09-07),
    not the next one.
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Engineer", "Acme", "https://x/1")
    tracker.mark_applied("job1")
    _backdate_applied_at(settings.db_path, "job1", datetime(2026, 9, 14, 1, tzinfo=UTC))

    cmd_report(settings, report_args(by_week=True, format="json"))

    assert json.loads(capsys.readouterr().out)["by_week"] == {"2026-09-07": {"applied": 1}}


def test_report_json_includes_by_week_only_when_requested(tmp_path, capsys):
    settings = make_settings(tmp_path)
    Tracker(settings.db_path)

    cmd_report(settings, report_args(format="json"))

    assert "by_week" not in json.loads(capsys.readouterr().out)


def test_report_by_missing_qualifications_counts_most_common_gaps_first(tmp_path, capsys):
    """"Docker" is deliberately the least-frequent phrase, in the middle of
    insertion order - this only passes if the breakdown is genuinely sorted
    by count, not coincidentally by insertion or alphabetical order.
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
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
    tracker.upsert_job("job3", "Unscored Role", "Gamma", "https://x/3")  # no missing_qualifications at all

    cmd_report(settings, report_args(by_missing_qualifications=True))

    out = capsys.readouterr().out
    assert "Most common missing qualifications:" in out
    lines = [line.strip() for line in out.splitlines() if "Kubernetes" in line or "Docker" in line]
    assert lines == ["2  Kubernetes experience", "1  Docker"]


def test_report_by_missing_qualifications_prints_nothing_when_none_are_recorded(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score("job1", "Backend Engineer", "Acme", "https://x/1", score=90, should_apply=True)

    cmd_report(settings, report_args(by_missing_qualifications=True))

    assert "Most common missing qualifications:" not in capsys.readouterr().out


def test_report_by_missing_qualifications_omitted_without_the_flag(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score(
        "job1",
        "Backend Engineer",
        "Acme",
        "https://x/1",
        score=70,
        should_apply=True,
        missing_qualifications=["Kubernetes experience"],
    )

    cmd_report(settings, report_args())

    assert "Most common missing qualifications" not in capsys.readouterr().out


def test_report_json_includes_by_missing_qualifications_only_when_requested(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score(
        "job1",
        "Backend Engineer",
        "Acme",
        "https://x/1",
        score=70,
        should_apply=True,
        missing_qualifications=["Kubernetes experience"],
    )

    cmd_report(settings, report_args(by_missing_qualifications=True, format="json"))
    payload = json.loads(capsys.readouterr().out)
    assert payload["by_missing_qualifications"] == {"Kubernetes experience": 1}

    cmd_report(settings, report_args(format="json"))
    payload = json.loads(capsys.readouterr().out)
    assert "by_missing_qualifications" not in payload


def test_report_missing_qualifications_limit_keeps_only_the_n_most_common(tmp_path, capsys):
    """Three distinct phrases at counts 2/1/1 - --missing-qualifications-limit 1 must keep only
    "Kubernetes experience" (the genuine count=2 repeat), not one of the two count=1 phrases an
    unlimited breakdown would list right alongside it.
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
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
        missing_qualifications=["Kubernetes experience", "Terraform"],
    )

    cmd_report(settings, report_args(by_missing_qualifications=True, missing_qualifications_limit=1))

    out = capsys.readouterr().out
    assert "Kubernetes experience" in out
    assert "Docker" not in out
    assert "Terraform" not in out


def test_report_json_missing_qualifications_limit_matches_the_text_output(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
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
        missing_qualifications=["Kubernetes experience", "Terraform"],
    )

    cmd_report(
        settings,
        report_args(by_missing_qualifications=True, missing_qualifications_limit=1, format="json"),
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["by_missing_qualifications"] == {"Kubernetes experience": 2}


def test_report_missing_qualifications_no_limit_by_default(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score(
        "job1",
        "Backend Engineer",
        "Acme",
        "https://x/1",
        score=70,
        should_apply=True,
        missing_qualifications=["Kubernetes experience", "Docker"],
    )

    cmd_report(settings, report_args(by_missing_qualifications=True))

    out = capsys.readouterr().out
    assert "Kubernetes experience" in out
    assert "Docker" in out


def test_score_bucket_label_falls_back_for_an_out_of_range_score():
    """SCORE_BUCKETS spans 0-100 inclusive, which every real LLM-produced
    score should fall within - this is the fallback for a score outside
    that range slipping through anyway, so a stray value still renders as
    "?" instead of silently vanishing from every bucket.
    """
    assert _score_bucket_label(-5) == "?"
    assert _score_bucket_label(150) == "?"


# --- _apply_provider_overrides ---


def test_apply_provider_overrides_switches_provider(tmp_path):
    settings = make_settings(tmp_path)
    args = argparse.Namespace(provider="ollama", model=None)

    _apply_provider_overrides(settings, args)

    assert settings.llm_provider == "ollama"


def test_apply_provider_overrides_sets_claude_model_when_provider_stays_claude(tmp_path):
    settings = make_settings(tmp_path)
    args = argparse.Namespace(provider=None, model="claude-opus-5")

    _apply_provider_overrides(settings, args)

    assert settings.claude_model == "claude-opus-5"
    assert settings.llm_provider == "claude"


def test_apply_provider_overrides_sets_ollama_model_when_switching_to_ollama(tmp_path):
    settings = make_settings(tmp_path)
    args = argparse.Namespace(provider="ollama", model="deepseek-r1:8b")

    _apply_provider_overrides(settings, args)

    assert settings.ollama_model == "deepseek-r1:8b"


def test_apply_provider_overrides_leaves_settings_unchanged_when_neither_is_given(tmp_path):
    settings = make_settings(tmp_path, llm_provider="claude", claude_model="claude-opus-5")
    args = argparse.Namespace(provider=None, model=None)

    _apply_provider_overrides(settings, args)

    assert settings.llm_provider == "claude"
    assert settings.claude_model == "claude-opus-5"


def test_report_by_score_omitted_without_the_flag(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score("job1", "Backend Engineer", "Acme", "https://x/1", score=92, should_apply=True)

    cmd_report(settings, report_args())

    assert "Outcomes by match score" not in capsys.readouterr().out


def test_report_json_includes_counts_total_and_stale(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.upsert_job("job2", "Frontend Engineer", "Beta", "https://x/2")
    tracker.mark_applied("job2")
    _backdate_applied_at(settings.db_path, "job2", datetime.now(UTC) - timedelta(days=20))

    cmd_report(settings, report_args(stale_days=14, format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"] == {"seen": 1, "applied": 1}
    assert payload["total"] == 2
    assert payload["stale_days"] == 14
    assert len(payload["stale"]) == 1
    stale_entry = payload["stale"][0]
    assert stale_entry["job_id"] == "job2"
    assert stale_entry["company"] == "Beta"
    assert stale_entry["title"] == "Frontend Engineer"
    assert stale_entry["applied_at"]  # a non-empty ISO timestamp string


def test_report_json_includes_by_score_only_when_requested(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score("job1", "Backend Engineer", "Acme", "https://x/1", score=92, should_apply=True)

    cmd_report(settings, report_args(by_score=True, format="json"))
    payload = json.loads(capsys.readouterr().out)
    assert payload["by_score"] == {"90-100": {"seen": 1}}

    cmd_report(settings, report_args(format="json"))
    payload = json.loads(capsys.readouterr().out)
    assert "by_score" not in payload


def test_report_json_on_empty_tracker_is_still_valid_json(tmp_path, capsys):
    settings = make_settings(tmp_path)
    Tracker(settings.db_path)

    cmd_report(settings, report_args(format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"counts": {}, "total": 0, "stale_days": settings.stale_after_days, "stale": []}


# --- blacklist ---


def test_blacklist_add_list_remove_round_trip(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="add", company=["Acme Corp"], reason=None))
    assert "Added to blacklist: Acme Corp" in capsys.readouterr().out

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="list", company=None, search=None, format="text"))
    assert "Acme Corp" in capsys.readouterr().out  # display casing preserved, not normalized

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="remove", company=["Acme Corp"]))
    assert "Removed from blacklist: Acme Corp" in capsys.readouterr().out

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="list", company=None, search=None, format="text"))
    assert "Blacklist is empty." in capsys.readouterr().out


def test_blacklist_add_with_reason_shows_it_in_list_text_output(tmp_path, capsys):
    settings = make_settings(tmp_path)
    cmd_blacklist(
        settings,
        argparse.Namespace(blacklist_action="add", company=["Acme Corp"], reason="no H1B sponsorship"),
    )
    capsys.readouterr()

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="list", company=None, search=None, format="text"))

    assert "Acme Corp  - no H1B sponsorship" in capsys.readouterr().out


def test_blacklist_add_without_reason_shows_no_dash_in_list_text_output(tmp_path, capsys):
    """A company added with no --reason must print as a bare name, not a
    trailing "  - " with nothing after it.
    """
    settings = make_settings(tmp_path)
    cmd_blacklist(settings, argparse.Namespace(blacklist_action="add", company=["Acme Corp"], reason=None))
    capsys.readouterr()

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="list", company=None, search=None, format="text"))

    out = capsys.readouterr().out
    assert out.strip() == "Acme Corp"


def test_blacklist_add_reason_applies_to_every_company_in_one_call(tmp_path, capsys):
    settings = make_settings(tmp_path)
    cmd_blacklist(
        settings,
        argparse.Namespace(
            blacklist_action="add", company=["Acme Corp", "Beta Inc"], reason="past employer"
        ),
    )
    capsys.readouterr()

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="list", company=None, search=None, format="text"))

    out = capsys.readouterr().out
    assert "Acme Corp  - past employer" in out
    assert "Beta Inc  - past employer" in out


def test_blacklist_add_re_adding_a_company_updates_its_reason(tmp_path, capsys):
    """add() already overwrites the display-name casing on a re-add - the
    reason must update the same way, not stick with whatever the first
    `add` call gave it.
    """
    settings = make_settings(tmp_path)
    cmd_blacklist(
        settings, argparse.Namespace(blacklist_action="add", company=["Acme Corp"], reason="typo reason")
    )
    capsys.readouterr()

    cmd_blacklist(
        settings, argparse.Namespace(blacklist_action="add", company=["Acme Corp"], reason="correct reason")
    )
    capsys.readouterr()
    cmd_blacklist(settings, argparse.Namespace(blacklist_action="list", company=None, search=None, format="text"))

    out = capsys.readouterr().out
    assert "Acme Corp  - correct reason" in out
    assert "typo reason" not in out


def test_blacklist_list_format_json_prints_name_and_reason(tmp_path, capsys):
    """Same reasoning `job-bot faq list --format json` was added for: a
    script or cron job that wants the structured data shouldn't have to
    scrape human-formatted text - `blacklist export` gives JSON-adjacent
    output but only ever the plain one-company-per-line shape `import`
    reads back, never real JSON. Each entry is {"name", "reason"} now
    (reason "" when none was given), not a bare name string - the shape
    change that came with `job-bot blacklist add --reason`.
    """
    settings = make_settings(tmp_path)
    cmd_blacklist(
        settings,
        argparse.Namespace(
            blacklist_action="add", company=["Acme Corp"], reason="no H1B sponsorship"
        ),
    )
    cmd_blacklist(settings, argparse.Namespace(blacklist_action="add", company=["Beta Inc"], reason=None))
    capsys.readouterr()

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="list", company=None, search=None, format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {"name": "Acme Corp", "reason": "no H1B sponsorship"},
        {"name": "Beta Inc", "reason": ""},
    ]


def test_blacklist_list_search_matches_the_company_name(tmp_path, capsys):
    settings = make_settings(tmp_path)
    cmd_blacklist(settings, argparse.Namespace(blacklist_action="add", company=["Acme Corp"], reason=None))
    cmd_blacklist(settings, argparse.Namespace(blacklist_action="add", company=["Beta Inc"], reason=None))
    capsys.readouterr()

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="list", company=None, search="acme", format="text"))

    out = capsys.readouterr().out
    assert "Acme Corp" in out
    assert "Beta Inc" not in out


def test_blacklist_list_search_matches_the_reason_not_just_the_name(tmp_path, capsys):
    """The whole point of searching by reason: once the list is long
    enough to need searching, remembering *why* a company was
    blacklisted is often easier than its exact name.
    """
    settings = make_settings(tmp_path)
    cmd_blacklist(
        settings,
        argparse.Namespace(blacklist_action="add", company=["Acme Corp"], reason="no H1B sponsorship"),
    )
    cmd_blacklist(settings, argparse.Namespace(blacklist_action="add", company=["Beta Inc"], reason=None))
    capsys.readouterr()

    cmd_blacklist(
        settings, argparse.Namespace(blacklist_action="list", company=None, search="sponsorship", format="text")
    )

    out = capsys.readouterr().out
    assert "Acme Corp" in out
    assert "Beta Inc" not in out


def test_blacklist_list_search_says_so_when_nothing_matches(tmp_path, capsys):
    settings = make_settings(tmp_path)
    cmd_blacklist(settings, argparse.Namespace(blacklist_action="add", company=["Acme Corp"], reason=None))
    capsys.readouterr()

    cmd_blacklist(
        settings, argparse.Namespace(blacklist_action="list", company=None, search="nonexistent", format="text")
    )

    assert 'No blacklisted companies matching "nonexistent".' in capsys.readouterr().out


def test_blacklist_list_format_json_on_empty_blacklist_is_still_valid_json(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="list", company=None, search=None, format="json"))

    assert json.loads(capsys.readouterr().out) == []


def test_blacklist_add_warns_about_in_progress_applications_at_that_company(tmp_path, capsys):
    """Real mistake this guards against: blacklisting only stops future
    applications (CompanyBlacklist doesn't touch the tracker at all) - if
    you're mid-interview somewhere and blacklist the wrong name (a typo, or
    a company you confused with a similarly-named one), nothing here would
    otherwise tell you that name still has an active tracked application.
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme Corp", "https://x/1")
    tracker.mark_applied("job1")
    tracker.update_status("job1", "interviewing")

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="add", company=["Acme Corp"], reason=None))

    out = capsys.readouterr().out
    assert "Added to blacklist: Acme Corp" in out
    assert "1 tracked application(s) at Acme Corp are still in progress (interviewing)" in out
    assert "blacklisting only stops future applications" in out


def test_blacklist_add_warning_matches_case_and_spacing_insensitively(tmp_path, capsys):
    """CompanyBlacklist.is_blocked() itself matches via normalize_company_name()
    - this warning must agree with that, or it would miss the exact "same
    company, different casing/spacing" case the blacklist itself already
    treats as identical.
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "  ACME   corp  ", "https://x/1")
    tracker.mark_applied("job1")

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="add", company=["Acme Corp"], reason=None))

    assert "still in progress" in capsys.readouterr().out


def test_blacklist_add_no_warning_when_nothing_is_in_progress(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme Corp", "https://x/1")  # status=seen

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="add", company=["Acme Corp"], reason=None))

    assert "still in progress" not in capsys.readouterr().out


def test_blacklist_add_no_warning_for_closed_outcomes(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme Corp", "https://x/1")
    tracker.mark_applied("job1")
    tracker.update_status("job1", "rejected")

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="add", company=["Acme Corp"], reason=None))

    assert "still in progress" not in capsys.readouterr().out


def test_blacklist_remove_of_absent_company_says_so(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="remove", company=["Nobody Inc"]))

    assert "Not on the blacklist: Nobody Inc" in capsys.readouterr().out


def test_blacklist_check_says_so_when_not_blacklisted(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="check", company="Acme Corp", format="text"))

    assert "Acme Corp is not blacklisted." in capsys.readouterr().out


def test_blacklist_check_shows_the_reason_when_blacklisted(tmp_path, capsys):
    settings = make_settings(tmp_path)
    cmd_blacklist(
        settings,
        argparse.Namespace(blacklist_action="add", company=["Acme Corp"], reason="no H1B sponsorship"),
    )
    capsys.readouterr()

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="check", company="Acme Corp", format="text"))

    assert "Acme Corp is blacklisted - no H1B sponsorship." in capsys.readouterr().out


def test_blacklist_check_omits_the_dash_when_no_reason_was_given(tmp_path, capsys):
    settings = make_settings(tmp_path)
    cmd_blacklist(settings, argparse.Namespace(blacklist_action="add", company=["Acme Corp"], reason=None))
    capsys.readouterr()

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="check", company="Acme Corp", format="text"))

    assert capsys.readouterr().out.strip() == "Acme Corp is blacklisted."


def test_blacklist_check_matches_case_and_spacing_insensitively_and_shows_stored_casing(tmp_path, capsys):
    """The stored display casing/spacing is shown back, not whatever the
    user happened to type on the `check` call itself - the same "as
    typed at add time" display CompanyBlacklist already preserves for
    `list`.
    """
    settings = make_settings(tmp_path)
    cmd_blacklist(settings, argparse.Namespace(blacklist_action="add", company=["Acme   Corp"], reason=None))
    capsys.readouterr()

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="check", company="  acme corp  ", format="text"))

    assert "Acme   Corp is blacklisted." in capsys.readouterr().out


def test_blacklist_check_format_json_when_blacklisted(tmp_path, capsys):
    settings = make_settings(tmp_path)
    cmd_blacklist(
        settings,
        argparse.Namespace(blacklist_action="add", company=["Acme Corp"], reason="no H1B sponsorship"),
    )
    capsys.readouterr()

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="check", company="Acme Corp", format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"company": "Acme Corp", "blocked": True, "reason": "no H1B sponsorship"}


def test_blacklist_check_format_json_when_not_blacklisted(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="check", company="Acme Corp", format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"company": "Acme Corp", "blocked": False, "reason": None}


def test_blacklist_add_accepts_multiple_companies_in_one_call(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_blacklist(
        settings, argparse.Namespace(blacklist_action="add", company=["Acme Corp", "Beta Inc"], reason=None)
    )

    out = capsys.readouterr().out
    assert "Added to blacklist: Acme Corp" in out
    assert "Added to blacklist: Beta Inc" in out
    cmd_blacklist(settings, argparse.Namespace(blacklist_action="list", company=None, search=None, format="text"))
    listed = capsys.readouterr().out
    assert "Acme Corp" in listed
    assert "Beta Inc" in listed


def test_blacklist_remove_accepts_multiple_companies_in_one_call(tmp_path, capsys):
    settings = make_settings(tmp_path)
    cmd_blacklist(
        settings, argparse.Namespace(blacklist_action="add", company=["Acme Corp", "Beta Inc"], reason=None)
    )
    capsys.readouterr()

    cmd_blacklist(
        settings, argparse.Namespace(blacklist_action="remove", company=["Acme Corp", "Beta Inc"])
    )

    out = capsys.readouterr().out
    assert "Removed from blacklist: Acme Corp" in out
    assert "Removed from blacklist: Beta Inc" in out
    cmd_blacklist(settings, argparse.Namespace(blacklist_action="list", company=None, search=None, format="text"))
    assert "Blacklist is empty." in capsys.readouterr().out


def test_blacklist_import_adds_every_company_skipping_blanks_and_comments(tmp_path, capsys):
    settings = make_settings(tmp_path)
    import_file = tmp_path / "past_employers.txt"
    import_file.write_text(
        "# Companies I've already worked for\n"
        "Acme Corp\n"
        "\n"
        "  Beta Inc  \n"
        "# a trailing comment\n"
        "Gamma LLC\n",
        encoding="utf-8",
    )

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="import", file=import_file))

    out = capsys.readouterr().out
    assert f"Imported 3 companies from {import_file}." in out
    cmd_blacklist(settings, argparse.Namespace(blacklist_action="list", company=None, search=None, format="text"))
    listed = capsys.readouterr().out
    assert "Acme Corp" in listed
    assert "Beta Inc" in listed
    assert "Gamma LLC" in listed


def test_blacklist_import_reports_singular_for_one_company(tmp_path, capsys):
    settings = make_settings(tmp_path)
    import_file = tmp_path / "one.txt"
    import_file.write_text("Acme Corp\n", encoding="utf-8")

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="import", file=import_file))

    assert f"Imported 1 company from {import_file}." in capsys.readouterr().out


def test_blacklist_import_of_missing_file_exits_with_error(tmp_path, capsys):
    settings = make_settings(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        cmd_blacklist(
            settings, argparse.Namespace(blacklist_action="import", file=tmp_path / "does-not-exist.txt")
        )

    assert exc_info.value.code == 1
    assert "Error" in capsys.readouterr().err


def test_blacklist_import_of_a_non_utf8_file_exits_with_a_clean_error(tmp_path, capsys):
    """Real failure this guards against: a --file saved with a non-UTF-8
    encoding (e.g. exported from a spreadsheet on Windows) previously
    crashed this with a raw, uncaught UnicodeDecodeError instead of the
    same clean "could not read" message an unreadable/missing file
    already gets here.
    """
    settings = make_settings(tmp_path)
    bad_file = tmp_path / "companies.txt"
    bad_file.write_bytes("Acme ’s Corp".encode("cp1252"))

    with pytest.raises(SystemExit) as exc_info:
        cmd_blacklist(settings, argparse.Namespace(blacklist_action="import", file=bad_file))

    assert exc_info.value.code == 1
    assert "Error: could not read" in capsys.readouterr().err


def test_blacklist_export_to_stdout_prints_one_company_per_line(tmp_path, capsys):
    settings = make_settings(tmp_path)
    cmd_blacklist(
        settings, argparse.Namespace(blacklist_action="add", company=["Acme Corp", "Beta Inc"], reason=None)
    )
    capsys.readouterr()

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="export", out=None))

    out = capsys.readouterr().out
    assert out == "Acme Corp\nBeta Inc\n"


def test_blacklist_export_round_trips_through_import(tmp_path, capsys):
    settings = make_settings(tmp_path)
    cmd_blacklist(
        settings, argparse.Namespace(blacklist_action="add", company=["Acme Corp", "Beta Inc"], reason=None)
    )
    capsys.readouterr()
    export_file = tmp_path / "backup.txt"

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="export", out=export_file))
    assert f"Exported 2 companies to {export_file}." in capsys.readouterr().out

    other_settings = make_settings(tmp_path, blacklist_path=tmp_path / "other_blacklist.json")
    cmd_blacklist(other_settings, argparse.Namespace(blacklist_action="import", file=export_file))
    capsys.readouterr()
    cmd_blacklist(other_settings, argparse.Namespace(blacklist_action="list", company=None, search=None, format="text"))
    listed = capsys.readouterr().out
    assert "Acme Corp" in listed
    assert "Beta Inc" in listed


def test_blacklist_export_of_empty_blacklist_prints_nothing(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="export", out=None))

    assert capsys.readouterr().out == ""


# --- export ---


def test_export_to_stdout_is_valid_csv_with_all_jobs(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1", match_score=80)
    tracker.upsert_job("job2", "Frontend Engineer", "Beta", "https://x/2", match_score=60)
    tracker.mark_applied("job2")

    cmd_export(settings, export_args(status=None, search=None, out=None, format="csv"))

    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert [r["job_id"] for r in rows] == ["job1", "job2"]
    assert rows[1]["status"] == "applied"
    assert rows[1]["applied_at"]


def test_export_csv_includes_notes(tmp_path, capsys):
    """Real gap this guards against: EXPORT_FIELDS didn't include `notes`
    when that column was added (see Tracker.set_note()), so a job's note
    was silently excluded from every export - the one place a user would
    most expect to see everything recorded about a job.
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.set_note("job1", "Recruiter mentioned $150k base.")

    cmd_export(settings, export_args(status=None, search=None, out=None, format="csv"))

    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert rows[0]["notes"] == "Recruiter mentioned $150k base."


def test_export_csv_includes_missing_qualifications(tmp_path, capsys):
    """Same gap class test_export_csv_includes_notes already guards
    against, for missing_qualifications: EXPORT_FIELDS didn't include it
    when the column was added, so it was silently excluded from every
    export even though it's now real, persisted data. Stored as JSON, so
    the CSV cell holds the raw JSON-encoded string - there's no better
    native representation for a list in one CSV cell.
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score(
        "job1",
        "Backend Engineer",
        "Acme",
        "https://x/1",
        score=70,
        should_apply=True,
        missing_qualifications=["AWS certification", "5+ years of Go"],
    )

    cmd_export(settings, export_args(status=None, search=None, out=None, format="csv"))

    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert json.loads(rows[0]["missing_qualifications"]) == ["AWS certification", "5+ years of Go"]


def test_export_filters_by_status(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.upsert_job("job2", "Frontend Engineer", "Beta", "https://x/2")
    tracker.mark_applied("job2")

    cmd_export(settings, export_args(status="applied", search=None, out=None, format="csv"))

    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert [r["job_id"] for r in rows] == ["job2"]


def test_export_filters_by_eligibility(tmp_path, capsys):
    """--eligibility is the exact counterpart to --status, for pulling
    every job with a given eligibility-gate verdict without guessing a
    search term that happens to match all of them.
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score(
        "job1", "Backend Engineer", "Acme", "https://x/1", score=20, should_apply=False, eligibility="fail"
    )
    tracker.record_score(
        "job2", "Frontend Engineer", "Beta", "https://x/2", score=90, should_apply=True, eligibility="pass"
    )

    cmd_export(settings, export_args(eligibility="fail"))

    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert [r["job_id"] for r in rows] == ["job1"]


def test_export_filters_by_company(tmp_path, capsys):
    """--company is an exact (normalized) match, unlike --search's fuzzy
    substring - "Acme Corp" here must not also pull "Acme Robotics", an
    unrelated company that happens to share the word "Acme".
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme Corp", "https://x/1")
    tracker.upsert_job("job2", "Frontend Engineer", "Acme Robotics", "https://x/2")

    cmd_export(settings, export_args(company="Acme Corp"))

    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert [r["job_id"] for r in rows] == ["job1"]


def test_export_company_matches_case_and_spacing_insensitively(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "  ACME   corp  ", "https://x/1")

    cmd_export(settings, export_args(company="Acme Corp"))

    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert [r["job_id"] for r in rows] == ["job1"]


def test_export_company_combines_with_status_and_eligibility(tmp_path, capsys):
    """--company is applied as a post-fetch filter on top of Tracker.list_jobs'
    own status/eligibility filtering (the same pattern --stale-days already
    uses) - this confirms the two genuinely AND together rather than one
    silently overriding the other, the exact kind of "wiring exists but was
    never proven to actually combine" gap this session has repeatedly
    caught for other filter combinations.
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score(
        "job1", "Backend Engineer", "Acme Corp", "https://x/1", score=90, should_apply=True, eligibility="pass"
    )
    tracker.record_score(
        "job2", "Frontend Engineer", "Acme Corp", "https://x/2", score=20, should_apply=False, eligibility="fail"
    )
    tracker.record_score(
        "job3",
        "DevOps Engineer",
        "Beta Inc",
        "https://x/3",
        score=90,
        should_apply=True,
        eligibility="pass",
    )

    cmd_export(settings, export_args(company="Acme Corp", eligibility="pass"))

    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert [r["job_id"] for r in rows] == ["job1"]


def test_export_filters_by_stale_days(tmp_path, capsys):
    """--stale-days is the row-level counterpart to `job-bot report
    --stale-days`'s own list section - useful for a script that wants to
    act on the stale set (send follow-up reminders, say), not just see a
    count.
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.mark_applied("job1")
    _backdate_applied_at(settings.db_path, "job1", datetime.now(UTC) - timedelta(days=20))
    tracker.upsert_job("job2", "Frontend Engineer", "Beta", "https://x/2")
    tracker.mark_applied("job2")  # applied just now - not stale
    tracker.upsert_job("job3", "DevOps Engineer", "Gamma", "https://x/3")  # never applied at all

    cmd_export(settings, export_args(stale_days=14))

    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert [r["job_id"] for r in rows] == ["job1"]


def test_export_stale_days_combines_with_status_and_eligibility(tmp_path, capsys):
    """--stale-days already implies status="applied" (see
    _stale_applications()), but this confirms it still ANDs correctly
    with an explicit --status/--eligibility rather than one silently
    overriding the other.
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score(
        "job1", "Backend Engineer", "Acme", "https://x/1", score=90, should_apply=True, eligibility="pass"
    )
    tracker.mark_applied("job1")
    _backdate_applied_at(settings.db_path, "job1", datetime.now(UTC) - timedelta(days=20))

    cmd_export(settings, export_args(status="skipped", stale_days=14))

    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert rows == []


def test_export_filters_by_search(tmp_path, capsys):
    """--search matches the dashboard's own search box (Tracker.list_jobs'
    `search`) - the dashboard's /api/export.csv already respected it, but
    the CLI command had no equivalent way to export a search result
    instead of a full status-filtered dump.
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.upsert_job("job2", "Frontend Engineer", "Beta", "https://x/2")

    cmd_export(settings, export_args(status=None, search="Frontend", out=None, format="csv"))

    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert [r["job_id"] for r in rows] == ["job2"]


def test_export_search_also_matches_notes(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.upsert_job("job2", "Frontend Engineer", "Beta", "https://x/2")
    tracker.set_note("job1", "Referred by Jane.")

    cmd_export(settings, export_args(status=None, search="Jane", out=None, format="csv"))

    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert [r["job_id"] for r in rows] == ["job1"]


def test_export_to_file_writes_csv_and_reports_count(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    out_path = tmp_path / "export.csv"

    cmd_export(settings, export_args(status=None, search=None, out=out_path, format="csv"))

    assert f"Exported 1 job(s) to {out_path}" in capsys.readouterr().out
    rows = list(csv.DictReader(out_path.open(encoding="utf-8")))
    assert rows[0]["job_id"] == "job1"


def test_export_with_no_jobs_writes_header_only(tmp_path, capsys):
    settings = make_settings(tmp_path)
    Tracker(settings.db_path)

    cmd_export(settings, export_args(status=None, search=None, out=None, format="csv"))

    lines = capsys.readouterr().out.strip("\r\n").splitlines()
    assert len(lines) == 1
    assert lines[0].split(",")[0] == "job_id"


def test_export_json_to_stdout_is_a_json_array_with_all_jobs(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1", match_score=80)
    tracker.upsert_job("job2", "Frontend Engineer", "Beta", "https://x/2", match_score=60)
    tracker.mark_applied("job2")

    cmd_export(settings, export_args(status=None, search=None, out=None, format="json"))

    rows = json.loads(capsys.readouterr().out)
    assert [r["job_id"] for r in rows] == ["job1", "job2"]
    assert rows[1]["status"] == "applied"
    assert rows[1]["applied_at"]


def test_export_json_includes_notes(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.set_note("job1", "Recruiter mentioned $150k base.")

    cmd_export(settings, export_args(status=None, search=None, out=None, format="json"))

    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["notes"] == "Recruiter mentioned $150k base."


def test_export_json_includes_missing_qualifications_as_a_real_array(tmp_path, capsys):
    """Unlike CSV, the JSON export decodes the stored JSON-encoded string
    back into a real nested array (see write_export_json's docstring) -
    otherwise a script consuming the export would have to json.loads() a
    string-within-JSON a second time itself.
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score(
        "job1",
        "Backend Engineer",
        "Acme",
        "https://x/1",
        score=70,
        should_apply=True,
        missing_qualifications=["AWS certification"],
    )

    cmd_export(settings, export_args(status=None, search=None, out=None, format="json"))

    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["missing_qualifications"] == ["AWS certification"]


def test_export_json_filters_by_status(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.upsert_job("job2", "Frontend Engineer", "Beta", "https://x/2")
    tracker.mark_applied("job2")

    cmd_export(settings, export_args(status="applied", search=None, out=None, format="json"))

    rows = json.loads(capsys.readouterr().out)
    assert [r["job_id"] for r in rows] == ["job2"]


def test_export_json_to_file_writes_json_and_reports_count(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    out_path = tmp_path / "export.json"

    cmd_export(settings, export_args(status=None, search=None, out=out_path, format="json"))

    assert f"Exported 1 job(s) to {out_path}" in capsys.readouterr().out
    rows = json.loads(out_path.read_text(encoding="utf-8"))
    assert rows[0]["job_id"] == "job1"


def test_export_json_with_no_jobs_writes_empty_array(tmp_path, capsys):
    settings = make_settings(tmp_path)
    Tracker(settings.db_path)

    cmd_export(settings, export_args(status=None, search=None, out=None, format="json"))

    assert json.loads(capsys.readouterr().out) == []


# --- doctor ---


def test_doctor_flags_missing_resume_and_passes_api_key_check(tmp_path, capsys):
    settings = make_settings(tmp_path)  # resume_path points at a file that was never created

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] Resume file" in out
    assert "[OK] Anthropic API key" in out
    assert "checks passed." in out


def test_doctor_passes_resume_check_once_the_file_exists(tmp_path, capsys):
    settings = make_settings(tmp_path)
    settings.resume_path.write_text("resume", encoding="utf-8")

    cmd_doctor(settings, doctor_args())

    assert "[OK] Resume file" in capsys.readouterr().out


def test_doctor_flags_a_present_but_unparseable_resume(tmp_path, capsys):
    """Real failure this guards against: a resume file that exists (passing
    the old exists()-only check) but can't actually be parsed - here an
    empty resume.txt (see resume/parser.py) - used to only surface once
    `job-bot run` was already underway, well past `doctor` giving it a
    clean bill of health.
    """
    settings = make_settings(tmp_path)
    settings.resume_path.write_text("", encoding="utf-8")

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] Resume file" in out
    assert "No extractable text found" in out


def test_doctor_flags_a_resume_with_an_unsupported_extension(tmp_path, capsys):
    settings = make_settings(tmp_path, resume_path=tmp_path / "resume.rtf")
    settings.resume_path.write_text("resume", encoding="utf-8")

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] Resume file" in out
    assert "Unsupported resume format" in out


def test_doctor_passes_blacklist_check_when_no_blacklist_file_exists(tmp_path, capsys):
    settings = make_settings(tmp_path)  # blacklist_path points at a file that was never created

    cmd_doctor(settings, doctor_args())

    assert "[OK] Blacklist file valid" in capsys.readouterr().out


def test_doctor_flags_a_blacklist_path_whose_directory_cannot_be_created(tmp_path, capsys):
    """Real failure this guards against: BLACKLIST_PATH pointing at a
    directory that can't be created (e.g. a regular file already
    occupying that path) previously passed this check silently (a missing
    file always returned OK, with no writability probe) - the first
    `job-bot blacklist add` (or a dashboard one-click blacklist) would
    then fail deep in CompanyBlacklist._save(), well past `job-bot
    doctor` giving a clean bill of health.
    """
    blocking_file = tmp_path / "blacklist_dir"
    blocking_file.write_text("not a directory", encoding="utf-8")
    settings = make_settings(tmp_path, blacklist_path=blocking_file / "nested" / "blacklist.json")

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] Blacklist file valid" in out


def test_doctor_passes_blacklist_check_with_a_real_blacklist_file(tmp_path, capsys):
    settings = make_settings(tmp_path)
    settings.blacklist_path.write_text('["Acme Corp"]', encoding="utf-8")

    cmd_doctor(settings, doctor_args())

    assert "[OK] Blacklist file valid" in capsys.readouterr().out


def test_doctor_flags_a_corrupted_blacklist_file(tmp_path, capsys):
    """Real failure this guards against: CompanyBlacklist._load()
    (safety/blacklist.py) silently falls back to an empty blacklist on
    invalid JSON rather than raising - `job-bot run` would then proceed
    with zero blacklist protection and no visible error, and the next
    `job-bot blacklist add` would overwrite the file with only the newly
    added company, permanently losing every previously blacklisted one.
    """
    settings = make_settings(tmp_path)
    settings.blacklist_path.write_text("not valid json {{{", encoding="utf-8")

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] Blacklist file valid" in out
    assert "not valid JSON" in out


def test_doctor_flags_a_non_utf8_blacklist_file(tmp_path, capsys):
    """Real failure this guards against: a company_blacklist.json saved
    with a non-UTF-8 encoding previously crashed this check (and every
    other command touching the blacklist) with a raw, uncaught
    UnicodeDecodeError, the one corruption shape _blacklist_check's own
    OSError/JSONDecodeError handling didn't cover.
    """
    settings = make_settings(tmp_path)
    settings.blacklist_path.write_bytes("Acme ’s Corp".encode("cp1252"))

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] Blacklist file valid" in out
    assert "not valid UTF-8" in out


def test_doctor_flags_an_unreadable_blacklist_file(tmp_path, capsys):
    """Real gap this guards against: _blacklist_check's `except OSError`
    branch (e.g. a permission-denied file) is separate from - and was
    never exercised by - the JSONDecodeError test above. A directory
    where the blacklist file is expected exists (.exists() is True) but
    read_text() raises IsADirectoryError, a subclass of OSError - a
    portable, deterministic way to trigger this branch without relying on
    chmod semantics that can vary by OS/filesystem/user.
    """
    settings = make_settings(tmp_path)
    settings.blacklist_path.mkdir(parents=True)

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] Blacklist file valid" in out
    assert str(settings.blacklist_path) in out


def test_doctor_flags_a_blacklist_file_that_is_not_a_json_array(tmp_path, capsys):
    settings = make_settings(tmp_path)
    settings.blacklist_path.write_text('{"not": "a list"}', encoding="utf-8")

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] Blacklist file valid" in out
    assert "must be a JSON array" in out


def test_doctor_passes_faq_check_when_no_faq_file_exists(tmp_path, capsys):
    settings = make_settings(tmp_path)  # faq_path points at a file that was never created

    cmd_doctor(settings, doctor_args())

    assert "[OK] FAQ cache valid" in capsys.readouterr().out


def test_doctor_flags_a_faq_path_whose_directory_cannot_be_created(tmp_path, capsys):
    """Same reasoning as _blacklist_check's own equivalent test - a missing
    FAQ_PATH whose directory can't be created previously passed this check
    silently, with no writability probe.
    """
    blocking_file = tmp_path / "faq_dir"
    blocking_file.write_text("not a directory", encoding="utf-8")
    settings = make_settings(tmp_path, faq_path=blocking_file / "nested" / "faq.json")

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] FAQ cache valid" in out


def test_doctor_passes_faq_check_with_a_real_faq_file(tmp_path, capsys):
    settings = make_settings(tmp_path)
    settings.faq_path.write_text('{"Years of Python experience?": "5"}', encoding="utf-8")

    cmd_doctor(settings, doctor_args())

    assert "[OK] FAQ cache valid" in capsys.readouterr().out


def test_doctor_flags_a_corrupted_faq_file(tmp_path, capsys):
    settings = make_settings(tmp_path)
    settings.faq_path.write_text("not valid json {{{", encoding="utf-8")

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] FAQ cache valid" in out
    assert "not valid JSON" in out


def test_doctor_flags_a_non_utf8_faq_file(tmp_path, capsys):
    """Same non-UTF-8 corruption gap as _blacklist_check's - see
    test_doctor_flags_a_non_utf8_blacklist_file.
    """
    settings = make_settings(tmp_path)
    settings.faq_path.write_bytes("Isn’t sponsorship needed?".encode("cp1252"))

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] FAQ cache valid" in out
    assert "not valid UTF-8" in out


def test_doctor_flags_an_unreadable_faq_file(tmp_path, capsys):
    """Same OSError-branch gap as _blacklist_check's - see
    test_doctor_flags_an_unreadable_blacklist_file for why a directory in
    place of the file is the portable way to trigger it.
    """
    settings = make_settings(tmp_path)
    settings.faq_path.mkdir(parents=True)

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] FAQ cache valid" in out
    assert str(settings.faq_path) in out


def test_doctor_flags_a_faq_file_that_is_not_a_json_object(tmp_path, capsys):
    settings = make_settings(tmp_path)
    settings.faq_path.write_text("[1, 2, 3]", encoding="utf-8")

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] FAQ cache valid" in out
    assert "must be a JSON object" in out


def test_doctor_passes_answer_gaps_check_when_no_answer_gaps_file_exists(tmp_path, capsys):
    settings = make_settings(tmp_path)  # answer_gaps_path points at a file that was never created

    cmd_doctor(settings, doctor_args())

    assert "[OK] Answer-gaps file valid" in capsys.readouterr().out


def test_doctor_flags_an_answer_gaps_path_whose_directory_cannot_be_created(tmp_path, capsys):
    """Same reasoning as _blacklist_check's own equivalent test - a missing
    ANSWER_GAPS_PATH whose directory can't be created previously passed
    this check silently, with no writability probe.
    """
    blocking_file = tmp_path / "answer_gaps_dir"
    blocking_file.write_text("not a directory", encoding="utf-8")
    settings = make_settings(tmp_path, answer_gaps_path=blocking_file / "nested" / "answer_gaps.json")

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] Answer-gaps file valid" in out


def test_doctor_passes_answer_gaps_check_with_a_real_answer_gaps_file(tmp_path, capsys):
    settings = make_settings(tmp_path)
    settings.answer_gaps_path.write_text(
        '{"Are you comfortable commuting?": {"count": 3}}', encoding="utf-8"
    )

    cmd_doctor(settings, doctor_args())

    assert "[OK] Answer-gaps file valid" in capsys.readouterr().out


def test_doctor_flags_a_corrupted_answer_gaps_file(tmp_path, capsys):
    """Real failure this guards against: AnswerGapStore._load()
    (safety/answer_gaps.py) silently falls back to an empty store on
    invalid JSON rather than raising - the very next unanswered question
    job-bot run hits would call record(), which loads (getting {}),
    mutates, and saves the whole file - silently overwriting it with just
    that one new entry and permanently losing every previously recorded
    gap `job-bot review-answers` had queued up.
    """
    settings = make_settings(tmp_path)
    settings.answer_gaps_path.write_text("not valid json {{{", encoding="utf-8")

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] Answer-gaps file valid" in out
    assert "not valid JSON" in out


def test_doctor_flags_a_non_utf8_answer_gaps_file(tmp_path, capsys):
    """Same non-UTF-8 corruption gap as _blacklist_check's/_faq_check's -
    see test_doctor_flags_a_non_utf8_blacklist_file.
    """
    settings = make_settings(tmp_path)
    settings.answer_gaps_path.write_bytes("Isn’t sponsorship needed?".encode("cp1252"))

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] Answer-gaps file valid" in out
    assert "not valid UTF-8" in out


def test_doctor_flags_an_unreadable_answer_gaps_file(tmp_path, capsys):
    """Same OSError-branch gap as _blacklist_check's/_faq_check's - see
    test_doctor_flags_an_unreadable_blacklist_file for why a directory in
    place of the file is the portable way to trigger it.
    """
    settings = make_settings(tmp_path)
    settings.answer_gaps_path.mkdir(parents=True)

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] Answer-gaps file valid" in out
    assert str(settings.answer_gaps_path) in out


def test_doctor_flags_an_answer_gaps_file_that_is_not_a_json_object(tmp_path, capsys):
    settings = make_settings(tmp_path)
    settings.answer_gaps_path.write_text("[1, 2, 3]", encoding="utf-8")

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] Answer-gaps file valid" in out
    assert "must be a JSON object" in out


def test_doctor_creates_and_passes_the_applications_dir_check_when_missing(tmp_path, capsys):
    """Unlike the resume/blacklist/FAQ/answer-gaps checks, APPLICATIONS_DIR
    not existing yet is the normal case on a fresh install - generate_
    materials() (cmd_run) creates it on first use via mkdir(parents=True,
    exist_ok=True), so this check does the same rather than failing on
    something `job-bot run` itself would just create.
    """
    settings = make_settings(tmp_path)
    assert not settings.applications_dir.exists()

    cmd_doctor(settings, doctor_args())

    assert "[OK] Applications directory writable" in capsys.readouterr().out
    assert settings.applications_dir.is_dir()


def test_doctor_passes_the_applications_dir_check_when_it_already_exists(tmp_path, capsys):
    settings = make_settings(tmp_path)
    settings.applications_dir.mkdir(parents=True)

    cmd_doctor(settings, doctor_args())

    assert "[OK] Applications directory writable" in capsys.readouterr().out


def test_doctor_flags_an_applications_dir_that_cannot_be_created(tmp_path, capsys):
    """Real failure this guards against: a misconfigured APPLICATIONS_DIR
    (e.g. pointing at a path a regular file already occupies) makes
    mkdir() raise - previously only surfaced as a confusing prep_error on
    the first posting worth applying to, well past `job-bot doctor` giving
    a clean bill of health.
    """
    blocking_file = tmp_path / "applications"
    blocking_file.write_text("not a directory", encoding="utf-8")
    settings = make_settings(tmp_path, applications_dir=blocking_file / "nested")

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] Applications directory writable" in out


def test_doctor_creates_and_passes_the_audit_log_check_when_missing(tmp_path, capsys):
    """Same reasoning as the applications-dir check above - AUDIT_LOG_PATH
    not existing yet is the normal case on a fresh install (AuditLogger
    creates it via mkdir(parents=True, exist_ok=True) on its own, but
    never opens the file itself until the first real log() call), so this
    check proves it's writable rather than failing on something normal.
    """
    settings = make_settings(tmp_path)
    assert not settings.audit_log_path.exists()

    cmd_doctor(settings, doctor_args())

    assert "[OK] Audit log writable" in capsys.readouterr().out
    assert settings.audit_log_path.exists()


def test_doctor_passes_the_audit_log_check_with_existing_log_entries(tmp_path, capsys):
    """Opening in append mode and closing without writing must never
    truncate or otherwise disturb an audit log that already has real
    entries in it.
    """
    settings = make_settings(tmp_path)
    settings.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    settings.audit_log_path.write_text('{"action": "applied"}\n', encoding="utf-8")

    cmd_doctor(settings, doctor_args())

    assert "[OK] Audit log writable" in capsys.readouterr().out
    assert settings.audit_log_path.read_text(encoding="utf-8") == '{"action": "applied"}\n'


def test_doctor_flags_an_audit_log_that_cannot_be_written(tmp_path, capsys):
    """Real failure this guards against: a misconfigured AUDIT_LOG_PATH
    (e.g. pointing at a path a regular file already occupies, or an
    unwritable directory) previously wasn't caught anywhere - the first
    real action `job-bot run` took would fail deep into a run trying to
    log it, well past `job-bot doctor` giving a clean bill of health.
    """
    blocking_file = tmp_path / "audit"
    blocking_file.write_text("not a directory", encoding="utf-8")
    settings = make_settings(tmp_path, audit_log_path=blocking_file / "nested" / "audit.log")

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] Audit log writable" in out


def test_doctor_creates_and_passes_the_failed_applications_log_check_when_missing(tmp_path, capsys):
    """Same reasoning as _audit_log_check's own equivalent test -
    FAILED_APPLICATIONS_LOG_PATH not existing yet is the normal case on a
    fresh install, so this check proves it's writable rather than failing
    on something normal.
    """
    settings = make_settings(tmp_path)
    assert not settings.failed_applications_log_path.exists()

    cmd_doctor(settings, doctor_args())

    assert "[OK] Failed-applications log writable" in capsys.readouterr().out
    assert settings.failed_applications_log_path.exists()


def test_doctor_passes_the_failed_applications_log_check_with_existing_log_entries(tmp_path, capsys):
    settings = make_settings(tmp_path)
    settings.failed_applications_log_path.parent.mkdir(parents=True, exist_ok=True)
    settings.failed_applications_log_path.write_text('{"action": "prep_error"}\n', encoding="utf-8")

    cmd_doctor(settings, doctor_args())

    assert "[OK] Failed-applications log writable" in capsys.readouterr().out
    assert settings.failed_applications_log_path.read_text(encoding="utf-8") == '{"action": "prep_error"}\n'


def test_doctor_flags_a_failed_applications_log_that_cannot_be_written(tmp_path, capsys):
    """Real failure this guards against: a misconfigured
    FAILED_APPLICATIONS_LOG_PATH (e.g. pointing at a path a regular file
    already occupies) previously wasn't caught anywhere - cmd_run
    constructs an AuditLogger for this path unconditionally on every run,
    so the first real search_error/prep_error/apply_error would fail deep
    into a run trying to log it, well past `job-bot doctor` giving a clean
    bill of health.
    """
    blocking_file = tmp_path / "failed"
    blocking_file.write_text("not a directory", encoding="utf-8")
    settings = make_settings(
        tmp_path, failed_applications_log_path=blocking_file / "nested" / "failed_applications.log"
    )

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] Failed-applications log writable" in out


def test_doctor_passes_tracker_db_check_when_no_db_file_exists(tmp_path, capsys):
    """Unlike the resume/blacklist/FAQ/answer-gaps checks, no db.sqlite3 yet
    is the normal case on a fresh install - job-bot run (or any other
    command) creates it fresh, so this must not flag a plain-missing file.
    """
    settings = make_settings(tmp_path)
    assert not settings.db_path.exists()

    cmd_doctor(settings, doctor_args())

    assert "[OK] Tracker database readable" in capsys.readouterr().out


def test_doctor_passes_tracker_db_check_with_a_real_database(tmp_path, capsys):
    settings = make_settings(tmp_path)
    Tracker(settings.db_path)  # creates a real, valid db.sqlite3

    cmd_doctor(settings, doctor_args())

    assert "[OK] Tracker database readable" in capsys.readouterr().out


def test_doctor_flags_a_corrupted_tracker_database(tmp_path, capsys):
    """Real failure this guards against: every other command opens the
    tracker eagerly (Tracker.__init__()'s _init_db() runs on every
    construction), so a corrupted db.sqlite3 - here, a plain text file
    where a SQLite database should be - previously only surfaced as a raw
    sqlite3.DatabaseError from whichever command happened to run first,
    well past `job-bot doctor` giving a clean bill of health.
    """
    settings = make_settings(tmp_path)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.db_path.write_text("not a sqlite database", encoding="utf-8")

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] Tracker database readable" in out


def test_doctor_flags_missing_anthropic_api_key(tmp_path, capsys):
    settings = make_settings(tmp_path, anthropic_api_key=None)

    cmd_doctor(settings, doctor_args())

    assert "[!!] Anthropic API key" in capsys.readouterr().out


def test_doctor_checks_ollama_base_url_when_using_ollama(tmp_path, capsys):
    settings = make_settings(tmp_path, llm_provider="ollama", anthropic_api_key=None)

    cmd_doctor(settings, doctor_args())

    assert "[OK] Ollama base URL configured" in capsys.readouterr().out


def test_doctor_flags_missing_gmail_credentials(tmp_path, capsys):
    settings = make_settings(tmp_path)  # gmail_credentials_path points at a file never created

    cmd_doctor(settings, doctor_args())

    assert "[!!] Gmail credentials (optional, for gmail-sync)" in capsys.readouterr().out


def test_doctor_passes_gmail_credentials_check_with_a_real_file(tmp_path, capsys):
    settings = make_settings(tmp_path)
    settings.gmail_credentials_path.write_text("{}", encoding="utf-8")

    cmd_doctor(settings, doctor_args())

    assert "[OK] Gmail credentials (optional, for gmail-sync)" in capsys.readouterr().out


def test_doctor_flags_missing_gmail_authorization(tmp_path, capsys):
    """Real gap this guards against: gmail_credentials_path existing only
    means gmail-sync *can* be set up, not that the one-time OAuth consent
    (which writes gmail_token_path - see gmail_client.py) was ever actually
    completed - doctor previously gave no way to tell "never configured"
    apart from "configured but never authorized".
    """
    settings = make_settings(tmp_path)
    settings.gmail_credentials_path.write_text("{}", encoding="utf-8")  # credentials set up...

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[OK] Gmail credentials (optional, for gmail-sync)" in out
    assert "[!!] Gmail authorized (optional, run `job-bot gmail-sync` once)" in out  # ...but never authorized


def test_doctor_passes_gmail_authorization_check_with_a_real_token(tmp_path, capsys):
    settings = make_settings(tmp_path)
    settings.gmail_credentials_path.write_text("{}", encoding="utf-8")
    settings.gmail_token_path.write_text("{}", encoding="utf-8")

    cmd_doctor(settings, doctor_args())

    assert "[OK] Gmail authorized (optional, run `job-bot gmail-sync` once)" in capsys.readouterr().out


def test_doctor_flags_daily_cap_above_the_hard_ceiling(tmp_path, capsys):
    settings = make_settings(tmp_path, daily_application_cap=999)

    cmd_doctor(settings, doctor_args())

    assert "[!!] Daily application cap within hard ceiling" in capsys.readouterr().out


def test_doctor_passes_daily_cap_usage_check_with_no_applications_today(tmp_path, capsys):
    settings = make_settings(tmp_path, daily_application_cap=5)

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[OK] Daily application cap available" in out
    assert "0/5 applications submitted today" in out


def test_doctor_shows_partial_daily_cap_usage(tmp_path, capsys):
    settings = make_settings(tmp_path, daily_application_cap=5)
    rate_limiter = RateLimiter(settings.db_path, settings.effective_daily_cap())
    rate_limiter.record_application()
    rate_limiter.record_application()

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[OK] Daily application cap available" in out
    assert "2/5 applications submitted today" in out


def test_doctor_flags_daily_cap_usage_check_when_cap_reached(tmp_path, capsys):
    """Real gap this guards against: someone running `job-bot doctor` to
    understand why `job-bot run` isn't applying to anything (or a fresh
    --loop stops almost immediately) previously had no way to see "you've
    already used today's cap" without starting a real run and watching it
    apply to nothing.
    """
    settings = make_settings(tmp_path, daily_application_cap=1)
    rate_limiter = RateLimiter(settings.db_path, settings.effective_daily_cap())
    rate_limiter.record_application()

    cmd_doctor(settings, doctor_args())

    out = capsys.readouterr().out
    assert "[!!] Daily application cap available" in out
    assert "1/1 applications submitted today" in out
    assert "job-bot run will apply to nothing more until tomorrow" in out


def test_doctor_daily_cap_usage_check_does_not_crash_on_a_corrupted_tracker_database(tmp_path, capsys):
    """RateLimiter shares db_path with Tracker but has no guard of its own
    against a corrupted file - without this check's own try/except, a
    corrupted db.sqlite3 crashed doctor entirely on this check (a raw
    sqlite3.DatabaseError) instead of the clean diagnosis
    _tracker_db_check already gives for the same root cause. Only one
    failed check should be reported for it, not two.
    """
    settings = make_settings(tmp_path)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.db_path.write_text("not a sqlite database", encoding="utf-8")

    cmd_doctor(settings, doctor_args())  # must not raise

    out = capsys.readouterr().out
    assert "[OK] Daily application cap available" in out
    assert "[!!] Tracker database readable" in out


def test_doctor_json_reports_each_check_and_the_pass_count(tmp_path, capsys):
    settings = make_settings(tmp_path)  # resume_path points at a file that was never created

    cmd_doctor(settings, doctor_args(format="json"))

    payload = json.loads(capsys.readouterr().out)
    by_label = {c["label"]: c for c in payload["checks"]}
    assert by_label["Resume file readable"]["ok"] is False
    assert by_label["Anthropic API key (ANTHROPIC_API_KEY)"]["ok"] is True
    assert payload["total"] == len(payload["checks"])
    assert payload["passed"] == sum(c["ok"] for c in payload["checks"])


def test_doctor_json_passed_count_matches_a_fully_healthy_setup(tmp_path, capsys):
    settings = make_settings(tmp_path)
    settings.resume_path.write_text("resume", encoding="utf-8")
    settings.browser_profile_dir.mkdir(parents=True)
    (settings.browser_profile_dir / "placeholder").write_text("x")

    cmd_doctor(settings, doctor_args(format="json"))

    payload = json.loads(capsys.readouterr().out)
    # Gmail credentials and Gmail authorization (both optional) still missing
    assert payload["passed"] == payload["total"] - 2


# --- review-answers ---


def test_review_answers_says_so_when_nothing_to_review(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_review_answers(settings, review_answers_args())

    assert "nothing to review" in capsys.readouterr().out


def test_review_answers_search_narrows_which_gaps_are_prompted(tmp_path, monkeypatch, capsys):
    """The same --search semantics `faq list --search`/`qa-history
    --search` already use - useful once there are enough recorded gaps
    that reviewing them all in frequency order isn't the fastest way to
    find a specific one.
    """
    settings = make_settings(tmp_path)
    store = AnswerGapStore(settings.answer_gaps_path)
    store.record("Are you willing to sponsor... wait, are YOU sponsored?", job_id="1", company="Acme", title="X")
    store.record("Willing to relocate?", job_id="2", company="Acme", title="X")
    monkeypatch.setattr("builtins.input", lambda prompt: "")

    cmd_review_answers(settings, review_answers_args(search="sponsor"))

    out = capsys.readouterr().out
    assert "sponsor" in out.lower()
    assert "relocate" not in out.lower()


def test_review_answers_search_says_so_when_nothing_matches(tmp_path, capsys):
    settings = make_settings(tmp_path)
    AnswerGapStore(settings.answer_gaps_path).record("Willing to relocate?", job_id="1", company="Acme", title="X")

    cmd_review_answers(settings, review_answers_args(search="cobol"))

    assert 'No unanswered required questions matching "cobol".' in capsys.readouterr().out


def test_review_answers_format_json_respects_search(tmp_path, capsys):
    settings = make_settings(tmp_path)
    store = AnswerGapStore(settings.answer_gaps_path)
    store.record("Sponsorship required?", job_id="1", company="Acme", title="X")
    store.record("Willing to relocate?", job_id="2", company="Acme", title="X")

    cmd_review_answers(settings, review_answers_args(search="sponsor", format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert [gap["question"] for gap in payload] == ["Sponsorship required?"]


def test_review_answers_saves_a_given_answer_to_faq_and_resolves_the_gap(tmp_path, monkeypatch, capsys):
    """The whole point of the mechanism: an answer given here must (a) land
    in FAQ_PATH, where qa_answerer.py's prompt picks it up as context for
    every future posting that asks the same question, and (b) stop showing
    up as a gap to review again.
    """
    settings = make_settings(tmp_path)
    question = "Are you comfortable commuting to this job's location?"
    AnswerGapStore(settings.answer_gaps_path).record(
        question, job_id="1", company="Acme", title="Backend Engineer"
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "Yes")

    cmd_review_answers(settings, review_answers_args())

    faq = ResumeStore(settings.resume_path, settings.faq_path).faq_answers()
    assert faq[question] == "Yes"
    assert AnswerGapStore(settings.answer_gaps_path).list_unanswered() == {}
    assert "Answered 1 question(s). 0 still unanswered." in capsys.readouterr().out


def test_review_answers_leaves_a_skipped_question_as_a_gap(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    question = "Are you comfortable commuting to this job's location?"
    AnswerGapStore(settings.answer_gaps_path).record(
        question, job_id="1", company="Acme", title="Backend Engineer"
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "")  # blank = skip

    cmd_review_answers(settings, review_answers_args())

    faq = ResumeStore(settings.resume_path, settings.faq_path).faq_answers()
    assert question not in faq
    assert question in AnswerGapStore(settings.answer_gaps_path).list_unanswered()


def test_review_answers_orders_most_frequently_seen_first(tmp_path, monkeypatch, capsys):
    settings = make_settings(tmp_path)
    store = AnswerGapStore(settings.answer_gaps_path)
    store.record("Rare question", job_id="1", company="Acme", title="X")
    for job_id in ("2", "3", "4"):
        store.record("Common question", job_id=job_id, company="Acme", title="X")
    monkeypatch.setattr("builtins.input", lambda prompt: "")

    cmd_review_answers(settings, review_answers_args())

    out = capsys.readouterr().out
    assert out.index("Common question") < out.index("Rare question")


def test_review_answers_stops_cleanly_on_keyboard_interrupt_mid_review(tmp_path, monkeypatch, capsys):
    settings = make_settings(tmp_path)
    store = AnswerGapStore(settings.answer_gaps_path)
    store.record("Question one", job_id="1", company="Acme", title="X")
    store.record("Question two", job_id="2", company="Acme", title="X")

    def raise_interrupt(prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", raise_interrupt)

    cmd_review_answers(settings, review_answers_args())  # must not raise

    assert len(AnswerGapStore(settings.answer_gaps_path).list_unanswered()) == 2


def test_review_answers_json_format_lists_gaps_without_prompting(tmp_path, monkeypatch, capsys):
    """--format json must never call input() - a monitoring script running
    this unattended (cron, no terminal) needs a plain data dump, not a
    prompt that would just hit EOF the same way the interactive mode
    already handles that case.
    """
    settings = make_settings(tmp_path)
    AnswerGapStore(settings.answer_gaps_path).record(
        "Are you comfortable commuting?", job_id="1", company="Acme", title="Backend Engineer"
    )

    def fail_if_called(prompt):
        raise AssertionError("input() must not be called in --format json mode")

    monkeypatch.setattr("builtins.input", fail_if_called)

    cmd_review_answers(settings, review_answers_args(format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["question"] == "Are you comfortable commuting?"
    assert payload[0]["count"] == 1
    assert payload[0]["example_company"] == "Acme"


def test_review_answers_json_format_orders_most_frequently_seen_first(tmp_path, capsys):
    settings = make_settings(tmp_path)
    store = AnswerGapStore(settings.answer_gaps_path)
    store.record("Rare question", job_id="1", company="Acme", title="X")
    for job_id in ("2", "3", "4"):
        store.record("Common question", job_id=job_id, company="Acme", title="X")

    cmd_review_answers(settings, review_answers_args(format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert [entry["question"] for entry in payload] == ["Common question", "Rare question"]


def test_review_answers_json_format_on_empty_gaps_is_an_empty_array(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_review_answers(settings, review_answers_args(format="json"))

    assert json.loads(capsys.readouterr().out) == []


def test_review_answers_dismiss_removes_the_gap_without_touching_faq(tmp_path, monkeypatch, capsys):
    """Real gap this closes: leaving the interactive prompt blank only
    skips a gap for this run - it keeps resurfacing every time this
    command runs. --dismiss is the only way to permanently discard a
    noise/duplicate/garbled gap without inventing a throwaway FAQ answer
    just to make it stop showing up.
    """
    settings = make_settings(tmp_path)
    question = "Are you comfortable commuting to this job's location?"
    AnswerGapStore(settings.answer_gaps_path).record(
        question, job_id="1", company="Acme", title="Backend Engineer"
    )

    def fail_if_called(prompt):
        raise AssertionError("input() must not be called when --dismiss is given")

    monkeypatch.setattr("builtins.input", fail_if_called)

    cmd_review_answers(settings, review_answers_args(dismiss=[question]))

    assert AnswerGapStore(settings.answer_gaps_path).list_unanswered() == {}
    assert ResumeStore(settings.resume_path, settings.faq_path).faq_answers() == {}
    assert f'Dismissed: "{question}"' in capsys.readouterr().out


def test_review_answers_dismiss_accepts_multiple_questions_in_one_call(tmp_path, capsys):
    settings = make_settings(tmp_path)
    store = AnswerGapStore(settings.answer_gaps_path)
    store.record("Question one", job_id="1", company="Acme", title="X")
    store.record("Question two", job_id="2", company="Acme", title="X")

    cmd_review_answers(settings, review_answers_args(dismiss=["Question one", "Question two"]))

    assert AnswerGapStore(settings.answer_gaps_path).list_unanswered() == {}


def test_review_answers_dismiss_of_a_nonexistent_question_says_so(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_review_answers(settings, review_answers_args(dismiss=["Never recorded"]))

    assert 'No unanswered gap matching: "Never recorded"' in capsys.readouterr().out


# --- faq ---


def test_faq_list_says_so_when_nothing_cached(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_faq(settings, argparse.Namespace(faq_action="list", question=None, search=None, format="text"))

    assert "No cached FAQ answers." in capsys.readouterr().out


def test_faq_list_prints_every_cached_question_and_answer(tmp_path, capsys):
    settings = make_settings(tmp_path)
    store = ResumeStore(settings.resume_path, settings.faq_path)
    store.save_faq_answer("Years of Python experience?", "5")
    store.save_faq_answer("Willing to relocate?", "No")

    cmd_faq(settings, argparse.Namespace(faq_action="list", question=None, search=None, format="text"))

    out = capsys.readouterr().out
    assert "Years of Python experience?" in out
    assert "5" in out
    assert "Willing to relocate?" in out
    assert "No" in out


def test_faq_list_search_matches_the_question_text(tmp_path, capsys):
    settings = make_settings(tmp_path)
    store = ResumeStore(settings.resume_path, settings.faq_path)
    store.save_faq_answer("Years of Python experience?", "5")
    store.save_faq_answer("Willing to relocate?", "No")

    cmd_faq(settings, argparse.Namespace(faq_action="list", question=None, search="python", format="text"))

    out = capsys.readouterr().out
    assert "Years of Python experience?" in out
    assert "Willing to relocate?" not in out


def test_faq_list_search_matches_the_answer_text(tmp_path, capsys):
    """The same "search" semantics job-bot export/the dashboard already
    use: matches either side of the pair, not just the question - a cache
    large enough to need searching is exactly one where you might
    remember the gist of an answer but not the exact question wording
    that produced it.
    """
    settings = make_settings(tmp_path)
    store = ResumeStore(settings.resume_path, settings.faq_path)
    store.save_faq_answer("Years of Python experience?", "5")
    store.save_faq_answer("Willing to relocate?", "No")

    cmd_faq(settings, argparse.Namespace(faq_action="list", question=None, search="no", format="text"))

    out = capsys.readouterr().out
    assert "Willing to relocate?" in out
    assert "Years of Python experience?" not in out


def test_faq_list_search_is_case_insensitive(tmp_path, capsys):
    settings = make_settings(tmp_path)
    store = ResumeStore(settings.resume_path, settings.faq_path)
    store.save_faq_answer("Years of Python experience?", "5")

    cmd_faq(settings, argparse.Namespace(faq_action="list", question=None, search="PYTHON", format="text"))

    assert "Years of Python experience?" in capsys.readouterr().out


def test_faq_list_search_says_so_when_nothing_matches(tmp_path, capsys):
    settings = make_settings(tmp_path)
    store = ResumeStore(settings.resume_path, settings.faq_path)
    store.save_faq_answer("Years of Python experience?", "5")

    cmd_faq(settings, argparse.Namespace(faq_action="list", question=None, search="cobol", format="text"))

    assert 'No cached FAQ answers matching "cobol".' in capsys.readouterr().out


def test_faq_list_format_json_prints_the_full_cache_as_one_object(tmp_path, capsys):
    """`job-bot faq list --format json` gives a monitoring script the same
    structured-output option report/export/status/doctor/review-answers/
    gmail-sync all already have - the same reasoning cmd_report's docstring
    gives for --format json: scraping human-formatted text otherwise.
    """
    settings = make_settings(tmp_path)
    store = ResumeStore(settings.resume_path, settings.faq_path)
    store.save_faq_answer("Years of Python experience?", "5")
    store.save_faq_answer("Willing to relocate?", "No")

    cmd_faq(settings, argparse.Namespace(faq_action="list", question=None, search=None, format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"Years of Python experience?": "5", "Willing to relocate?": "No"}


def test_faq_list_format_json_respects_search(tmp_path, capsys):
    settings = make_settings(tmp_path)
    store = ResumeStore(settings.resume_path, settings.faq_path)
    store.save_faq_answer("Years of Python experience?", "5")
    store.save_faq_answer("Willing to relocate?", "No")

    cmd_faq(settings, argparse.Namespace(faq_action="list", question=None, search="python", format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"Years of Python experience?": "5"}


def test_faq_list_format_json_on_empty_cache_is_still_valid_json(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_faq(settings, argparse.Namespace(faq_action="list", question=None, search=None, format="json"))

    assert json.loads(capsys.readouterr().out) == {}


def test_faq_remove_deletes_a_cached_answer(tmp_path, capsys):
    settings = make_settings(tmp_path)
    store = ResumeStore(settings.resume_path, settings.faq_path)
    store.save_faq_answer("Years of Python experience?", "5")

    cmd_faq(settings, argparse.Namespace(faq_action="remove", question=["Years of Python experience?"]))

    assert 'Removed cached answer for: "Years of Python experience?"' in capsys.readouterr().out
    assert ResumeStore(settings.resume_path, settings.faq_path).faq_answers() == {}


def test_faq_remove_of_uncached_question_says_so(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_faq(settings, argparse.Namespace(faq_action="remove", question=["Never asked"]))

    assert 'No cached answer for: "Never asked"' in capsys.readouterr().out


def test_faq_remove_accepts_multiple_questions_in_one_call(tmp_path, capsys):
    settings = make_settings(tmp_path)
    store = ResumeStore(settings.resume_path, settings.faq_path)
    store.save_faq_answer("Years of Python experience?", "5")
    store.save_faq_answer("Willing to relocate?", "No")

    cmd_faq(
        settings,
        argparse.Namespace(
            faq_action="remove", question=["Years of Python experience?", "Willing to relocate?"]
        ),
    )

    out = capsys.readouterr().out
    assert 'Removed cached answer for: "Years of Python experience?"' in out
    assert 'Removed cached answer for: "Willing to relocate?"' in out
    assert ResumeStore(settings.resume_path, settings.faq_path).faq_answers() == {}


def test_faq_import_merges_answers_from_a_json_file(tmp_path, capsys):
    settings = make_settings(tmp_path)
    store = ResumeStore(settings.resume_path, settings.faq_path)
    store.save_faq_answer("Willing to relocate?", "No")
    import_file = tmp_path / "faq_backup.json"
    import_file.write_text(
        json.dumps({"Years of Python experience?": "5", "Willing to relocate?": "Yes"}), encoding="utf-8"
    )

    cmd_faq(settings, argparse.Namespace(faq_action="import", file=import_file))

    out = capsys.readouterr().out
    assert f"Imported 2 FAQ answer(s) from {import_file}." in out
    answers = ResumeStore(settings.resume_path, settings.faq_path).faq_answers()
    assert answers["Years of Python experience?"] == "5"
    assert answers["Willing to relocate?"] == "Yes"  # imported value wins over the existing one


def test_faq_import_of_missing_file_exits_with_error(tmp_path, capsys):
    settings = make_settings(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        cmd_faq(settings, argparse.Namespace(faq_action="import", file=tmp_path / "does-not-exist.json"))

    assert exc_info.value.code == 1
    assert "Error" in capsys.readouterr().err


def test_faq_import_of_invalid_json_exits_with_error(tmp_path, capsys):
    settings = make_settings(tmp_path)
    import_file = tmp_path / "bad.json"
    import_file.write_text("not valid json {{{", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        cmd_faq(settings, argparse.Namespace(faq_action="import", file=import_file))

    assert exc_info.value.code == 1
    assert "not valid JSON" in capsys.readouterr().err


def test_faq_import_of_a_non_utf8_file_exits_with_a_clean_error(tmp_path, capsys):
    """Same reasoning as blacklist import's own non-UTF-8 regression test -
    a --file saved with a non-UTF-8 encoding previously crashed this with
    a raw, uncaught UnicodeDecodeError instead of the same clean "could
    not read" message an unreadable/missing file already gets here.
    """
    settings = make_settings(tmp_path)
    bad_file = tmp_path / "bad.json"
    bad_file.write_bytes("Isn’t sponsorship needed?".encode("cp1252"))

    with pytest.raises(SystemExit) as exc_info:
        cmd_faq(settings, argparse.Namespace(faq_action="import", file=bad_file))

    assert exc_info.value.code == 1
    assert "Error: could not read" in capsys.readouterr().err


def test_faq_import_of_non_object_json_exits_with_error(tmp_path, capsys):
    settings = make_settings(tmp_path)
    import_file = tmp_path / "list.json"
    import_file.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        cmd_faq(settings, argparse.Namespace(faq_action="import", file=import_file))

    assert exc_info.value.code == 1
    assert "must contain a JSON object" in capsys.readouterr().err


def test_faq_export_to_stdout_prints_valid_json(tmp_path, capsys):
    settings = make_settings(tmp_path)
    store = ResumeStore(settings.resume_path, settings.faq_path)
    store.save_faq_answer("Willing to relocate?", "No")

    cmd_faq(settings, argparse.Namespace(faq_action="export", out=None))

    printed = json.loads(capsys.readouterr().out)
    assert printed == {"Willing to relocate?": "No"}


def test_faq_export_round_trips_through_import(tmp_path, capsys):
    settings = make_settings(tmp_path)
    store = ResumeStore(settings.resume_path, settings.faq_path)
    store.save_faq_answer("Willing to relocate?", "No")
    store.save_faq_answer("Years of Python experience?", "5")
    export_file = tmp_path / "backup.json"

    cmd_faq(settings, argparse.Namespace(faq_action="export", out=export_file))
    assert f"Exported 2 FAQ answer(s) to {export_file}." in capsys.readouterr().out

    other_settings = make_settings(tmp_path, faq_path=tmp_path / "other_faq.json")
    cmd_faq(other_settings, argparse.Namespace(faq_action="import", file=export_file))
    capsys.readouterr()
    answers = ResumeStore(other_settings.resume_path, other_settings.faq_path).faq_answers()
    assert answers == {"Willing to relocate?": "No", "Years of Python experience?": "5"}


def test_faq_export_of_empty_cache_prints_empty_object(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_faq(settings, argparse.Namespace(faq_action="export", out=None))

    assert json.loads(capsys.readouterr().out) == {}


# --- qa-history ---


def test_qa_history_says_so_when_nothing_recorded(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_qa_history(settings, qa_history_args())

    assert "No Q&A history recorded yet." in capsys.readouterr().out


def test_qa_history_prints_every_pair_with_job_context_most_recent_first(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.upsert_job("job2", "Frontend Engineer", "Beta", "https://x/2")
    tracker.record_qa("job1", "Years of experience?", "5")
    tracker.record_qa("job2", "Willing to relocate?", "No")

    cmd_qa_history(settings, qa_history_args())

    out = capsys.readouterr().out
    assert out.index("Willing to relocate?") < out.index("Years of experience?")
    assert "[job2] Beta - Frontend Engineer" in out
    assert "[job1] Acme - Backend Engineer" in out


def test_qa_history_search_matches_question_or_answer(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.record_qa("job1", "Years of Python experience?", "5")
    tracker.record_qa("job1", "Willing to relocate?", "No")

    cmd_qa_history(settings, qa_history_args(search="python"))

    out = capsys.readouterr().out
    assert "Years of Python experience?" in out
    assert "Willing to relocate?" not in out


def test_qa_history_search_says_so_when_nothing_matches(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.record_qa("job1", "Years of experience?", "5")

    cmd_qa_history(settings, qa_history_args(search="cobol"))

    assert 'No Q&A history matching "cobol".' in capsys.readouterr().out


def test_qa_history_company_is_an_exact_match_not_a_substring(tmp_path, capsys):
    """--company must not also pull "Acme Robotics", an unrelated company
    that happens to share the word "Acme" - unlike --search's fuzzy
    substring match.
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme Corp", "https://x/1")
    tracker.upsert_job("job2", "Frontend Engineer", "Acme Robotics", "https://x/2")
    tracker.record_qa("job1", "Years of experience?", "5")
    tracker.record_qa("job2", "Willing to relocate?", "No")

    cmd_qa_history(settings, qa_history_args(company="Acme Corp"))

    out = capsys.readouterr().out
    assert "Years of experience?" in out
    assert "Willing to relocate?" not in out


def test_qa_history_company_says_so_when_nothing_matches(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_qa_history(settings, qa_history_args(company="Nobody Inc"))

    assert 'No Q&A history matching company "Nobody Inc".' in capsys.readouterr().out


def test_qa_history_search_and_company_combine(tmp_path, capsys):
    """--search and --company are applied as two separate filters
    (search_qa()'s own SQL search, then a Python post-filter for
    --company) - this confirms they genuinely AND together rather than
    one silently overriding the other, the same kind of "wiring exists
    but was never proven to combine" gap this session has repeatedly
    caught for other filter pairs.
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme Corp", "https://x/1")
    tracker.upsert_job("job2", "Frontend Engineer", "Acme Corp", "https://x/2")
    tracker.upsert_job("job3", "DevOps Engineer", "Beta Inc", "https://x/3")
    tracker.record_qa("job1", "Years of Python experience?", "5")
    tracker.record_qa("job2", "Willing to relocate?", "No")
    tracker.record_qa("job3", "Years of Python experience?", "3")

    cmd_qa_history(settings, qa_history_args(search="python", company="Acme Corp"))

    out = capsys.readouterr().out
    assert "[job1]" in out
    assert "[job2]" not in out  # right company, wrong search term
    assert "[job3]" not in out  # right search term, wrong company


def test_qa_history_format_json_prints_pairs_with_job_context(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.record_qa("job1", "Years of experience?", "5")

    cmd_qa_history(settings, qa_history_args(format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "job_id": "job1",
            "company": "Acme",
            "title": "Backend Engineer",
            "question": "Years of experience?",
            "answer": "5",
            "created_at": payload[0]["created_at"],
        }
    ]


def test_qa_history_format_json_on_empty_history_is_still_valid_json(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_qa_history(settings, qa_history_args(format="json"))

    assert json.loads(capsys.readouterr().out) == []


# --- resume-history ---


def test_resume_history_says_so_when_nothing_recorded(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_resume_history(settings, resume_history_args())

    assert "No resume generations recorded yet." in capsys.readouterr().out


def test_resume_history_prints_every_generation_with_status_most_recent_first(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.upsert_job("job2", "Frontend Engineer", "Beta", "https://x/2")
    tracker.record_resume_generation("job1", "Backend Engineer", "Acme", "For job 1.", ["Python"], [])
    tracker.record_resume_generation("job2", "Frontend Engineer", "Beta", "For job 2.", ["React"], [])
    tracker.update_status("job1", "interviewing")

    cmd_resume_history(settings, resume_history_args())

    out = capsys.readouterr().out
    assert out.index("For job 2.") < out.index("For job 1.")
    assert "[job1] Acme - Backend Engineer (interviewing" in out
    assert "[job2] Beta - Frontend Engineer (seen" in out


def test_resume_history_shows_not_tracked_for_a_generation_with_no_matching_job(tmp_path, capsys):
    """record_resume_generation() has no foreign-key requirement that
    job_id already exist in `jobs` (list_resume_generations() uses a LEFT
    JOIN, not a plain JOIN, for exactly this reason) - such a row's status
    comes back None, printed as "not tracked" rather than a bare "None".
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_resume_generation("orphan", "Engineer", "Acme", "Orphaned.", [], [])

    cmd_resume_history(settings, resume_history_args())

    assert "[orphan] Acme - Engineer (not tracked" in capsys.readouterr().out


def test_resume_history_search_matches_summary_company_or_title(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_resume_generation("job1", "Backend Engineer", "Acme", "Python-heavy tailoring.", [], [])
    tracker.record_resume_generation("job2", "Frontend Engineer", "Beta", "React-focused tailoring.", [], [])

    cmd_resume_history(settings, resume_history_args(search="python"))

    out = capsys.readouterr().out
    assert "Python-heavy tailoring." in out
    assert "React-focused tailoring." not in out


def test_resume_history_search_says_so_when_nothing_matches(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_resume_generation("job1", "Backend Engineer", "Acme", "Some summary.", [], [])

    cmd_resume_history(settings, resume_history_args(search="cobol"))

    assert 'No resume generations matching "cobol".' in capsys.readouterr().out


def test_resume_history_company_is_an_exact_match_not_a_substring(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_resume_generation("job1", "Backend Engineer", "Acme Corp", "For job 1.", [], [])
    tracker.record_resume_generation("job2", "Frontend Engineer", "Acme Robotics", "For job 2.", [], [])

    cmd_resume_history(settings, resume_history_args(company="Acme Corp"))

    out = capsys.readouterr().out
    assert "For job 1." in out
    assert "For job 2." not in out


def test_resume_history_company_says_so_when_nothing_matches(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_resume_history(settings, resume_history_args(company="Nobody Inc"))

    assert 'No resume generations matching company "Nobody Inc".' in capsys.readouterr().out


def test_resume_history_search_and_company_combine(tmp_path, capsys):
    """Same AND-combination verification as
    test_qa_history_search_and_company_combine, for --search/--company's
    own separate filter passes here (list_resume_generations()'s SQL
    search, then a Python post-filter for --company).
    """
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_resume_generation("job1", "Backend Engineer", "Acme Corp", "Python-heavy.", [], [])
    tracker.record_resume_generation("job2", "Frontend Engineer", "Acme Corp", "React-focused.", [], [])
    tracker.record_resume_generation("job3", "DevOps Engineer", "Beta Inc", "Python-heavy.", [], [])

    cmd_resume_history(settings, resume_history_args(search="python", company="Acme Corp"))

    out = capsys.readouterr().out
    assert "[job1]" in out
    assert "[job2]" not in out  # right company, wrong search term
    assert "[job3]" not in out  # right search term, wrong company


def test_resume_history_format_json_prints_generations_with_status(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.record_resume_generation("job1", "Backend Engineer", "Acme", "A summary.", ["Python"], ["Did X."])

    cmd_resume_history(settings, resume_history_args(format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "job_id": "job1",
            "title": "Backend Engineer",
            "company": "Acme",
            "summary": "A summary.",
            "skills": ["Python"],
            "bullets": ["Did X."],
            "created_at": payload[0]["created_at"],
            "status": "seen",
        }
    ]


def test_resume_history_format_json_on_empty_history_is_still_valid_json(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_resume_history(settings, resume_history_args(format="json"))

    assert json.loads(capsys.readouterr().out) == []


# --- audit-log ---


def test_audit_log_says_so_when_nothing_recorded(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_audit_log(settings, audit_log_args())

    assert "No audit log entries recorded yet." in capsys.readouterr().out


def test_audit_log_prints_every_entry_most_recent_first(tmp_path, capsys):
    settings = make_settings(tmp_path)
    audit = AuditLogger(settings.audit_log_path)
    audit.log("scored", job_id="1", score=90)
    audit.log("applied", job_id="1", company="Acme")

    cmd_audit_log(settings, audit_log_args())

    out = capsys.readouterr().out
    assert out.index("applied") < out.index("scored")
    assert "job_id=1" in out
    assert "company=Acme" in out


def test_audit_log_search_matches_action_or_details(tmp_path, capsys):
    settings = make_settings(tmp_path)
    audit = AuditLogger(settings.audit_log_path)
    audit.log("applied", job_id="1", company="Acme Corp")
    audit.log("applied", job_id="2", company="Beta Inc")

    cmd_audit_log(settings, audit_log_args(search="acme"))

    out = capsys.readouterr().out
    assert "Acme Corp" in out
    assert "Beta Inc" not in out


def test_audit_log_search_says_so_when_nothing_matches(tmp_path, capsys):
    settings = make_settings(tmp_path)
    AuditLogger(settings.audit_log_path).log("applied", job_id="1", company="Acme")

    cmd_audit_log(settings, audit_log_args(search="nonexistent"))

    assert 'No audit log entries matching "nonexistent".' in capsys.readouterr().out


def test_audit_log_action_is_an_exact_match_not_a_substring(tmp_path, capsys):
    settings = make_settings(tmp_path)
    audit = AuditLogger(settings.audit_log_path)
    audit.log("applied", job_id="1")
    audit.log("dry_run_stopped", job_id="2")

    cmd_audit_log(settings, audit_log_args(action="applied"))

    out = capsys.readouterr().out
    assert "job_id=1" in out
    assert "job_id=2" not in out


def test_audit_log_action_says_so_when_nothing_matches(tmp_path, capsys):
    settings = make_settings(tmp_path)
    AuditLogger(settings.audit_log_path).log("applied", job_id="1")

    cmd_audit_log(settings, audit_log_args(action="nonexistent_action"))

    assert 'No audit log entries matching action "nonexistent_action".' in capsys.readouterr().out


def test_audit_log_search_and_action_combine(tmp_path, capsys):
    settings = make_settings(tmp_path)
    audit = AuditLogger(settings.audit_log_path)
    audit.log("applied", job_id="1", company="Acme Corp")
    audit.log("scored", job_id="2", score=40, error="Acme Corp mismatch")
    audit.log("applied", job_id="3", company="Beta Inc")

    cmd_audit_log(settings, audit_log_args(search="acme", action="applied"))

    out = capsys.readouterr().out
    assert "job_id=1" in out
    assert "job_id=2" not in out
    assert "job_id=3" not in out


def test_audit_log_format_json_prints_entries(tmp_path, capsys):
    settings = make_settings(tmp_path)
    AuditLogger(settings.audit_log_path).log("applied", job_id="1", company="Acme")

    cmd_audit_log(settings, audit_log_args(format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["action"] == "applied"
    assert payload[0]["details"] == {"job_id": "1", "company": "Acme"}


def test_audit_log_format_json_on_empty_log_is_still_valid_json(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_audit_log(settings, audit_log_args(format="json"))

    assert json.loads(capsys.readouterr().out) == []


def test_audit_log_does_not_crash_on_an_entry_with_a_null_details_field(tmp_path, capsys):
    """Real bug this guards against: a hand-edited or externally-authored
    audit.log line with "details": null (valid JSON, a valid entry -
    AuditLogger.read_entries() accepts it) previously crashed this
    command's text output with a raw AttributeError from calling .items()
    directly on None.
    """
    settings = make_settings(tmp_path)
    settings.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    settings.audit_log_path.write_text(
        '{"timestamp": "t", "action": "manual_note", "details": null}\n', encoding="utf-8"
    )

    cmd_audit_log(settings, audit_log_args())  # must not raise

    assert "manual_note" in capsys.readouterr().out


def test_audit_log_failures_reads_the_failed_applications_log_instead(tmp_path, capsys):
    """--failures points this at FAILED_APPLICATIONS_LOG_PATH instead of
    AUDIT_LOG_PATH - the exact file `_print_cycle_summary` tells a user to
    check by hand after a run that couldn't finish some postings. An entry
    only in one of the two files must never show up when reading the other.
    """
    settings = make_settings(tmp_path)
    AuditLogger(settings.audit_log_path).log("applied", job_id="1", company="Acme")
    AuditLogger(settings.failed_applications_log_path).log("prep_error", job_id="2", error="boom")

    cmd_audit_log(settings, audit_log_args(failures=True))

    out = capsys.readouterr().out
    assert "prep_error" in out
    assert "applied" not in out


def test_audit_log_failures_says_so_when_nothing_recorded(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_audit_log(settings, audit_log_args(failures=True))

    assert "No failed-application entries recorded yet." in capsys.readouterr().out


def test_audit_log_failures_supports_search_and_format_json(tmp_path, capsys):
    settings = make_settings(tmp_path)
    AuditLogger(settings.failed_applications_log_path).log("prep_error", job_id="1", error="Acme Corp boom")
    AuditLogger(settings.failed_applications_log_path).log("apply_error", job_id="2", error="unrelated")

    cmd_audit_log(settings, audit_log_args(failures=True, search="acme", format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["action"] == "prep_error"


# --- main() ---


def test_main_stops_cleanly_on_keyboard_interrupt(tmp_path, monkeypatch, capsys):
    """Real failure this guards against: Ctrl+C during a real browser action
    (mid Easy Apply form fill, waiting on Ollama, ...) used to propagate a
    raw traceback all the way out of main() - confusing on its own, and it
    also correlated with the *next* run failing outright with "profile is
    already in use by another instance of Chromium" (seen live), since the
    interrupted browser_session() cleanup never finished cleanly. A plain
    `job-bot run` had no KeyboardInterrupt handling at all before this fix -
    only --loop mode's own inner loop did.
    """
    settings = make_settings(tmp_path)
    monkeypatch.setattr("job_bot.cli.get_settings", lambda: settings)
    monkeypatch.setattr("job_bot.cli.configure_logging", lambda: None)
    monkeypatch.setattr("sys.argv", ["job-bot", "run"])

    def raise_interrupt(settings, args):
        raise KeyboardInterrupt

    monkeypatch.setattr("job_bot.cli.cmd_run", raise_interrupt)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert "Stopped." in capsys.readouterr().out


def test_main_reports_an_expected_error_and_exits_1(tmp_path, monkeypatch, capsys):
    """EXPECTED_ERRORS (ClaudeProviderError, ResumeParseError, ...) are
    user-facing configuration/input problems, not bugs - main() must turn
    them into a clean "Error: ..." line and exit(1), never a traceback.
    """
    settings = make_settings(tmp_path)
    monkeypatch.setattr("job_bot.cli.get_settings", lambda: settings)
    monkeypatch.setattr("job_bot.cli.configure_logging", lambda: None)
    monkeypatch.setattr("sys.argv", ["job-bot", "doctor"])
    monkeypatch.setattr(
        "job_bot.cli.cmd_doctor",
        lambda settings, args: (_ for _ in ()).throw(EXPECTED_ERRORS[0]("bad config")),
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert "Error: bad config" in capsys.readouterr().err


def test_main_reports_a_corrupted_tracker_database_cleanly(tmp_path, monkeypatch, capsys):
    """Real failure this guards against: every command that touches the
    tracker (Tracker.__init__()'s _init_db() runs on every construction)
    opens it eagerly, so a corrupted db.sqlite3 previously crashed whichever
    command hit it first with a raw sqlite3.DatabaseError traceback -
    sqlite3.DatabaseError isn't one of EXPECTED_ERRORS, so it fell straight
    through main()'s except clause uncaught. `job-bot status` (which
    doctor's own equivalent check doesn't run) is used here to prove the fix
    isn't doctor-specific.
    """
    settings = make_settings(tmp_path)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.db_path.write_text("not a sqlite database", encoding="utf-8")
    monkeypatch.setattr("job_bot.cli.get_settings", lambda: settings)
    monkeypatch.setattr("job_bot.cli.configure_logging", lambda: None)
    monkeypatch.setattr("sys.argv", ["job-bot", "status", "job1", "applied"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "Error: The tracker database at" in err
    assert str(settings.db_path) in err
    assert "job-bot doctor" in err


@pytest.mark.parametrize(
    "argv, cmd_name",
    [
        (["job-bot", "login"], "cmd_login"),
        (["job-bot", "run"], "cmd_run"),
        (["job-bot", "test-provider"], "cmd_test_provider"),
        (["job-bot", "doctor"], "cmd_doctor"),
        (["job-bot", "status", "job1", "applied"], "cmd_status"),
        (["job-bot", "review-answers"], "cmd_review_answers"),
        (["job-bot", "report"], "cmd_report"),
        (["job-bot", "export"], "cmd_export"),
        (["job-bot", "gmail-sync"], "cmd_gmail_sync"),
        (["job-bot", "dashboard"], "cmd_dashboard"),
        (["job-bot", "blacklist", "list"], "cmd_blacklist"),
        (["job-bot", "faq", "list"], "cmd_faq"),
        (["job-bot", "qa-history"], "cmd_qa_history"),
        (["job-bot", "resume-history"], "cmd_resume_history"),
        (["job-bot", "audit-log"], "cmd_audit_log"),
    ],
)
def test_main_dispatches_each_subcommand_to_its_own_handler(tmp_path, monkeypatch, argv, cmd_name):
    """main()'s dispatch is a long if/elif chain matched on args.command -
    every individual cmd_* function has its own tests that call it
    directly, bypassing this chain entirely, so nothing else in this suite
    would catch a copy-paste mistake here (matching the wrong branch, or a
    command silently falling through to no handler at all).
    """
    settings = make_settings(tmp_path)
    monkeypatch.setattr("job_bot.cli.get_settings", lambda: settings)
    monkeypatch.setattr("job_bot.cli.configure_logging", lambda: None)
    monkeypatch.setattr("sys.argv", argv)

    calls = []
    monkeypatch.setattr(f"job_bot.cli.{cmd_name}", lambda *a, **kw: calls.append((a, kw)))

    main()

    assert len(calls) == 1


# --- test-provider, gmail-sync, dashboard ---


def test_test_provider_prints_provider_and_result(tmp_path, monkeypatch, capsys):
    settings = make_settings(tmp_path)
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: FakeScoreProvider())

    cmd_test_provider(settings, argparse.Namespace(format="text"))

    out = capsys.readouterr().out
    assert "Provider OK: claude" in out
    assert "score=90" in out


def test_test_provider_format_json_prints_the_result_as_json(tmp_path, monkeypatch, capsys):
    settings = make_settings(tmp_path)
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: FakeScoreProvider())

    cmd_test_provider(settings, argparse.Namespace(format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == "claude"
    assert payload["result"]["score"] == 90


def test_gmail_sync_prints_scanned_count_with_no_emails(tmp_path, monkeypatch, capsys):
    settings = make_settings(tmp_path)
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: FakeScoreProvider())
    monkeypatch.setattr("job_bot.cli.GmailClient", FakeGmailClientForCli)

    cmd_gmail_sync(settings, gmail_sync_args())

    assert "Scanned 0 email(s)." in capsys.readouterr().out


def test_gmail_sync_prints_updates_low_confidence_and_unmatched_sections(tmp_path, monkeypatch, capsys):
    from job_bot.integrations.gmail_sync import GmailSyncResult

    settings = make_settings(tmp_path)
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: FakeScoreProvider())
    monkeypatch.setattr("job_bot.cli.GmailClient", FakeGmailClientForCli)
    fake_result = GmailSyncResult(
        total_emails=5,
        updated=[("job1", "Acme", "interviewing")],
        unmatched_subjects=["Re: your application"],
        skipped_low_confidence=2,
    )
    monkeypatch.setattr("job_bot.cli.sync_gmail", lambda *a, **kw: fake_result)

    cmd_gmail_sync(settings, gmail_sync_args())

    out = capsys.readouterr().out
    assert "Scanned 5 email(s)." in out
    assert "Updated: Acme (job1) -> interviewing" in out
    assert "Skipped 2 low-confidence email(s)." in out
    assert "couldn't confidently match" in out
    assert "Re: your application" in out


def test_gmail_sync_prints_a_classification_errors_line_when_nonzero(tmp_path, monkeypatch, capsys):
    from job_bot.integrations.gmail_sync import GmailSyncResult

    settings = make_settings(tmp_path)
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: FakeScoreProvider())
    monkeypatch.setattr("job_bot.cli.GmailClient", FakeGmailClientForCli)
    fake_result = GmailSyncResult(total_emails=3, classification_errors=1)
    monkeypatch.setattr("job_bot.cli.sync_gmail", lambda *a, **kw: fake_result)

    cmd_gmail_sync(settings, gmail_sync_args())

    out = capsys.readouterr().out
    assert "1 email(s) could not be classified (LLM provider error)" in out
    assert "job-bot audit-log --action gmail_sync_classify_error" in out


def test_gmail_sync_omits_the_classification_errors_line_when_zero(tmp_path, monkeypatch, capsys):
    from job_bot.integrations.gmail_sync import GmailSyncResult

    settings = make_settings(tmp_path)
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: FakeScoreProvider())
    monkeypatch.setattr("job_bot.cli.GmailClient", FakeGmailClientForCli)
    fake_result = GmailSyncResult(total_emails=3)
    monkeypatch.setattr("job_bot.cli.sync_gmail", lambda *a, **kw: fake_result)

    cmd_gmail_sync(settings, gmail_sync_args())

    assert "could not be classified" not in capsys.readouterr().out


def test_gmail_sync_dry_run_prefixes_updates_as_would_update(tmp_path, monkeypatch, capsys):
    from job_bot.integrations.gmail_sync import GmailSyncResult

    settings = make_settings(tmp_path)
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: FakeScoreProvider())
    monkeypatch.setattr("job_bot.cli.GmailClient", FakeGmailClientForCli)
    fake_result = GmailSyncResult(total_emails=1, updated=[("job1", "Acme", "offer")])
    monkeypatch.setattr("job_bot.cli.sync_gmail", lambda *a, **kw: fake_result)

    cmd_gmail_sync(settings, gmail_sync_args(dry_run=True))

    assert "[dry-run] Would update: Acme (job1) -> offer" in capsys.readouterr().out


def test_gmail_sync_format_json_prints_the_full_result(tmp_path, monkeypatch, capsys):
    from job_bot.integrations.gmail_sync import GmailSyncResult

    settings = make_settings(tmp_path)
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: FakeScoreProvider())
    monkeypatch.setattr("job_bot.cli.GmailClient", FakeGmailClientForCli)
    fake_result = GmailSyncResult(
        total_emails=5,
        updated=[("job1", "Acme", "interviewing")],
        unmatched_subjects=["Re: your application"],
        skipped_low_confidence=2,
        classification_errors=1,
    )
    monkeypatch.setattr("job_bot.cli.sync_gmail", lambda *a, **kw: fake_result)

    cmd_gmail_sync(settings, gmail_sync_args(format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "dry_run": False,
        "total_emails": 5,
        "updated": [{"job_id": "job1", "company": "Acme", "new_status": "interviewing"}],
        "skipped_low_confidence": 2,
        "unmatched_subjects": ["Re: your application"],
        "classification_errors": 1,
    }


def test_gmail_sync_format_json_reflects_dry_run(tmp_path, monkeypatch, capsys):
    from job_bot.integrations.gmail_sync import GmailSyncResult

    settings = make_settings(tmp_path)
    monkeypatch.setattr("job_bot.cli.get_provider", lambda settings: FakeScoreProvider())
    monkeypatch.setattr("job_bot.cli.GmailClient", FakeGmailClientForCli)
    fake_result = GmailSyncResult(total_emails=1, updated=[("job1", "Acme", "offer")])
    monkeypatch.setattr("job_bot.cli.sync_gmail", lambda *a, **kw: fake_result)

    cmd_gmail_sync(settings, gmail_sync_args(dry_run=True, format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["updated"] == [{"job_id": "job1", "company": "Acme", "new_status": "offer"}]


def _fake_run_dashboard(calls: list) -> object:
    """A stand-in for job_bot.cli.run_dashboard that just records every
    positional/keyword argument it was called with, for the three tests
    below to assert cmd_dashboard wires settings/args through correctly
    without starting a real server.
    """

    def fake(
        db_path,
        blacklist_path,
        audit_log_path,
        failed_applications_log_path,
        answer_gaps_path,
        resume_path,
        faq_path,
        port,
        open_browser,
        stale_after_days,
    ):
        calls.append(
            (
                db_path,
                blacklist_path,
                audit_log_path,
                failed_applications_log_path,
                answer_gaps_path,
                resume_path,
                faq_path,
                port,
                open_browser,
                stale_after_days,
            )
        )

    return fake


def test_dashboard_passes_port_and_open_browser_through(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    calls = []
    monkeypatch.setattr("job_bot.cli.run_dashboard", _fake_run_dashboard(calls))
    args = argparse.Namespace(port=9999, no_open=True)

    cmd_dashboard(settings, args)

    assert calls == [
        (
            settings.db_path,
            settings.blacklist_path,
            settings.audit_log_path,
            settings.failed_applications_log_path,
            settings.answer_gaps_path,
            settings.resume_path,
            settings.faq_path,
            9999,
            False,
            settings.stale_after_days,
        )
    ]


def test_dashboard_falls_back_to_settings_port_when_not_given(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, dashboard_port=8765)
    calls = []
    monkeypatch.setattr("job_bot.cli.run_dashboard", _fake_run_dashboard(calls))
    args = argparse.Namespace(port=None, no_open=False)

    cmd_dashboard(settings, args)

    assert calls == [
        (
            settings.db_path,
            settings.blacklist_path,
            settings.audit_log_path,
            settings.failed_applications_log_path,
            settings.answer_gaps_path,
            settings.resume_path,
            settings.faq_path,
            8765,
            True,
            settings.stale_after_days,
        )
    ]


def test_dashboard_passes_settings_stale_after_days_through(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, stale_after_days=30)
    calls = []
    monkeypatch.setattr("job_bot.cli.run_dashboard", _fake_run_dashboard(calls))
    args = argparse.Namespace(port=None, no_open=False)

    cmd_dashboard(settings, args)

    assert [call[-1] for call in calls] == [30]


# --- docs consistency ---


def test_every_cli_flag_is_documented_in_the_readme():
    """Real gap this guards against: `job-bot report`'s --stale-days flag
    had no mention anywhere in README.md - not even a usage example -
    despite every other subcommand flag being documented at least once.
    Walks every subcommand's own flags (skipping --help, which argparse
    adds automatically, not something anyone hand-documents) and asserts
    each appears as a literal "--flag-name" substring somewhere in
    README.md, so a newly added flag can't silently go undocumented the
    same way again.
    """
    readme_text = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    parser = build_parser()
    sub_action = next(a for a in parser._subparsers._group_actions if a.choices)

    undocumented = []
    for cmd_name, subparser in sub_action.choices.items():
        for action in subparser._actions:
            for opt in action.option_strings:
                if opt != "--help" and opt not in readme_text:
                    undocumented.append(f"{cmd_name} {opt}")
    assert undocumented == []


# --- test-helper/parser consistency ---


@pytest.mark.parametrize(
    "helper, argv",
    [
        (report_args, ["report"]),
        (export_args, ["export"]),
        (doctor_args, ["doctor"]),
        (status_args, ["status", "job1"]),
        (review_answers_args, ["review-answers"]),
        (qa_history_args, ["qa-history"]),
        (resume_history_args, ["resume-history"]),
        (audit_log_args, ["audit-log"]),
    ],
)
def test_args_helper_stays_in_sync_with_the_real_parser(helper, argv):
    """Real gap this guards against: report_args() above silently missed the
    new --by-company flag added alongside `job-bot report --by-company`
    (see cli.py) - it only surfaced as an AttributeError from cmd_report()
    itself on the next full-suite run, from whichever test happened to
    exercise the gap, rather than immediately and by name. These five
    commands' cmd_* functions consume every attribute their own argparse
    subparser defines (unlike `run`/`gmail-sync`, whose --provider/--model
    are read by main()'s _apply_provider_overrides() before dispatch, not
    by cmd_run/cmd_gmail_sync themselves - so their test helpers
    legitimately omit those two keys and don't belong in this check).
    Comparing each helper's default Namespace keys against build_parser()'s
    own argparse output for the same subcommand catches a missing, stale,
    or renamed key the moment a helper drifts from the real CLI.
    """
    real_keys = set(vars(build_parser().parse_args(argv))) - {"command"}
    assert set(vars(helper())) == real_keys
