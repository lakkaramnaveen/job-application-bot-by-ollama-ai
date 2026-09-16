"""Tests for the small, previously-uncovered CLI commands: report, status,
blacklist, and export. Unlike test_cli_run.py these don't need a fake
browser/LLM provider - each command only touches the tracker DB, the
blacklist file, or stdout/a file.
"""

import argparse
import csv
import io
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from job_bot.cli import (
    cmd_blacklist,
    cmd_doctor,
    cmd_export,
    cmd_report,
    cmd_review_answers,
    cmd_status,
    main,
)
from job_bot.config import Settings
from job_bot.resume.store import ResumeStore
from job_bot.safety.answer_gaps import AnswerGapStore
from job_bot.tracker.db import Tracker


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
    )
    defaults.update(overrides)
    return Settings(**defaults)


def report_args(**overrides) -> argparse.Namespace:
    defaults = dict(stale_days=None, by_score=False)
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

    cmd_status(settings, argparse.Namespace(job_id="job1", status="interviewing"))

    assert tracker.get_job("job1")["status"] == "interviewing"
    assert "job1 -> interviewing" in capsys.readouterr().out


def test_status_on_unknown_job_id_exits_with_error(tmp_path, capsys):
    settings = make_settings(tmp_path)
    Tracker(settings.db_path)  # create the (empty) DB

    with pytest.raises(SystemExit) as exc_info:
        cmd_status(settings, argparse.Namespace(job_id="does-not-exist", status="offer"))

    assert exc_info.value.code == 1
    assert "Error" in capsys.readouterr().err


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


def test_report_by_score_omitted_without_the_flag(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.record_score("job1", "Backend Engineer", "Acme", "https://x/1", score=92, should_apply=True)

    cmd_report(settings, report_args())

    assert "Outcomes by match score" not in capsys.readouterr().out


# --- blacklist ---


def test_blacklist_add_list_remove_round_trip(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="add", company="Acme Corp"))
    assert "Added to blacklist: Acme Corp" in capsys.readouterr().out

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="list", company=None))
    assert "acme corp" in capsys.readouterr().out  # blacklist entries are stored normalized

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="remove", company="Acme Corp"))
    assert "Removed from blacklist: Acme Corp" in capsys.readouterr().out

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="list", company=None))
    assert "Blacklist is empty." in capsys.readouterr().out


def test_blacklist_remove_of_absent_company_says_so(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_blacklist(settings, argparse.Namespace(blacklist_action="remove", company="Nobody Inc"))

    assert "Not on the blacklist: Nobody Inc" in capsys.readouterr().out


# --- export ---


def test_export_to_stdout_is_valid_csv_with_all_jobs(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1", match_score=80)
    tracker.upsert_job("job2", "Frontend Engineer", "Beta", "https://x/2", match_score=60)
    tracker.mark_applied("job2")

    cmd_export(settings, argparse.Namespace(status=None, out=None))

    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert [r["job_id"] for r in rows] == ["job1", "job2"]
    assert rows[1]["status"] == "applied"
    assert rows[1]["applied_at"]


def test_export_filters_by_status(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    tracker.upsert_job("job2", "Frontend Engineer", "Beta", "https://x/2")
    tracker.mark_applied("job2")

    cmd_export(settings, argparse.Namespace(status="applied", out=None))

    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert [r["job_id"] for r in rows] == ["job2"]


def test_export_to_file_writes_csv_and_reports_count(tmp_path, capsys):
    settings = make_settings(tmp_path)
    tracker = Tracker(settings.db_path)
    tracker.upsert_job("job1", "Backend Engineer", "Acme", "https://x/1")
    out_path = tmp_path / "export.csv"

    cmd_export(settings, argparse.Namespace(status=None, out=out_path))

    assert f"Exported 1 job(s) to {out_path}" in capsys.readouterr().out
    rows = list(csv.DictReader(out_path.open(encoding="utf-8")))
    assert rows[0]["job_id"] == "job1"


def test_export_with_no_jobs_writes_header_only(tmp_path, capsys):
    settings = make_settings(tmp_path)
    Tracker(settings.db_path)

    cmd_export(settings, argparse.Namespace(status=None, out=None))

    lines = capsys.readouterr().out.strip("\r\n").splitlines()
    assert len(lines) == 1
    assert lines[0].split(",")[0] == "job_id"


# --- doctor ---


def test_doctor_flags_missing_resume_and_passes_api_key_check(tmp_path, capsys):
    settings = make_settings(tmp_path)  # resume_path points at a file that was never created

    cmd_doctor(settings)

    out = capsys.readouterr().out
    assert "[!!] Resume file" in out
    assert "[OK] Anthropic API key" in out
    assert "checks passed." in out


def test_doctor_passes_resume_check_once_the_file_exists(tmp_path, capsys):
    settings = make_settings(tmp_path)
    settings.resume_path.write_text("resume", encoding="utf-8")

    cmd_doctor(settings)

    assert "[OK] Resume file" in capsys.readouterr().out


def test_doctor_flags_missing_anthropic_api_key(tmp_path, capsys):
    settings = make_settings(tmp_path, anthropic_api_key=None)

    cmd_doctor(settings)

    assert "[!!] Anthropic API key" in capsys.readouterr().out


def test_doctor_checks_ollama_base_url_when_using_ollama(tmp_path, capsys):
    settings = make_settings(tmp_path, llm_provider="ollama", anthropic_api_key=None)

    cmd_doctor(settings)

    assert "[OK] Ollama base URL configured" in capsys.readouterr().out


def test_doctor_flags_daily_cap_above_the_hard_ceiling(tmp_path, capsys):
    settings = make_settings(tmp_path, daily_application_cap=999)

    cmd_doctor(settings)

    assert "[!!] Daily application cap within hard ceiling" in capsys.readouterr().out


# --- review-answers ---


def test_review_answers_says_so_when_nothing_to_review(tmp_path, capsys):
    settings = make_settings(tmp_path)

    cmd_review_answers(settings)

    assert "nothing to review" in capsys.readouterr().out


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

    cmd_review_answers(settings)

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

    cmd_review_answers(settings)

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

    cmd_review_answers(settings)

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

    cmd_review_answers(settings)  # must not raise

    assert len(AnswerGapStore(settings.answer_gaps_path).list_unanswered()) == 2


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
