"""Tests for the small, previously-uncovered CLI commands: report, status,
blacklist, and export. Unlike test_cli_run.py these don't need a fake
browser/LLM provider - each command only touches the tracker DB, the
blacklist file, or stdout/a file.
"""

import argparse
import csv
import io

import pytest

from job_bot.cli import cmd_blacklist, cmd_export, cmd_report, cmd_status
from job_bot.config import Settings
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
        applications_dir=tmp_path / "applications",
    )
    defaults.update(overrides)
    return Settings(**defaults)


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

    cmd_report(settings)

    out = capsys.readouterr().out
    assert "seen" in out
    assert "applied" in out
    assert "total" in out
    assert "2" in out


def test_report_on_empty_tracker_says_so(tmp_path, capsys):
    settings = make_settings(tmp_path)
    Tracker(settings.db_path)

    cmd_report(settings)

    assert "No jobs tracked yet." in capsys.readouterr().out


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
