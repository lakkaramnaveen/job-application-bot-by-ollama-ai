import pytest

from job_bot.tracker.db import InvalidStatus, Tracker


def make_tracker(tmp_path) -> Tracker:
    return Tracker(tmp_path / "db.sqlite3")


def test_upsert_then_has_applied_is_false_until_marked(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1", match_score=80)

    assert tracker.has_applied("1") is False

    tracker.mark_applied("1")

    assert tracker.has_applied("1") is True


def test_mark_skipped_sets_status(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1")

    tracker.mark_skipped("1")

    assert tracker.has_applied("1") is False
    assert tracker.status_counts() == {"skipped": 1}


def test_update_status_accepts_valid_outcome_status(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1")
    tracker.mark_applied("1")

    tracker.update_status("1", "interviewing")

    assert tracker.status_counts() == {"interviewing": 1}


def test_update_status_rejects_unknown_status(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1")

    with pytest.raises(InvalidStatus, match="Unknown status"):
        tracker.update_status("1", "ghosted")


def test_update_status_rejects_unknown_job_id(tmp_path):
    tracker = make_tracker(tmp_path)

    with pytest.raises(ValueError, match="No tracked job"):
        tracker.update_status("does-not-exist", "offer")


def test_status_counts_reflects_multiple_jobs(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "A", "Acme", "https://example.com/1")
    tracker.upsert_job("2", "B", "Acme", "https://example.com/2")
    tracker.upsert_job("3", "C", "Acme", "https://example.com/3")
    tracker.mark_applied("1")
    tracker.mark_applied("2")
    tracker.mark_skipped("3")

    assert tracker.status_counts() == {"applied": 2, "skipped": 1}


def test_record_qa_and_upsert_job_do_not_conflict(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1")

    tracker.record_qa("1", "Years of experience?", "5")
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1", match_score=90)

    assert tracker.status_counts() == {"seen": 1}
