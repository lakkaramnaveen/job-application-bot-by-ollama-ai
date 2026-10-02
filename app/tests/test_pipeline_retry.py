"""recent_failures_to_retry(): which recently failed postings
`job-bot run --retry-failed` applies to again."""

from datetime import UTC, datetime

from job_bot.pipeline.retry import recent_failures_to_retry
from job_bot.tracker.db import Tracker

NOW = datetime(2026, 10, 2, 18, 0, tzinfo=UTC)
SINCE = datetime(2026, 9, 30, 18, 0, tzinfo=UTC)


def entry(job_id, when, action="apply_error"):
    return {"timestamp": when, "action": action, "details": {"job_id": job_id, "error": "boom"}}


def make_tracker(tmp_path):
    t = Tracker(tmp_path / "t.sqlite3")
    for job_id, score in (("a", 90), ("b", 80), ("low", 60), ("done", 85), ("decided", 85), ("old", 90), ("prep", 77)):
        t.record_score(job_id, "Engineer", "Acme", f"https://example.com/{job_id}", score, True)
    t.mark_applied("done")
    t.mark_skipped("decided")
    return t


def test_selects_recent_undecided_strong_fits_most_recent_first(tmp_path):
    entries = [
        entry("b", "2026-10-02T15:00:00+00:00"),
        entry("a", "2026-10-01T09:00:00+00:00"),
        entry("a", "2026-10-02T16:00:00+00:00"),  # a's latest failure is the newest
        entry("prep", "2026-10-01T10:00:00+00:00", action="prep_error"),
        entry("low", "2026-10-02T12:00:00+00:00"),  # below the bar
        entry("done", "2026-10-02T12:00:00+00:00"),  # applied since
        entry("decided", "2026-10-02T12:00:00+00:00"),  # decided against
        entry("old", "2026-09-20T12:00:00+00:00"),  # before the window
        entry("untracked", "2026-10-02T12:00:00+00:00"),
        {"timestamp": "2026-10-02T12:00:00+00:00", "action": "search_error", "details": {"error": "x"}},
        {"timestamp": "not a date", "action": "apply_error", "details": {"job_id": "b"}},
    ]

    picked = recent_failures_to_retry(entries, make_tracker(tmp_path), since=SINCE, min_score=75)

    assert picked == ["a", "b", "prep"]


def test_a_naive_timestamp_is_read_as_utc(tmp_path):
    picked = recent_failures_to_retry(
        [entry("a", "2026-10-02T12:00:00")], make_tracker(tmp_path), since=SINCE, min_score=75
    )
    assert picked == ["a"]
