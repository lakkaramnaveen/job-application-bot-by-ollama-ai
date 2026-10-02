"""Which recently failed postings are worth applying to again - for
`job-bot run --retry-failed`.

Real case (2026-10-02): the tracker held 395 postings that scored at or
above the bar but were never applied to; 220 of them had failed on a bug
since fixed (the Easy Apply dialog not found, stuck forms, unanswered
yes/no questions, off-screen radio buttons). Once a posting drops out of
LinkedIn's last-24h/3-day search, or hits MAX_APPLY_ATTEMPTS while the
bug was still there, nothing retries it. 45 such postings from just the
previous two days were still open, at companies not yet applied to.

Pure selection: the failure log entries and the tracker are read, never
written. cli.py feeds the result through the same path as `--job-id`, so
the usual skip rules (company limit, blacklist, ...) still apply.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from job_bot.tracker.db import Tracker

# Actions in failed_applications.log that mean "this posting itself could
# not be completed" - a search_error has no posting to retry.
RETRYABLE_ACTIONS = frozenset({"apply_error", "prep_error"})


def recent_failures_to_retry(
    failure_entries: Iterable[dict[str, Any]],
    tracker: Tracker,
    *,
    since: datetime,
    min_score: int,
) -> list[str]:
    """Job ids, most recent failure first, of postings that failed at or
    after `since` and are still undecided: tracked, never applied to,
    status "seen", and scored at least `min_score`. Each id appears once.

    `failure_entries` are failed_applications.log entries in any order
    (AuditLogger.read_entries() gives most recent first); entries with an
    unparseable timestamp are ignored rather than guessed at.
    """
    latest: dict[str, datetime] = {}
    for entry in failure_entries:
        if entry.get("action") not in RETRYABLE_ACTIONS:
            continue
        job_id = str((entry.get("details") or {}).get("job_id") or "")
        when = _parse_timestamp(entry.get("timestamp"))
        if not job_id or when is None or when < since:
            continue
        if job_id not in latest or when > latest[job_id]:
            latest[job_id] = when

    retry: list[str] = []
    for job_id in sorted(latest, key=lambda j: latest[j], reverse=True):
        job = tracker.get_job(job_id)
        if job is None or job["applied_at"] or job["status"] != "seen":
            continue
        if job["match_score"] is None or job["match_score"] < min_score:
            continue
        retry.append(job_id)
    return retry


def since_days_ago(days: float, *, now: datetime) -> datetime:
    return now - timedelta(days=days)


def _parse_timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    # AuditLogger writes UTC; a naive timestamp is read as UTC rather than
    # failing the comparison with the timezone-aware cutoff.
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
