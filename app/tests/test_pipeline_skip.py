"""SkipPolicy: one set of rules behind both the audited skip (reason())
and search's silent filter (is_dead_end()) - see job_bot/pipeline/skip.py."""

import pytest

from job_bot.browser.base_adapter import JobPosting
from job_bot.pipeline.skip import SkipPolicy, SkipReason
from job_bot.safety.blacklist import CompanyBlacklist
from job_bot.tracker.db import Tracker


def posting(job_id, company="Umbrella", title="Backend Engineer"):
    return JobPosting(job_id=job_id, title=title, company=company, url=f"https://example.com/{job_id}", description="")


@pytest.fixture
def tracker(tmp_path):
    t = Tracker(tmp_path / "t.sqlite3")
    for job_id, company in (("applied", "Initech"), ("decided", "Hooli"), ("failing", "Globex"), ("fresh", "Umbrella")):
        t.upsert_job(job_id, "Engineer", company, f"https://example.com/{job_id}")
    t.mark_applied("applied")
    t.mark_skipped("decided")
    t.record_apply_failure("failing")
    t.record_apply_failure("failing")
    return t


@pytest.fixture
def blacklist(tmp_path):
    b = CompanyBlacklist(tmp_path / "blacklist.json")
    b.add("Evil Corp")
    return b


def policy(tracker, blacklist, **overrides):
    kwargs = dict(max_applications_per_company=1, max_apply_attempts=2, exclude_keywords=("staff",))
    kwargs.update(overrides)
    return SkipPolicy(tracker, blacklist, **kwargs)


@pytest.mark.parametrize(
    ("job", "expected"),
    [
        (posting("applied", "Initech"), SkipReason.ALREADY_APPLIED),
        (posting("failing", "Globex"), SkipReason.TOO_MANY_FAILURES),
        (posting("new-1", "Evil Corp"), SkipReason.BLACKLISTED),
        (posting("new-2", "Initech"), SkipReason.COMPANY_LIMIT),  # Initech already has an application
        (posting("new-3", title="Staff Engineer"), SkipReason.EXCLUDED_KEYWORD),
        (posting("fresh"), None),
        (posting("decided", "Hooli"), None),  # handled by the posting loop, not recorded as a skip
    ],
)
def test_reason(tracker, blacklist, job, expected):
    assert policy(tracker, blacklist).reason(job) is expected


def test_reason_checks_rules_in_order(tracker, blacklist):
    """An applied posting at a blacklisted company is recorded as applied."""
    blacklist.add("Initech")
    assert policy(tracker, blacklist).reason(posting("applied", "Initech")) is SkipReason.ALREADY_APPLIED


def test_a_requested_posting_bypasses_only_the_retry_limit(tracker, blacklist):
    p = policy(tracker, blacklist, requested_ids=["failing", "applied"])
    assert p.reason(posting("failing", "Globex")) is None
    assert p.reason(posting("applied", "Initech")) is SkipReason.ALREADY_APPLIED


def test_limits_of_zero_mean_unlimited(tracker, blacklist):
    p = policy(tracker, blacklist, max_applications_per_company=0, max_apply_attempts=0)
    assert p.reason(posting("new-2", "Initech")) is None
    assert p.reason(posting("failing", "Globex")) is None


@pytest.mark.parametrize(
    ("job", "dead_end"),
    [
        (posting("applied", "Initech"), True),
        (posting("decided", "Hooli"), True),  # the 8-of-25 pool leak (e264063)
        (posting("failing", "Globex"), True),
        (posting("new-2", "Initech"), True),
        # Left for the audited path on purpose:
        (posting("new-1", "Evil Corp"), False),
        (posting("new-3", title="Staff Engineer"), False),
        (posting("fresh"), False),
    ],
)
def test_is_dead_end(tracker, blacklist, job, dead_end):
    assert policy(tracker, blacklist).is_dead_end(job) is dead_end
