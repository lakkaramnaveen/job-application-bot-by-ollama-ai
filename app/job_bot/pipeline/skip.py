"""Which postings a run passes over before spending a model call on them -
one policy, two views.

Before this module, the rules lived in two closures in cli.py's
_run_apply_cycle(): should_skip() (audited and printed, run on every
posting) and is_dead_end() (silent, handed to search() so it pages past
postings the run won't use). They overlapped, and they drifted: is_dead_end()
missed "already decided against in an earlier run", so those postings kept
filling 8 of 25 search slots in a live cycle (fixed in e264063). Here both
views are composed from the same per-rule predicates, so a rule changed
once changes everywhere.

Pure policy: no printing, no audit logging - SkipPolicy only says why.
cli.py turns a SkipReason into the audit entry and the line it prints.
See docs/architecture.md.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection
from datetime import UTC, datetime, timedelta
from enum import Enum

from job_bot.browser.base_adapter import JobPosting
from job_bot.matching.scorer import ENTRY_LEVEL_MAX_YEARS, SENIOR_TITLE_MIN_YEARS
from job_bot.safety.blacklist import CompanyBlacklist
from job_bot.tracker.db import Tracker


class SkipReason(Enum):
    ALREADY_APPLIED = "already_applied"
    TOO_MANY_FAILURES = "too_many_failures"
    BLACKLISTED = "blacklisted"
    COMPANY_LIMIT = "company_limit"
    EXCLUDED_KEYWORD = "excluded_keyword"
    ABOVE_LEVEL = "above_level"
    NEEDS_STATUS = "needs_status"


# Title words that put a posting above the level MAX_YEARS_EXPERIENCE
# targets - the same tiers the scorer's seniority rule asks the model to
# apply (see matching/scorer.py). Checked here in code as well because the
# model doesn't reliably apply them: live (2026-10-10, cap 2) it passed
# "Senior Developer" and "Founding Full-Stack Engineer". Whole words only,
# so "Leading" or "Staffing" don't match.
_STAFF_TITLE_WORDS = ("staff", "principal", "lead", "director", "head", "vp", "architect")
_SENIOR_TITLE_WORDS = ("senior", "sr")
_MID_TITLE_WORDS = ("founding", "ii", "iii", "iv", "mid", "mid-level")


def title_is_above_level(title: str, max_years_experience: int | None) -> bool:
    if max_years_experience is None:
        return False
    words: tuple[str, ...] = _STAFF_TITLE_WORDS
    if max_years_experience < SENIOR_TITLE_MIN_YEARS:
        words += _SENIOR_TITLE_WORDS
    if max_years_experience <= ENTRY_LEVEL_MAX_YEARS:
        words += _MID_TITLE_WORDS
    return re.search(rf"(?<![\w-])({'|'.join(words)})(?![\w-])", title.casefold()) is not None


# Title words saying the posting is closed to someone on a visa: a
# citizenship, permanent residency, or security clearance requirement -
# the scorer's NEEDS_VISA_SPONSORSHIP rule, checked here in code too because
# the model doesn't reliably apply it when the requirement is only in the
# title: live (2026-10-20) it passed "AI GTM Engineer (VIE, EU citizen)".
# Whole words only.
_STATUS_TITLE_PATTERN = re.compile(
    r"(?<![\w-])("
    r"citizens?|citizenship|green card|gc|usc|permanent residents?"
    r"|clearance|cleared|ts/sci|top secret|secret|public trust|polygraph"
    r")(?![\w-])"
)


def title_requires_status(title: str, needs_visa_sponsorship: bool) -> bool:
    return needs_visa_sponsorship and _STATUS_TITLE_PATTERN.search(title.casefold()) is not None


class SkipPolicy:
    def __init__(
        self,
        tracker: Tracker,
        blacklist: CompanyBlacklist,
        *,
        max_applications_per_company: int,
        max_apply_attempts: int,
        company_limit_window_days: int = 0,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        exclude_keywords: Collection[str] = (),
        requested_ids: Collection[str] = (),
        max_years_experience: int | None = None,
        needs_visa_sponsorship: bool = False,
    ):
        """exclude_keywords must already be casefolded. requested_ids are
        `job-bot run --job-id` postings: a deliberate retry, so the retry
        limit doesn't apply to them."""
        self._tracker = tracker
        self._blacklist = blacklist
        self._company_limit = max_applications_per_company
        self._company_window_days = company_limit_window_days
        self._now = now
        self._max_attempts = max_apply_attempts
        self._exclude_keywords = tuple(exclude_keywords)
        self._requested_ids = frozenset(requested_ids)
        self._max_years_experience = max_years_experience
        self._needs_visa_sponsorship = needs_visa_sponsorship

    def reason(self, posting: JobPosting) -> SkipReason | None:
        """Why this run passes over `posting` before scoring it, or None.
        Checked in this order - the first match is the reason recorded.
        "Already decided against in an earlier run" isn't one: the posting
        loop handles that itself, after this, without recording anything."""
        if self._already_applied(posting):
            return SkipReason.ALREADY_APPLIED
        if self._too_many_failures(posting) and posting.job_id not in self._requested_ids:
            return SkipReason.TOO_MANY_FAILURES
        if self._blacklist.is_blocked(posting.company):
            return SkipReason.BLACKLISTED
        if self._company_limit_reached(posting):
            return SkipReason.COMPANY_LIMIT
        if self._has_excluded_keyword(posting):
            return SkipReason.EXCLUDED_KEYWORD
        if title_is_above_level(posting.title, self._max_years_experience):
            return SkipReason.ABOVE_LEVEL
        if title_requires_status(posting.title, self._needs_visa_sponsorship):
            return SkipReason.NEEDS_STATUS
        return None

    def is_dead_end(self, posting: JobPosting) -> bool:
        """The silent view, for search(): postings this run will certainly
        pass over, so they don't take a slot in the search pool.

        Deliberately not included: a blacklisted company (rare, and its
        skip_blacklisted audit entry is a safety record that reason()'s
        caller must still write), and an excluded keyword or a title above
        the targeted level (cheap to check after search, and both settings
        change between runs).
        """
        return (
            self._already_applied(posting)
            or self._already_decided(posting)
            or self._company_limit_reached(posting)
            or self._too_many_failures(posting)
        )

    @property
    def company_limit(self) -> int:
        return self._company_limit

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    def _already_applied(self, posting: JobPosting) -> bool:
        return self._tracker.has_applied(posting.job_id)

    def _already_decided(self, posting: JobPosting) -> bool:
        """Decided against in an earlier run - scored below the bar,
        ineligible, or set to a final status by hand."""
        existing = self._tracker.get_job(posting.job_id)
        return existing is not None and existing["status"] != "seen"

    def _too_many_failures(self, posting: JobPosting) -> bool:
        return self._max_attempts > 0 and self._tracker.apply_failures(posting.job_id) >= self._max_attempts

    def _company_limit_reached(self, posting: JobPosting) -> bool:
        if self._company_limit <= 0:
            return False
        since = self._now() - timedelta(days=self._company_window_days) if self._company_window_days > 0 else None
        return self._tracker.applications_at_company(posting.company, since=since) >= self._company_limit

    def _has_excluded_keyword(self, posting: JobPosting) -> bool:
        title = posting.title.casefold()
        return any(keyword in title for keyword in self._exclude_keywords)
