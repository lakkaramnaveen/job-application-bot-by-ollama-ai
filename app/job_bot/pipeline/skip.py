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

from collections.abc import Collection
from enum import Enum

from job_bot.browser.base_adapter import JobPosting
from job_bot.safety.blacklist import CompanyBlacklist
from job_bot.tracker.db import Tracker


class SkipReason(Enum):
    ALREADY_APPLIED = "already_applied"
    TOO_MANY_FAILURES = "too_many_failures"
    BLACKLISTED = "blacklisted"
    COMPANY_LIMIT = "company_limit"
    EXCLUDED_KEYWORD = "excluded_keyword"


class SkipPolicy:
    def __init__(
        self,
        tracker: Tracker,
        blacklist: CompanyBlacklist,
        *,
        max_applications_per_company: int,
        max_apply_attempts: int,
        exclude_keywords: Collection[str] = (),
        requested_ids: Collection[str] = (),
    ):
        """exclude_keywords must already be casefolded. requested_ids are
        `job-bot run --job-id` postings: a deliberate retry, so the retry
        limit doesn't apply to them."""
        self._tracker = tracker
        self._blacklist = blacklist
        self._company_limit = max_applications_per_company
        self._max_attempts = max_apply_attempts
        self._exclude_keywords = tuple(exclude_keywords)
        self._requested_ids = frozenset(requested_ids)

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
        return None

    def is_dead_end(self, posting: JobPosting) -> bool:
        """The silent view, for search(): postings this run will certainly
        pass over, so they don't take a slot in the search pool.

        Deliberately not included: a blacklisted company (rare, and its
        skip_blacklisted audit entry is a safety record that reason()'s
        caller must still write) and an excluded keyword (cheap to check
        after search, and the list changes between runs).
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
        return self._company_limit > 0 and self._tracker.applications_at_company(posting.company) >= self._company_limit

    def _has_excluded_keyword(self, posting: JobPosting) -> bool:
        title = posting.title.casefold()
        return any(keyword in title for keyword in self._exclude_keywords)
