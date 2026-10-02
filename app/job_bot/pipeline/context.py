"""Everything one `job-bot run` holds fixed across its cycles, as one value.

Before this, cli.py's _run_apply_cycle() took 21 keyword arguments - the
stores, loggers, confirmers and settings it shares with the whole run,
plus the options resolved from flags and .env - and every new dependency
meant editing the signature, the call site, and every test that built
one. RunContext groups the run-wide ones; what changes per session or per
cycle (the browser adapter and page, the resume text re-read each cycle)
stays an explicit argument. See docs/architecture.md.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from job_bot.config import Settings
from job_bot.llm.base import LLMProvider
from job_bot.resume.store import ResumeStore
from job_bot.safety.answer_gaps import AnswerGapStore
from job_bot.safety.audit_log import AuditLogger
from job_bot.safety.blacklist import CompanyBlacklist
from job_bot.safety.confirm import SubmitConfirmer
from job_bot.safety.rate_limiter import RateLimiter
from job_bot.tracker.db import Tracker


@dataclass(frozen=True)
class RunContext:
    # Shared services and configuration.
    settings: Settings
    args: argparse.Namespace
    provider: LLMProvider
    resume_store: ResumeStore
    tracker: Tracker
    rate_limiter: RateLimiter
    blacklist: CompanyBlacklist
    confirmer: SubmitConfirmer
    external_confirmer: SubmitConfirmer
    audit: AuditLogger
    failure_log: AuditLogger
    answer_gaps: AnswerGapStore
    # Options resolved once from flags and .env (see cmd_run()).
    min_score: int
    exclude_keywords: tuple[str, ...]
    experience_levels: list[str] | None
    max_years_experience: int | None
    require_w2: bool
    include_external: bool
