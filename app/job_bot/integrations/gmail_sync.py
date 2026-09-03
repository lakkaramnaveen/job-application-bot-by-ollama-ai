"""Reads recent Gmail messages, classifies the job-application-related ones,
and updates the tracker's status for the application each one matches -
without ever guessing at a match it isn't confident about.

Safety properties (see SECURITY.md):
- Read-only Gmail scope (job_bot.integrations.gmail_client.SCOPES) - this
  never sends, deletes, labels, or modifies mail.
- Email content is treated as untrusted data by the classifier prompt, the
  same way job postings are (job_bot.integrations.email_classifier).
- A status update only happens when exactly one tracked job's company name
  matches the email (never on an ambiguous or zero match), the classifier's
  confidence clears `confidence_threshold`, and the move is forward-only
  (STATUS_RANK below) - so a low-signal or garbled email can't downgrade or
  overwrite a status you already confirmed by hand. A `skipped` job (the bot
  chose not to apply) is likewise never touched - see NEVER_UPDATE_VIA_EMAIL.
"""

import dataclasses

from job_bot.integrations.email_classifier import classify_email
from job_bot.integrations.gmail_client import EmailMessage, GmailClient
from job_bot.llm.base import LLMProvider
from job_bot.models.schemas import EmailCategory
from job_bot.safety.audit_log import AuditLogger
from job_bot.text_utils import normalize_company_name
from job_bot.tracker.db import Tracker

CATEGORY_TO_STATUS: dict[EmailCategory, str] = {
    "interview_invite": "interviewing",
    "rejection": "rejected",
    "offer": "offer",
    "application_confirmation": "applied",
}

# Never move a job's status backward, and never touch one already in a
# terminal state - a later, possibly-misclassified email (an onboarding
# email after an offer, a stray digest after a rejection) shouldn't undo or
# relitigate an outcome the tracker already recorded.
STATUS_RANK = {
    "seen": 0,
    "applied": 1,
    "interviewing": 2,
    "offer": 3,
    "rejected": 3,
    "withdrawn": 3,
    "no_response": 3,
}
TERMINAL_STATUSES = frozenset({"offer", "rejected", "withdrawn", "no_response"})
# "skipped" means the bot decided *not* to apply - there's no real
# application behind it to correlate an email with, so it's excluded from
# email-driven updates the same way a terminal status is, rather than
# defaulting to STATUS_RANK's fallback of 0 (same rank as "seen"), which
# would let a loosely-matched email overwrite it as if the bot had applied.
NEVER_UPDATE_VIA_EMAIL = TERMINAL_STATUSES | {"skipped"}

DEFAULT_QUERY_TEMPLATE = (
    "newer_than:{days}d (interview OR application OR applying OR position OR role "
    'OR offer OR unfortunately OR "thank you for applying")'
)


@dataclasses.dataclass
class GmailSyncResult:
    total_emails: int = 0
    updated: list[tuple[str, str, str]] = dataclasses.field(
        default_factory=list
    )  # job_id, company, new_status
    unmatched_subjects: list[str] = dataclasses.field(default_factory=list)
    skipped_low_confidence: int = 0


def find_matching_job(jobs: list[dict], company_guess: str) -> dict | None:
    """Match a classifier's company guess to exactly one tracked job by
    substring overlap on the normalized company name. Returns None on zero
    or multiple candidates - an ambiguous match is treated as no match,
    never resolved by guessing.
    """
    norm_guess = normalize_company_name(company_guess)
    if not norm_guess:
        return None

    candidates = [
        job
        for job in jobs
        if (norm_company := normalize_company_name(job["company"]))
        and (norm_company in norm_guess or norm_guess in norm_company)
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _should_update(current_status: str, new_status: str) -> bool:
    if current_status in NEVER_UPDATE_VIA_EMAIL:
        return False
    if current_status == new_status:
        return False
    return STATUS_RANK.get(new_status, 0) >= STATUS_RANK.get(current_status, 0)


def sync_gmail(
    provider: LLMProvider,
    gmail_client: GmailClient,
    tracker: Tracker,
    *,
    days: int = 14,
    max_emails: int = 50,
    confidence_threshold: float = 0.6,
    dry_run: bool = False,
    audit: AuditLogger | None = None,
) -> GmailSyncResult:
    query = DEFAULT_QUERY_TEMPLATE.format(days=days)
    emails: list[EmailMessage] = gmail_client.search_messages(query, max_results=max_emails)
    tracked_jobs = tracker.list_jobs()

    result = GmailSyncResult(total_emails=len(emails))

    for email in emails:
        classification = classify_email(provider, email)
        if not classification.is_job_related or classification.category == "other":
            continue
        if classification.confidence < confidence_threshold:
            result.skipped_low_confidence += 1
            continue

        new_status = CATEGORY_TO_STATUS.get(classification.category)
        if new_status is None:
            continue

        job = find_matching_job(tracked_jobs, classification.company_guess)
        if job is None:
            result.unmatched_subjects.append(email.subject)
            continue

        if not _should_update(job["status"], new_status):
            continue

        if not dry_run:
            tracker.update_status(job["job_id"], new_status)
            job["status"] = new_status  # keep this run's in-memory copy consistent
        result.updated.append((job["job_id"], job["company"], new_status))

        if audit is not None:
            audit.log(
                "gmail_sync_update",
                job_id=job["job_id"],
                company=job["company"],
                new_status=new_status,
                email_subject=email.subject,
                confidence=classification.confidence,
                dry_run=dry_run,
            )

    return result
