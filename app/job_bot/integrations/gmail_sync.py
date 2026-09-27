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
import re

from job_bot.integrations.email_classifier import classify_email
from job_bot.integrations.gmail_client import EmailMessage, GmailClient
from job_bot.llm.base import LLMProvider
from job_bot.llm.claude_provider import ClaudeProviderError
from job_bot.llm.ollama_provider import OllamaProviderError
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
    # Emails the LLM provider failed to classify (a retry-exhausted
    # validation failure, or the provider being unreachable mid-batch) -
    # see sync_gmail()'s own docstring for why this doesn't abort the rest
    # of the batch the way it previously did.
    classification_errors: int = 0


def _contains_as_whole_word(haystack: str, needle: str) -> bool:
    """Whether `needle` (one or more whole, space-separated words) appears
    in `haystack` without being butted up against surrounding word
    characters - the same (?<!\\w)/(?!\\w) lookaround technique
    linkedin_adapter.py's _best_match_index() uses for the analogous
    problem of matching a short string inside a longer one. Plain substring
    containment (`needle in haystack`) matches a tracked job named "AI"
    against ANY email whose company guess merely contains "ai" as a
    substring - "OpenAI", "Mail.com", "Fairbank" - a wrong company
    confidently matched and updated on a field this safety-critical (see
    this module's docstring: "never resolved by guessing"). Word-boundary
    matching still lets a real abbreviation match ("Acme" inside "Acme
    Corp"), since that boundary sits on a space, while rejecting a
    coincidental mid-word one.
    """
    if not needle:
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


def find_matching_job(jobs: list[dict], company_guess: str) -> dict | None:
    """Match a classifier's company guess to exactly one tracked job by
    whole-word overlap on the normalized company name (see
    _contains_as_whole_word). Returns None on zero or multiple candidates -
    an ambiguous match is treated as no match, never resolved by guessing.
    """
    norm_guess = normalize_company_name(company_guess)
    if not norm_guess:
        return None

    candidates = [
        job
        for job in jobs
        if (norm_company := normalize_company_name(job["company"]))
        and (
            _contains_as_whole_word(norm_guess, norm_company)
            or _contains_as_whole_word(norm_company, norm_guess)
        )
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
    """Logs every real outcome to `audit` when given, not just the ones
    that actually changed a tracked job's status: "gmail_sync_update" (as
    before), and now "gmail_sync_skipped_low_confidence"/"gmail_sync_
    unmatched" too - GmailSyncResult's own skipped_low_confidence/
    unmatched_subjects already report these for the current run's printed
    output, but until now they left no trace in the audit trail once that
    output scrolled past, unlike every other part of this tool (cmd_run's
    search/scored/skip_*/applied events) that logs its non-"success" cases
    just as faithfully as its successes. `job-bot audit-log --action
    gmail_sync_unmatched` (or --search) now finds a past run's ambiguous
    emails after the fact, for a status the user can still go set by hand
    with `job-bot status <job_id> <status>`.

    A classify_email() failure for one email (confirmed live: a retry-
    exhausted structured-output validation failure - see ollama_provider.py's
    generate_structured()) previously propagated straight out of this
    function uncaught, aborting the whole run - every email after the
    failing one in the batch was never even attempted, and the run's own
    summary (result.updated/unmatched_subjects/skipped_low_confidence) for
    every email processed *before* the failure was lost too, since nothing
    caught the exception to return the partial result. Unlike cmd_run's
    own per-posting resilience (a prep_error/apply_error for one posting
    doesn't abort the rest of that cycle's postings), this had no
    equivalent. Now caught per email, logged as "gmail_sync_classify_error"
    and counted in classification_errors, and the loop moves on to the
    next email - status updates already made to the tracker for prior
    emails in this same run are unaffected either way, since each is
    written immediately inside the loop, not batched at the end.
    """
    query = DEFAULT_QUERY_TEMPLATE.format(days=days)
    emails: list[EmailMessage] = gmail_client.search_messages(query, max_results=max_emails)
    tracked_jobs = tracker.list_jobs()

    result = GmailSyncResult(total_emails=len(emails))

    for email in emails:
        try:
            classification = classify_email(provider, email)
        except (ClaudeProviderError, OllamaProviderError) as e:
            result.classification_errors += 1
            if audit is not None:
                audit.log("gmail_sync_classify_error", email_subject=email.subject, error=str(e))
            continue
        if not classification.is_job_related or classification.category == "other":
            continue
        if classification.confidence < confidence_threshold:
            result.skipped_low_confidence += 1
            if audit is not None:
                audit.log(
                    "gmail_sync_skipped_low_confidence",
                    email_subject=email.subject,
                    category=classification.category,
                    confidence=classification.confidence,
                )
            continue

        new_status = CATEGORY_TO_STATUS.get(classification.category)
        if new_status is None:
            continue

        job = find_matching_job(tracked_jobs, classification.company_guess)
        if job is None:
            result.unmatched_subjects.append(email.subject)
            if audit is not None:
                audit.log(
                    "gmail_sync_unmatched",
                    email_subject=email.subject,
                    category=classification.category,
                    company_guess=classification.company_guess,
                    confidence=classification.confidence,
                )
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
