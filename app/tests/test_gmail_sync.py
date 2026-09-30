import pytest

from job_bot.integrations.gmail_client import EmailMessage
from job_bot.integrations.gmail_sync import _contains_as_whole_word, find_matching_job, sync_gmail
from job_bot.llm.base import LLMProvider
from job_bot.llm.ollama_provider import OllamaProviderError
from job_bot.models.schemas import EmailClassification
from job_bot.safety.audit_log import AuditLogger
from job_bot.tracker.db import Tracker


class QueueProvider(LLMProvider):
    """Returns one canned EmailClassification per call, in order - or
    raises, if the next queued item is an exception instance instead, so a
    single call in the middle of a batch can simulate the LLM provider
    failing on just that one email (a retry-exhausted validation failure,
    or a transient outage) without affecting the canned results queued
    before or after it.
    """

    def __init__(self, results: list[EmailClassification | Exception]):
        self._results = list(results)
        self.calls = 0

    def generate_structured(self, *, system, prompt, schema):
        self.calls += 1
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeGmailClient:
    def __init__(self, emails: list[EmailMessage]):
        self._emails = emails
        self.last_query = None

    def search_messages(self, query: str, max_results: int = 50) -> list[EmailMessage]:
        self.last_query = query
        return self._emails


def make_email(subject="Interview invite", **overrides) -> EmailMessage:
    defaults = dict(id="1", subject=subject, sender="a@b.com", date="", snippet="", body_text="body")
    defaults.update(overrides)
    return EmailMessage(**defaults)


def make_classification(**overrides) -> EmailClassification:
    defaults = dict(
        is_job_related=True,
        category="interview_invite",
        company_guess="Acme",
        role_guess="Engineer",
        confidence=0.9,
    )
    defaults.update(overrides)
    return EmailClassification(**defaults)


def make_tracker_with_job(tmp_path, status="applied", company="Acme Corp"):
    tracker = Tracker(tmp_path / "db.sqlite3")
    tracker.upsert_job("job1", "Engineer", company, "https://example.com/job1")
    if status != "seen":
        tracker.mark_applied("job1")
        if status not in ("applied",):
            tracker.update_status("job1", status)
    return tracker


# --- _contains_as_whole_word ---


def test_contains_as_whole_word_matches_a_whole_word():
    assert _contains_as_whole_word("acme corp", "acme") is True


def test_contains_as_whole_word_rejects_a_mid_word_substring():
    assert _contains_as_whole_word("openai", "ai") is False


def test_contains_as_whole_word_rejects_an_empty_needle():
    """Not currently reachable through find_matching_job() (its one caller
    - both norm_guess and norm_company are already checked non-empty
    before either _contains_as_whole_word() call), but this is a real
    correctness guarantee of the function itself, not dead code: without
    it, an empty needle's word-boundary pattern ((?<!\\w)(?!\\w)) can
    spuriously match wherever a haystack has a non-word boundary - e.g.
    a comma, or an empty haystack - confirmed directly below. Locking
    this in with a test protects a future caller that doesn't happen to
    share find_matching_job()'s own non-empty precondition.
    """
    assert _contains_as_whole_word("acme, corp", "") is False
    assert _contains_as_whole_word("", "") is False


# --- find_matching_job ---


def test_find_matching_job_unique_substring_match():
    jobs = [{"job_id": "1", "company": "Acme Corp"}, {"job_id": "2", "company": "Globex"}]
    match = find_matching_job(jobs, "Acme")
    assert match["job_id"] == "1"


def test_find_matching_job_no_match_returns_none():
    jobs = [{"job_id": "1", "company": "Acme Corp"}]
    assert find_matching_job(jobs, "Totally Unrelated Inc") is None


def test_find_matching_job_ambiguous_match_returns_none():
    jobs = [{"job_id": "1", "company": "Acme Corp"}, {"job_id": "2", "company": "Acme Robotics"}]
    assert find_matching_job(jobs, "Acme") is None


def test_find_matching_job_empty_guess_returns_none():
    jobs = [{"job_id": "1", "company": "Acme Corp"}]
    assert find_matching_job(jobs, "") is None


def test_find_matching_job_rejects_a_short_mid_word_false_positive():
    """Real bug this guards against: plain substring containment matched a
    tracked job named "AI" against ANY company_guess merely containing "ai"
    as a substring - "OpenAI" here, but "Mail.com"/"Fairbank" equally -
    silently attributing an unrelated company's email to the wrong tracked
    job. This is exactly the "resolved by guessing" failure the module's
    own docstring says a match must never allow.
    """
    jobs = [{"job_id": "1", "company": "AI"}, {"job_id": "2", "company": "Globex"}]
    assert find_matching_job(jobs, "OpenAI") is None


def test_find_matching_job_still_matches_a_whole_word_abbreviation():
    """The fix for the false positive above must not break the legitimate
    case it's modeled on: an email that only gives the short form of a
    tracked company's full name, separated by a real word boundary (a
    space), the same "Acme" / "Acme Corp" shape the pre-existing unique-
    substring-match test already covers, just confirmed from the other
    direction (short guess -> long tracked name AND long guess -> short
    tracked name).
    """
    jobs = [{"job_id": "1", "company": "Acme"}, {"job_id": "2", "company": "Globex"}]
    assert find_matching_job(jobs, "Acme Corp recruiting team")["job_id"] == "1"


# --- sync_gmail ---


def test_sync_gmail_updates_matching_job(tmp_path):
    tracker = make_tracker_with_job(tmp_path, status="applied")
    gmail = FakeGmailClient([make_email()])
    provider = QueueProvider([make_classification()])
    audit = AuditLogger(tmp_path / "audit.log")

    result = sync_gmail(provider, gmail, tracker, audit=audit)

    assert result.updated == [("job1", "Acme Corp", "interviewing")]
    assert tracker.get_job("job1")["status"] == "interviewing"
    assert "gmail_sync_update" in (tmp_path / "audit.log").read_text()


def test_sync_gmail_dry_run_does_not_write(tmp_path):
    tracker = make_tracker_with_job(tmp_path, status="applied")
    gmail = FakeGmailClient([make_email()])
    provider = QueueProvider([make_classification()])

    result = sync_gmail(provider, gmail, tracker, dry_run=True)

    assert result.updated == [("job1", "Acme Corp", "interviewing")]
    assert tracker.get_job("job1")["status"] == "applied"


def test_sync_gmail_continues_past_a_classification_error(tmp_path):
    """Real bug this guards against, confirmed live: a single email
    causing an LLM provider failure (a retry-exhausted structured-output
    validation failure, or a transient outage) previously propagated
    straight out of sync_gmail(), aborting the whole run - every email
    after the failing one in the batch was never even attempted, and the
    run's own summary for every email processed *before* the failure was
    lost too, since nothing caught the exception to return a result at all.
    """
    tracker = make_tracker_with_job(tmp_path, status="applied")
    gmail = FakeGmailClient(
        [make_email(subject="email 1"), make_email(subject="email 2"), make_email(subject="email 3")]
    )
    provider = QueueProvider(
        [
            make_classification(),
            OllamaProviderError("simulated: model returned unparseable JSON after retries"),
            make_classification(),
        ]
    )

    result = sync_gmail(provider, gmail, tracker)

    assert provider.calls == 3  # every email was attempted, not just up to the failure
    assert result.classification_errors == 1
    assert result.updated == [("job1", "Acme Corp", "interviewing")]  # email 1's real result survived


def test_sync_gmail_logs_classification_errors_to_the_audit_trail(tmp_path):
    tracker = make_tracker_with_job(tmp_path, status="applied")
    gmail = FakeGmailClient([make_email(subject="Broken email")])
    provider = QueueProvider([OllamaProviderError("simulated failure")])
    audit = AuditLogger(tmp_path / "audit.log")

    sync_gmail(provider, gmail, tracker, audit=audit)

    log_text = (tmp_path / "audit.log").read_text()
    assert "gmail_sync_classify_error" in log_text
    assert "Broken email" in log_text
    assert "simulated failure" in log_text


def test_sync_gmail_skips_low_confidence(tmp_path):
    tracker = make_tracker_with_job(tmp_path, status="applied")
    gmail = FakeGmailClient([make_email()])
    provider = QueueProvider([make_classification(confidence=0.2)])

    result = sync_gmail(provider, gmail, tracker, confidence_threshold=0.6)

    assert result.updated == []
    assert result.skipped_low_confidence == 1
    assert tracker.get_job("job1")["status"] == "applied"


def test_sync_gmail_logs_low_confidence_skips_to_the_audit_trail(tmp_path):
    """Real gap this guards against: GmailSyncResult.skipped_low_confidence
    only ever reported this in the current run's own printed output -
    nothing recorded it to the audit trail the way every other outcome
    cmd_run/gmail_sync produces already does, so a past run's low-
    confidence skip left no trace `job-bot audit-log` could find later.
    """
    tracker = make_tracker_with_job(tmp_path, status="applied")
    gmail = FakeGmailClient([make_email(subject="Maybe an interview?")])
    provider = QueueProvider([make_classification(confidence=0.2)])
    audit = AuditLogger(tmp_path / "audit.log")

    sync_gmail(provider, gmail, tracker, confidence_threshold=0.6, audit=audit)

    log_text = (tmp_path / "audit.log").read_text()
    assert "gmail_sync_skipped_low_confidence" in log_text
    assert "Maybe an interview?" in log_text


def test_sync_gmail_skips_non_job_related(tmp_path):
    tracker = make_tracker_with_job(tmp_path, status="applied")
    gmail = FakeGmailClient([make_email()])
    provider = QueueProvider([make_classification(is_job_related=False, category="other")])

    result = sync_gmail(provider, gmail, tracker)

    assert result.updated == []
    assert result.unmatched_subjects == []


def test_sync_gmail_records_unmatched_subject_when_no_company_match(tmp_path):
    tracker = make_tracker_with_job(tmp_path, status="applied", company="Acme Corp")
    gmail = FakeGmailClient([make_email(subject="Mystery email")])
    provider = QueueProvider([make_classification(company_guess="Totally Different Co")])

    result = sync_gmail(provider, gmail, tracker)

    assert result.updated == []
    assert result.unmatched_subjects == ["Mystery email"]


def test_sync_gmail_logs_unmatched_subjects_to_the_audit_trail(tmp_path):
    """Same gap as the low-confidence case above - an unmatched email
    previously left no trace in the audit trail, only in the current run's
    own printed output.
    """
    tracker = make_tracker_with_job(tmp_path, status="applied", company="Acme Corp")
    gmail = FakeGmailClient([make_email(subject="Mystery email")])
    provider = QueueProvider([make_classification(company_guess="Totally Different Co")])
    audit = AuditLogger(tmp_path / "audit.log")

    sync_gmail(provider, gmail, tracker, audit=audit)

    log_text = (tmp_path / "audit.log").read_text()
    assert "gmail_sync_unmatched" in log_text
    assert "Mystery email" in log_text
    assert "Totally Different Co" in log_text


def test_sync_gmail_never_updates_a_terminal_status(tmp_path):
    tracker = make_tracker_with_job(tmp_path, status="offer")
    gmail = FakeGmailClient([make_email()])
    provider = QueueProvider([make_classification(category="rejection")])

    result = sync_gmail(provider, gmail, tracker)

    assert result.updated == []
    assert tracker.get_job("job1")["status"] == "offer"


@pytest.mark.parametrize(
    ("category", "expected"),
    [("interview_invite", "interviewing"), ("offer", "offer"), ("rejection", "rejected")],
)
def test_sync_gmail_a_late_reply_updates_a_no_response_job(tmp_path, category, expected):
    """no_response only means no reply had arrived yet - `job-bot
    mark-stale` sets it in bulk, and late replies are common. A real reply
    must still land on it; it used to be treated as terminal and ignored.
    """
    tracker = make_tracker_with_job(tmp_path, status="no_response")
    gmail = FakeGmailClient([make_email()])
    provider = QueueProvider([make_classification(category=category)])

    result = sync_gmail(provider, gmail, tracker)

    assert result.updated == [("job1", "Acme Corp", expected)]
    assert tracker.get_job("job1")["status"] == expected


def test_sync_gmail_a_confirmation_email_does_not_reopen_a_no_response_job(tmp_path):
    """An "application received" email isn't a reply to the application."""
    tracker = make_tracker_with_job(tmp_path, status="no_response")
    gmail = FakeGmailClient([make_email()])
    provider = QueueProvider([make_classification(category="application_confirmation")])

    result = sync_gmail(provider, gmail, tracker)

    assert result.updated == []
    assert tracker.get_job("job1")["status"] == "no_response"


@pytest.mark.parametrize("status", ["offer", "rejected", "withdrawn"])
def test_sync_gmail_still_never_touches_a_genuinely_final_status(tmp_path, status):
    tracker = make_tracker_with_job(tmp_path, status=status)
    gmail = FakeGmailClient([make_email()])
    provider = QueueProvider([make_classification(category="interview_invite")])

    assert sync_gmail(provider, gmail, tracker).updated == []
    assert tracker.get_job("job1")["status"] == status


def test_sync_gmail_never_updates_a_skipped_job(tmp_path):
    """A "skipped" job means the bot chose not to apply - there's no real
    application to correlate an email with, so it must never be touched by
    gmail-sync even on a plausible-looking company match (unlike "seen",
    which also ranks 0 but legitimately can be moved forward by e.g. an
    application_confirmation email arriving before the tracker's own upsert
    caught up).
    """
    tracker = make_tracker_with_job(tmp_path, status="skipped")
    gmail = FakeGmailClient([make_email()])
    provider = QueueProvider([make_classification(category="rejection")])

    result = sync_gmail(provider, gmail, tracker)

    assert result.updated == []
    assert tracker.get_job("job1")["status"] == "skipped"


def test_sync_gmail_application_confirmation_stamps_applied_at(tmp_path):
    """A job that's only ever been "seen" (never run through mark_applied())
    can legitimately be moved to "applied" by an application_confirmation
    email - e.g. it arrived before the tracker's own upsert caught up, or
    the application went out through a path the bot didn't observe. That
    transition must stamp applied_at, or has_applied() (which the run
    loop's dedup check and report --stale-days both rely on) stays False
    for a job the tracker itself now calls "applied".
    """
    tracker = make_tracker_with_job(tmp_path, status="seen")
    gmail = FakeGmailClient([make_email()])
    provider = QueueProvider([make_classification(category="application_confirmation")])

    result = sync_gmail(provider, gmail, tracker)

    assert result.updated == [("job1", "Acme Corp", "applied")]
    job = tracker.get_job("job1")
    assert job["status"] == "applied"
    assert job["applied_at"] is not None
    assert tracker.has_applied("job1") is True


def test_sync_gmail_never_moves_status_backward(tmp_path):
    tracker = make_tracker_with_job(tmp_path, status="interviewing")
    gmail = FakeGmailClient([make_email()])
    provider = QueueProvider([make_classification(category="application_confirmation")])

    result = sync_gmail(provider, gmail, tracker)

    assert result.updated == []
    assert tracker.get_job("job1")["status"] == "interviewing"


def test_sync_gmail_query_includes_days_window(tmp_path):
    tracker = make_tracker_with_job(tmp_path)
    gmail = FakeGmailClient([])
    provider = QueueProvider([])

    sync_gmail(provider, gmail, tracker, days=30)

    assert "newer_than:30d" in gmail.last_query


def test_sync_gmail_is_a_no_op_when_the_status_would_not_actually_change(tmp_path):
    """A job already "interviewing" getting another interview_invite email
    (e.g. scheduling a second round) must not re-record the same status as
    a fresh "update" - _should_update()'s same-status branch.
    """
    tracker = make_tracker_with_job(tmp_path, status="interviewing")
    gmail = FakeGmailClient([make_email()])
    provider = QueueProvider([make_classification(category="interview_invite")])

    result = sync_gmail(provider, gmail, tracker)

    assert result.updated == []
    assert tracker.get_job("job1")["status"] == "interviewing"


def test_sync_gmail_skips_a_category_with_no_mapped_status(tmp_path):
    """Defense in depth: CATEGORY_TO_STATUS covers every non-"other"
    EmailCategory value today, so this path isn't reachable through a real,
    schema-validated classification - but if EmailCategory ever grows a new
    category without a matching CATEGORY_TO_STATUS entry, sync_gmail must
    skip it rather than crash. model_construct() bypasses Pydantic's Literal
    validation to simulate exactly that future-mismatch scenario.
    """
    tracker = make_tracker_with_job(tmp_path, status="applied")
    gmail = FakeGmailClient([make_email()])
    unmapped = EmailClassification.model_construct(
        is_job_related=True,
        category="some_future_category",
        company_guess="Acme",
        role_guess="Engineer",
        confidence=0.9,
    )
    provider = QueueProvider([unmapped])

    result = sync_gmail(provider, gmail, tracker)

    assert result.updated == []
    assert tracker.get_job("job1")["status"] == "applied"
