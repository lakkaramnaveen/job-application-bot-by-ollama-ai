from job_bot.integrations.gmail_client import EmailMessage
from job_bot.integrations.gmail_sync import find_matching_job, sync_gmail
from job_bot.llm.base import LLMProvider
from job_bot.models.schemas import EmailClassification
from job_bot.safety.audit_log import AuditLogger
from job_bot.tracker.db import Tracker


class QueueProvider(LLMProvider):
    """Returns one canned EmailClassification per call, in order."""

    def __init__(self, results: list[EmailClassification]):
        self._results = list(results)
        self.calls = 0

    def generate_structured(self, *, system, prompt, schema):
        self.calls += 1
        return self._results.pop(0)


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


def test_sync_gmail_skips_low_confidence(tmp_path):
    tracker = make_tracker_with_job(tmp_path, status="applied")
    gmail = FakeGmailClient([make_email()])
    provider = QueueProvider([make_classification(confidence=0.2)])

    result = sync_gmail(provider, gmail, tracker, confidence_threshold=0.6)

    assert result.updated == []
    assert result.skipped_low_confidence == 1
    assert tracker.get_job("job1")["status"] == "applied"


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


def test_sync_gmail_never_updates_a_terminal_status(tmp_path):
    tracker = make_tracker_with_job(tmp_path, status="offer")
    gmail = FakeGmailClient([make_email()])
    provider = QueueProvider([make_classification(category="rejection")])

    result = sync_gmail(provider, gmail, tracker)

    assert result.updated == []
    assert tracker.get_job("job1")["status"] == "offer"


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
