import argparse
import sys

from job_bot.browser.linkedin_adapter import LinkedInAdapter
from job_bot.browser.session import browser_session
from job_bot.config import Settings, SettingsError, get_settings
from job_bot.dashboard.server import run_dashboard
from job_bot.generation.artifacts import UnsafeJobId, write_cover_letter, write_tailored_resume
from job_bot.generation.cover_letter import generate_cover_letter
from job_bot.generation.qa_answerer import answer_question
from job_bot.generation.resume_tailor import tailor_resume
from job_bot.integrations.gmail_client import GmailClient, GmailClientError
from job_bot.integrations.gmail_sync import sync_gmail
from job_bot.llm.claude_provider import ClaudeProviderError
from job_bot.llm.factory import get_provider
from job_bot.llm.ollama_provider import OllamaProviderError
from job_bot.logging_setup import configure_logging
from job_bot.matching.scorer import score_job_match
from job_bot.models.schemas import JobMatchScore
from job_bot.resume.parser import ResumeParseError
from job_bot.resume.store import ResumeStore
from job_bot.safety.audit_log import AuditLogger
from job_bot.safety.blacklist import CompanyBlacklist
from job_bot.safety.confirm import SubmitConfirmer
from job_bot.safety.rate_limiter import RateLimiter
from job_bot.tracker.db import TRACKER_STATUSES, InvalidStatus, Tracker

EXPECTED_ERRORS = (
    ClaudeProviderError,
    OllamaProviderError,
    ResumeParseError,
    SettingsError,
    InvalidStatus,
    GmailClientError,
    UnsafeJobId,
)


def _apply_provider_overrides(settings: Settings, args: argparse.Namespace) -> None:
    if getattr(args, "provider", None):
        settings.llm_provider = args.provider
    if getattr(args, "model", None):
        if settings.llm_provider == "claude":
            settings.claude_model = args.model
        else:
            settings.ollama_model = args.model


def cmd_login(settings: Settings) -> None:
    with browser_session(settings.browser_profile_dir, headless=False) as context:
        page = context.new_page()
        page.goto("https://www.linkedin.com/login")
        print(
            "A browser window has opened. Log in to LinkedIn manually, then "
            "press Enter here once you're logged in."
        )
        input()
    print("Session saved to", settings.browser_profile_dir)


def cmd_run(settings: Settings, args: argparse.Namespace) -> None:
    for warning in settings.validate_ready():
        print(f"Warning: {warning}")

    provider = get_provider(settings)
    resume_store = ResumeStore(settings.resume_path, settings.faq_path)
    rate_limiter = RateLimiter(settings.db_path, settings.effective_daily_cap())
    blacklist = CompanyBlacklist(settings.blacklist_path)
    confirmer = SubmitConfirmer(
        required=settings.require_confirm_before_submit and not args.yes_i_understand_the_risk
    )
    audit = AuditLogger(settings.audit_log_path)
    tracker = Tracker(settings.db_path)

    resume_text = resume_store.resume_text()

    with browser_session(settings.browser_profile_dir, headless=args.headless) as context:
        page = context.new_page()
        adapter = LinkedInAdapter(page)
        postings = adapter.search(args.keywords, args.location, max_results=args.search_pool)
        audit.log("search", keywords=args.keywords, location=args.location, results=len(postings))

        applied = 0
        for posting in postings:
            if applied >= args.max_apps:
                break
            if rate_limiter.remaining_today() <= 0:
                print("Daily application cap reached.")
                break
            if tracker.has_applied(posting.job_id):
                continue
            if blacklist.is_blocked(posting.company):
                audit.log("skip_blacklisted", job_id=posting.job_id, company=posting.company)
                continue

            description = adapter.load_description(posting)
            match: JobMatchScore = score_job_match(provider, resume_text, description)
            tracker.upsert_job(posting.job_id, posting.title, posting.company, posting.url, match.score)
            audit.log("scored", job_id=posting.job_id, score=match.score, should_apply=match.should_apply)

            if not match.should_apply:
                tracker.mark_skipped(posting.job_id)
                continue

            tailored = tailor_resume(provider, resume_text, description)
            cover_letter = generate_cover_letter(provider, resume_text, description, posting.company)
            write_tailored_resume(settings.applications_dir, posting.job_id, tailored)
            write_cover_letter(settings.applications_dir, posting.job_id, cover_letter)
            audit.log("generated_materials", job_id=posting.job_id)

            def answer(question: str, job_id: str = posting.job_id) -> str:
                result = answer_question(provider, resume_text, resume_store.faq_answers(), question)
                tracker.record_qa(job_id, question, result.answer)
                return result.answer

            if not confirmer.confirm(f"Apply to {posting.title} at {posting.company}?"):
                audit.log("user_declined", job_id=posting.job_id)
                continue

            try:
                submitted = adapter.fill_and_submit(
                    posting,
                    answer_question=answer,
                    resume_path=str(settings.resume_path),
                    cover_letter_text=cover_letter.body,
                    dry_run=args.dry_run,
                )
            except Exception as e:  # noqa: BLE001 - surface and continue to the next job
                audit.log("apply_error", job_id=posting.job_id, error=str(e))
                print(f"Error applying to {posting.title} at {posting.company}: {e}")
                continue

            if submitted:
                rate_limiter.record_application()
                tracker.mark_applied(posting.job_id)
                audit.log("applied", job_id=posting.job_id, company=posting.company)
                applied += 1
                print(f"Applied: {posting.title} at {posting.company}")
            else:
                audit.log("dry_run_stopped", job_id=posting.job_id)
                print(f"[dry-run] Would apply to {posting.title} at {posting.company}")

        print(f"Done. Applied to {applied} job(s). {rate_limiter.remaining_today()} remaining today.")


def cmd_status(settings: Settings, args: argparse.Namespace) -> None:
    tracker = Tracker(settings.db_path)
    try:
        tracker.update_status(args.job_id, args.status)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"{args.job_id} -> {args.status}")


def cmd_report(settings: Settings) -> None:
    tracker = Tracker(settings.db_path)
    counts = tracker.status_counts()
    if not counts:
        print("No jobs tracked yet.")
        return
    width = max(len(status) for status in counts)
    for status in sorted(counts):
        print(f"{status:<{width}}  {counts[status]}")
    print(f"{'total':<{width}}  {sum(counts.values())}")


def cmd_gmail_sync(settings: Settings, args: argparse.Namespace) -> None:
    provider = get_provider(settings)
    gmail_client = GmailClient(settings.gmail_credentials_path, settings.gmail_token_path)
    tracker = Tracker(settings.db_path)
    audit = AuditLogger(settings.audit_log_path)

    result = sync_gmail(
        provider,
        gmail_client,
        tracker,
        days=args.days if args.days is not None else settings.gmail_sync_days,
        max_emails=args.max_emails,
        confidence_threshold=settings.gmail_match_confidence,
        dry_run=args.dry_run,
        audit=audit,
    )

    print(f"Scanned {result.total_emails} email(s).")
    for job_id, company, new_status in result.updated:
        prefix = "[dry-run] Would update" if args.dry_run else "Updated"
        print(f"{prefix}: {company} ({job_id}) -> {new_status}")
    if result.skipped_low_confidence:
        print(f"Skipped {result.skipped_low_confidence} low-confidence email(s).")
    if result.unmatched_subjects:
        print("Job-related but couldn't confidently match to a tracked application:")
        for subject in result.unmatched_subjects:
            print(f"  - {subject}")


def cmd_dashboard(settings: Settings, args: argparse.Namespace) -> None:
    port = args.port if args.port is not None else settings.dashboard_port
    run_dashboard(settings.db_path, port=port, open_browser=not args.no_open)


def cmd_test_provider(settings: Settings) -> None:
    provider = get_provider(settings)
    result = provider.generate_structured(
        system="You are a test.",
        prompt=(
            "Respond as if evaluating a strong resume match: score 90, "
            "should_apply true, no missing qualifications, brief reasoning."
        ),
        schema=JobMatchScore,
    )
    print(f"Provider OK: {settings.llm_provider}")
    print(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="job_bot")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("login", help="Open a browser to log into LinkedIn once; the session persists locally.")

    run_p = sub.add_parser("run", help="Search, score, tailor, and apply to jobs.")
    run_p.add_argument("--keywords", default="software engineer")
    run_p.add_argument("--location", default="United States")
    run_p.add_argument("--max-apps", type=int, default=5)
    run_p.add_argument(
        "--search-pool",
        type=int,
        default=25,
        help="How many Easy-Apply postings to fetch and score before filtering down to --max-apps.",
    )
    run_p.add_argument("--dry-run", action="store_true", help="Stop right before the final Submit click.")
    run_p.add_argument("--headless", action="store_true", help="Run the browser without a visible window.")
    run_p.add_argument("--provider", choices=["claude", "ollama"], default=None)
    run_p.add_argument("--model", default=None)
    run_p.add_argument(
        "--yes-i-understand-the-risk",
        action="store_true",
        help="Skip the per-application confirmation prompt. The daily cap still applies.",
    )

    sub.add_parser("test-provider", help="Sanity-check the configured LLM provider with one call.")

    status_p = sub.add_parser(
        "status", help="Record an application outcome (interviewing, offer, rejected, ...) by hand."
    )
    status_p.add_argument(
        "job_id", help="The LinkedIn job id, as shown in `job-bot report` or the audit log."
    )
    status_p.add_argument("status", choices=sorted(TRACKER_STATUSES))

    sub.add_parser("report", help="Print a count of tracked jobs by status.")

    gmail_p = sub.add_parser(
        "gmail-sync",
        help="Scan recent Gmail for replies to tracked applications and update their status.",
    )
    gmail_p.add_argument(
        "--days", type=int, default=None, help="How far back to search (default: from .env)."
    )
    gmail_p.add_argument("--max-emails", type=int, default=50)
    gmail_p.add_argument(
        "--dry-run", action="store_true", help="Report what would change without writing it."
    )
    gmail_p.add_argument("--provider", choices=["claude", "ollama"], default=None)
    gmail_p.add_argument("--model", default=None)

    dashboard_p = sub.add_parser("dashboard", help="Serve a live one-page view of the tracker at localhost.")
    dashboard_p.add_argument("--port", type=int, default=None, help="default: from .env (8765)")
    dashboard_p.add_argument("--no-open", action="store_true", help="Don't auto-open a browser tab.")

    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    settings = get_settings()

    if args.command in ("run", "gmail-sync"):
        _apply_provider_overrides(settings, args)

    try:
        if args.command == "login":
            cmd_login(settings)
        elif args.command == "run":
            cmd_run(settings, args)
        elif args.command == "test-provider":
            cmd_test_provider(settings)
        elif args.command == "status":
            cmd_status(settings, args)
        elif args.command == "report":
            cmd_report(settings)
        elif args.command == "gmail-sync":
            cmd_gmail_sync(settings, args)
        elif args.command == "dashboard":
            cmd_dashboard(settings, args)
    except EXPECTED_ERRORS as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
