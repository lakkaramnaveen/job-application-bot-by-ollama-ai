"""The `job-bot` command-line entry point.

Each `cmd_*` function implements one subcommand and is wired to it in two
places: an `add_parser`/`add_argument` block in build_parser(), and a branch
in main()'s dispatch. `cmd_run` is the core flow (search -> score -> tailor
-> apply) and the one place safety modules (safety/) compose together;
every other command is a thinner read/write against the tracker DB or a
single integration.

EXPECTED_ERRORS are exceptions any command may raise as an expected,
user-facing failure (bad config, an unreachable API, ...) - main() catches
just this list and prints the message instead of a traceback. Anything else
propagating out of a command is a real bug.
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from job_bot.browser.base_adapter import JobPosting
from job_bot.browser.external_apply_adapter import ExternalApplyAdapter
from job_bot.browser.linkedin_adapter import (
    EXPERIENCE_LEVEL_CODES,
    LinkedInAdapter,
    LinkedInSignedOut,
    UnansweredRequiredQuestion,
)
from job_bot.browser.session import BrowserSessionError, browser_session
from job_bot.config import HARD_DAILY_APPLICATION_CEILING, Settings, SettingsError, get_settings
from job_bot.dashboard.server import DashboardPortInUse, run_dashboard
from job_bot.data_files import CorruptDataFile, assert_safe_to_overwrite
from job_bot.generation.artifacts import (
    UnsafeJobId,
    write_cover_letter,
    write_tailored_resume,
    write_tailored_resume_docx,
)
from job_bot.generation.cover_letter import generate_cover_letter
from job_bot.generation.qa_answerer import answer_question
from job_bot.generation.resume_tailor import tailor_resume
from job_bot.integrations.gmail_client import GmailClient, GmailClientError
from job_bot.integrations.gmail_sync import sync_gmail
from job_bot.llm.base import LLMProvider
from job_bot.llm.claude_provider import ClaudeProviderError
from job_bot.llm.factory import get_provider
from job_bot.llm.ollama_provider import OllamaProviderError, quit_ollama
from job_bot.logging_setup import configure_logging
from job_bot.matching.scorer import score_job_match
from job_bot.models.schemas import CoverLetter, JobMatchScore, TailoredResume
from job_bot.resume.parser import ResumeParseError, parse_resume
from job_bot.resume.store import ResumeStore, unusable_faq_reason
from job_bot.safety.answer_gaps import AnswerGapStore
from job_bot.safety.audit_log import AuditLogger
from job_bot.safety.blacklist import CompanyBlacklist
from job_bot.safety.confirm import SubmitConfirmer
from job_bot.safety.rate_limiter import DailyCapReached, RateLimiter
from job_bot.text_utils import is_echoed_question, is_per_position_field, normalize_company_name
from job_bot.tracker.db import (
    TRACKER_STATUSES,
    InvalidStatus,
    Tracker,
    write_export_csv,
    write_export_json,
)

EXPECTED_ERRORS = (
    ClaudeProviderError,
    OllamaProviderError,
    ResumeParseError,
    SettingsError,
    InvalidStatus,
    GmailClientError,
    UnsafeJobId,
    BrowserSessionError,
    DashboardPortInUse,
    CorruptDataFile,
)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _apply_provider_overrides(settings: Settings, args: argparse.Namespace) -> None:
    """Let `--provider`/`--model` on `run`/`gmail-sync` override .env for
    this invocation only - settings is a fresh in-memory instance per
    process, so this never writes back to .env.
    """
    if getattr(args, "provider", None):
        settings.llm_provider = args.provider
    if getattr(args, "model", None):
        if settings.llm_provider == "claude":
            settings.claude_model = args.model
        else:
            settings.ollama_model = args.model


LOGIN_WAIT_TIMEOUT_MS = 600_000  # 10 minutes - generous for 2FA/security checkpoints


def _login_finished(url: str) -> bool:
    """True once the browser has navigated away from both the login form and
    a security checkpoint/challenge page - either one still means the user
    hasn't finished authenticating yet.
    """
    return "linkedin.com/login" not in url and "/checkpoint/" not in url


def cmd_login(settings: Settings) -> None:
    """Open a browser window for the user to log into LinkedIn by hand - the
    bot never sees or stores the password itself. What it's a window *into*
    depends on settings.browser_cdp_url (see browser_session()'s docstring):
    by default, job-bot's own isolated Chromium profile, whose session
    cookies persist under settings.browser_profile_dir and get reused by
    every future `job-bot run` - run this once. With browser_cdp_url set,
    there's nothing to "save" here at all: it's attaching to whatever
    already-running Chrome that URL points at, so if that's already logged
    into LinkedIn this finishes immediately.

    Waits for the page to navigate away from the login/checkpoint flow on its
    own rather than blocking on input() for an Enter keypress - input()
    requires a terminal with live interactive stdin attached to *this*
    process, which isn't true in every environment this can be launched from
    (an agent's shell, a remote dev box, ...), and there raised an unhandled
    EOFError instead of ever giving the user a chance to log in.
    """
    with browser_session(
        settings.browser_profile_dir, headless=False, cdp_url=settings.browser_cdp_url
    ) as context:
        page = context.new_page()
        page.goto("https://www.linkedin.com/login")
        print(
            "A browser window has opened. Log in to LinkedIn manually - this "
            "continues on its own once you're done, no need to come back here."
        )
        try:
            page.wait_for_url(_login_finished, timeout=LOGIN_WAIT_TIMEOUT_MS)
        except PlaywrightTimeoutError:
            print(
                f"Still on the login page after {LOGIN_WAIT_TIMEOUT_MS // 60_000} "
                "minutes - closing without saving. Run `job-bot login` again when ready."
            )
            return
        except PlaywrightError as e:
            # The browser/tab closed out from under the wait - the user
            # closed the window, the browser crashed, or (in some sandboxed
            # shells) the process itself got killed before login finished.
            # Not the same as the timeout above, and not a bug to surface
            # as a traceback: nothing was saved, so just say so plainly.
            print(f"Browser closed before login finished ({e}). Run `job-bot login` again.")
            return
    if settings.browser_cdp_url:
        print("Logged in. `job-bot run` will reuse this same Chrome session.")
    else:
        print("Session saved to", settings.browser_profile_dir)


def cmd_run(settings: Settings, args: argparse.Namespace) -> None:
    """The main loop: search Easy-Apply postings, score each against the
    resume, generate tailored materials for the ones worth applying to, and
    submit (unless --dry-run). Every safety mechanism in safety/ is composed
    here - the daily cap (rate_limiter), the confirmation prompt (confirm),
    the blacklist, and the audit log - so this is the one place to read to
    understand what actually happens during a real run.
    """
    for warning in settings.validate_ready():
        print(f"Warning: {warning}")

    # CompanyBlacklist loads an unreadable file as empty (so other commands
    # keep working while `job-bot doctor` reports it) - for a run, that
    # would mean silently applying to every company the user blocked. Stop
    # before anything is searched or submitted instead.
    try:
        assert_safe_to_overwrite(settings.blacklist_path, list)
    except CorruptDataFile as e:
        print(
            f"Error: {settings.blacklist_path} could not be read ({e.reason}), so no company would be "
            "blocked this run - refusing to start. Fix or move the file aside, then re-run "
            "(`job-bot doctor` checks it).",
            file=sys.stderr,
        )
        sys.exit(1)

    provider = get_provider(settings)
    resume_store = ResumeStore(settings.resume_path, settings.faq_path)
    rate_limiter = RateLimiter(settings.db_path, settings.effective_daily_cap())
    blacklist = CompanyBlacklist(settings.blacklist_path)
    confirmer = SubmitConfirmer(
        required=settings.require_confirm_before_submit and not args.yes_i_understand_the_risk
    )
    # EXPERIMENTAL external-apply path (see Settings.enable_external_apply)
    # always confirms before submitting, with no override - it's far less
    # tested than the LinkedIn flow, on a form structure this hasn't been
    # tuned against at all, so --yes-i-understand-the-risk and
    # REQUIRE_CONFIRM_BEFORE_SUBMIT=false intentionally don't reach it.
    external_confirmer = SubmitConfirmer(required=True)
    audit = AuditLogger(settings.audit_log_path)
    failure_log = AuditLogger(settings.failed_applications_log_path)
    answer_gaps = AnswerGapStore(settings.answer_gaps_path)
    tracker = Tracker(settings.db_path)

    min_score = args.min_score if args.min_score is not None else settings.min_match_score
    raw_exclude = (
        args.exclude_title_keywords
        if args.exclude_title_keywords is not None
        else settings.exclude_title_keywords
    )
    exclude_keywords = [kw.casefold() for kw in _split_csv(raw_exclude)]
    raw_levels = args.experience_level if args.experience_level is not None else settings.default_experience_levels
    experience_levels = _split_csv(raw_levels) or None
    if experience_levels:
        invalid = [level for level in experience_levels if level not in EXPERIENCE_LEVEL_CODES]
        if invalid:
            print(
                f"Error: unknown --experience-level value(s): {', '.join(invalid)}. "
                f"Valid: {', '.join(sorted(EXPERIENCE_LEVEL_CODES))}",
                file=sys.stderr,
            )
            sys.exit(1)

    max_years_experience = (
        args.max_years_experience if args.max_years_experience is not None else settings.max_years_experience
    )
    require_w2 = args.require_w2 or settings.require_w2
    include_external = args.include_external_apply or settings.enable_external_apply

    with browser_session(
        settings.browser_profile_dir, headless=args.headless, cdp_url=settings.browser_cdp_url
    ) as context:
        page = context.new_page()
        adapter = LinkedInAdapter(page)

        def run_one_cycle() -> tuple[int, int, bool]:
            # Re-fetched every cycle, not captured once before the loop:
            # --loop can run for many hours, and ResumeStore.resume_text()
            # re-parses only if the file's mtime actually changed since the
            # last cycle, so this stays cheap while still picking up a
            # resume edited/re-exported mid-loop on the very next cycle.
            return _run_apply_cycle(
                adapter=adapter,
                page=page,
                provider=provider,
                resume_store=resume_store,
                resume_text=resume_store.resume_text(),
                tracker=tracker,
                rate_limiter=rate_limiter,
                blacklist=blacklist,
                confirmer=confirmer,
                external_confirmer=external_confirmer,
                audit=audit,
                failure_log=failure_log,
                answer_gaps=answer_gaps,
                settings=settings,
                args=args,
                min_score=min_score,
                exclude_keywords=exclude_keywords,
                experience_levels=experience_levels,
                max_years_experience=max_years_experience,
                require_w2=require_w2,
                include_external=include_external,
            )

        if not args.loop:
            applied, failed, _fatal_error = run_one_cycle()
            _print_cycle_summary(applied, failed, rate_limiter, settings)
            if rate_limiter.remaining_today() <= 0:
                _quit_ollama_if_configured(settings)
            return

        print(
            "Loop mode: searching and applying back-to-back until today's application cap is "
            f"reached, pausing {args.loop_interval_minutes} minute(s) between cycles only when a "
            "cycle applies to nothing, or you stop it (Ctrl+C)."
        )
        try:
            while True:
                applied, failed, fatal_error = run_one_cycle()
                _print_cycle_summary(applied, failed, rate_limiter, settings)
                if fatal_error:
                    # Same reasoning _is_ollama_unreachable()/
                    # _is_claude_misconfigured()/LinkedInSignedOut already
                    # apply within one cycle, one level up: retrying every
                    # loop_interval_minutes against a provider that's still
                    # down (or a signed-out LinkedIn session) won't fix
                    # itself, and looked identical to an ordinary quiet
                    # cycle ("Nothing to apply to this cycle - sleeping...")
                    # before this fix - confirmed live: a run with Ollama
                    # down retried the identical failure every 20 minutes
                    # until manually interrupted, with nothing distinguishing
                    # it from a normal "no eligible postings" cycle.
                    print(
                        "Stopping the loop - this cycle hit a problem (see above) that retrying "
                        "on a timer won't fix on its own. Fix it and re-run job-bot when ready."
                    )
                    break
                if rate_limiter.remaining_today() <= 0:
                    print("Daily application cap reached for today - stopping.")
                    _quit_ollama_if_configured(settings)
                    break
                if page.is_closed():
                    print("Browser window was closed - stopping.")
                    break
                if applied == 0:
                    # Nothing applied this cycle (no eligible postings found,
                    # or the search itself failed) - back off rather than
                    # re-hammering LinkedIn's search immediately. When the
                    # cycle DID apply to something, there's a real cap
                    # remaining and likely more eligible postings right
                    # behind it, so the next cycle starts right away instead
                    # of idling for loop_interval_minutes for no reason.
                    print(f"Nothing to apply to this cycle - sleeping {args.loop_interval_minutes} minute(s)...")
                    time.sleep(args.loop_interval_minutes * 60)
                else:
                    print("Applications remaining today - searching again immediately...")
        except KeyboardInterrupt:
            print("\nStopped.")


def _print_cycle_summary(applied: int, failed: int, rate_limiter: RateLimiter, settings: Settings) -> None:
    print(f"Done. Applied to {applied} job(s). {rate_limiter.remaining_today()} remaining today.")
    if failed:
        print(
            f"{failed} posting(s) could not be completed - run `job-bot audit-log --failures` "
            f"(or see {settings.failed_applications_log_path} directly) for what happened and why."
        )


def _is_ollama_unreachable(e: Exception) -> bool:
    """True for the specific OllamaProviderError raised when the local
    server can't be connected to at all (see ollama_provider.py's
    httpx.ConnectError handling) - deliberately narrower than "any
    OllamaProviderError", since the other two cases it covers (a malformed
    JSON response after retries, a 404 for an unpulled model) aren't
    necessarily going to fail identically on every remaining posting the
    way a fully unreachable server is. Checked by message rather than a
    dedicated exception subclass since that string is this project's own
    and unlikely to drift without both sides being updated together.
    """
    return isinstance(e, OllamaProviderError) and "Could not reach Ollama" in str(e)


# The three claude_provider.py ClaudeProviderError messages that mean the
# provider is fundamentally misconfigured - not a one-off request failure
# (rate limit, network blip) - and will therefore raise the exact same
# error on every remaining posting in the batch too. Same reasoning as
# _is_ollama_unreachable() above, and the same fix: recognize it after the
# first failure instead of repeating an identical, guaranteed-to-fail LLM
# call (and an identical printed error) once per remaining posting.
_CLAUDE_CONFIG_ERROR_MARKERS = (
    "Invalid ANTHROPIC_API_KEY",
    "API key lacks permission",
    "not found.",  # ClaudeProviderError(f"Model '{model}' not found.")
)


def _is_claude_misconfigured(e: Exception) -> bool:
    return isinstance(e, ClaudeProviderError) and any(
        marker in str(e) for marker in _CLAUDE_CONFIG_ERROR_MARKERS
    )


def _quit_ollama_if_configured(settings: Settings) -> None:
    """Called once `job-bot run` is done drawing on Ollama for the day (the
    daily cap was reached) - see Settings.quit_ollama_when_done. No-op for
    the Claude provider, and when the setting is off (its default).
    """
    if settings.llm_provider != "ollama" or not settings.quit_ollama_when_done:
        return
    if quit_ollama():
        print("Daily cap reached - quit Ollama.")
    else:
        print("Daily cap reached - could not quit Ollama (it may already be stopped).")


def _run_apply_cycle(
    *,
    adapter: LinkedInAdapter,
    page: Page,
    provider: LLMProvider,
    resume_store: ResumeStore,
    resume_text: str,
    tracker: Tracker,
    rate_limiter: RateLimiter,
    blacklist: CompanyBlacklist,
    confirmer: SubmitConfirmer,
    external_confirmer: SubmitConfirmer,
    audit: AuditLogger,
    failure_log: AuditLogger,
    answer_gaps: AnswerGapStore,
    settings: Settings,
    args: argparse.Namespace,
    min_score: int,
    exclude_keywords: list[str],
    experience_levels: list[str] | None,
    max_years_experience: int | None,
    require_w2: bool,
    include_external: bool,
) -> tuple[int, int, bool]:
    """One search -> score -> tailor -> apply pass over a fresh batch of
    postings. Called once for a plain `job-bot run`, or repeatedly for
    `--loop` (back-to-back with no sleep as long as each cycle keeps
    applying to something; only a cycle that applies to nothing pauses
    before the next one) - re-running search() each cycle is what lets loop
    mode pick up postings that appeared after the previous cycle, not just
    the ones visible at process start. Returns (applied, failed,
    fatal_error) for that cycle only, not a running total across
    cycles.

    fatal_error is True when this cycle stopped early because the
    LLM provider itself was unreachable/misconfigured (Ollama down, Claude
    misconfigured - see _is_ollama_unreachable()/_is_claude_misconfigured()
    below) or LinkedIn's session had expired (LinkedInSignedOut), not just
    "nothing worth applying to this cycle". --loop mode's
    caller uses this to stop the loop entirely instead of treating it like
    an ordinary quiet cycle and sleeping loop_interval_minutes before
    silently retrying against the same still-unreachable provider -
    confirmed live: a run with Ollama down correctly stopped *this* cycle's
    posting loop early, but --loop then printed "Nothing to apply to this
    cycle - sleeping 20 minute(s)..." and retried the identical failure
    every 20 minutes until manually interrupted, indistinguishable from a
    normal cycle that just found no eligible postings.
    """
    try:
        postings = adapter.search(
            args.keywords,
            args.location,
            max_results=args.search_pool,
            experience_levels=experience_levels,
            include_external=include_external,
        )
    except Exception as e:  # noqa: BLE001 - a search failure should cost this cycle, not crash the whole run/loop
        # Real bug this guards against: search() itself (not yet a specific
        # posting) failing - e.g. LinkedIn briefly rate-limiting/erroring on
        # the search results page itself after _goto_with_retry()'s own
        # retries are exhausted - propagated straight out of this function
        # uncaught. In --loop mode especially, that crashed the entire
        # unattended run instead of just costing this one cycle, exactly
        # the failure mode --loop exists to run through unattended over
        # many hours. Confirmed live before this fix.
        audit.log("search_error", keywords=args.keywords, location=args.location, error=str(e))
        failure_log.log("search_error", keywords=args.keywords, location=args.location, error=str(e))
        print(f"Error searching for postings: {e}")
        # A signed-out session fails every search identically until the
        # user runs `job-bot login` - fatal the same way a down provider is.
        return 0, 1, isinstance(e, LinkedInSignedOut)
    audit.log("search", keywords=args.keywords, location=args.location, results=len(postings))

    def should_skip(posting: JobPosting) -> bool:
        """Cheap, deterministic reasons to pass over this posting before
        spending an LLM call on it. Doesn't cover the cap/--max-apps
        checks - those stop the whole cycle, not just this one posting,
        so the main loop below handles them directly.
        """
        if tracker.has_applied(posting.job_id):
            return True
        if blacklist.is_blocked(posting.company):
            audit.log("skip_blacklisted", job_id=posting.job_id, company=posting.company)
            return True
        if exclude_keywords and any(kw in posting.title.casefold() for kw in exclude_keywords):
            # Not persisted to the tracker (unlike a real score/skip
            # decision), since the exclude list is expected to change
            # between runs and a posting excluded today should still be
            # re-evaluated normally if it's removed later.
            audit.log("skip_excluded_keyword", job_id=posting.job_id, title=posting.title)
            return True
        return False

    def clears_the_bar(posting: JobPosting, description: str, existing: dict[str, Any] | None) -> bool:
        """Scores this posting fresh, or - if an earlier run already did -
        re-checks that recorded score against *today's* min_score floor
        rather than trusting it outright. That re-check matters because
        the model's own should_apply verdict can't go stale between runs,
        but min_score is user config that can (e.g. tightening
        MIN_MATCH_SCORE in .env after seeing too many weak matches go
        through) - a "seen" status recorded under a looser floor shouldn't
        silently keep clearing a floor that's since been raised.
        """
        if existing is not None and existing["match_score"] is not None:
            if existing["match_score"] < min_score:
                tracker.update_status(posting.job_id, "skipped")
                audit.log("skip_below_min_score", job_id=posting.job_id, score=existing["match_score"])
                return False
            audit.log("reused_score", job_id=posting.job_id, score=existing["match_score"])
            return True

        match: JobMatchScore = score_job_match(
            provider,
            resume_text,
            description,
            max_years_experience=max_years_experience,
            require_w2=require_w2,
        )
        # min_score is an extra floor on top of the model's own
        # should_apply verdict, not a replacement for it - the scorer's
        # eligibility gate (see matching/scorer.py) can still force this
        # to False regardless of score.
        should_apply = match.should_apply and match.score >= min_score
        tracker.record_score(
            posting.job_id,
            posting.title,
            posting.company,
            posting.url,
            match.score,
            should_apply,
            reasoning=match.reasoning,
            eligibility=match.eligibility,
            eligibility_note=match.eligibility_note,
            missing_qualifications=match.missing_qualifications,
        )
        audit.log("scored", job_id=posting.job_id, score=match.score, should_apply=should_apply)
        return should_apply

    def generate_materials(posting: JobPosting, description: str) -> tuple[CoverLetter, str]:
        """Tailors the resume (using past generations that led to a real
        interview/offer as few-shot examples - see
        Tracker.best_resume_examples()) and a cover letter, writes both to
        disk as reference material, and records the generation. The
        returned cover letter's body is what gets filled into the
        application form itself; the returned resume path is what gets
        uploaded as the resume - a freshly tailored .docx when
        write_tailored_resume_docx() could confidently build one (see
        generation/resume_document.py's module docstring for exactly what
        it will and won't change), else the user's own unmodified
        resume_path, unchanged from this project's original behavior.
        """
        examples = [
            TailoredResume(summary=r["summary"], highlighted_skills=r["skills"], bullet_points=r["bullets"])
            for r in tracker.best_resume_examples(limit=3)
        ]
        tailored = tailor_resume(provider, resume_text, description, examples=examples)
        tracker.record_resume_generation(
            posting.job_id,
            posting.title,
            posting.company,
            tailored.summary,
            tailored.highlighted_skills,
            tailored.bullet_points,
        )
        cover_letter = generate_cover_letter(provider, resume_text, description, posting.company)
        write_tailored_resume(
            settings.applications_dir, posting.job_id, tailored, company=posting.company, title=posting.title
        )
        tailored_resume_path = write_tailored_resume_docx(
            settings.applications_dir,
            posting.job_id,
            resume_text,
            tailored,
            company=posting.company,
            title=posting.title,
        )
        write_cover_letter(
            settings.applications_dir, posting.job_id, cover_letter, company=posting.company, title=posting.title
        )
        audit.log("generated_materials", job_id=posting.job_id)
        resume_path = str(tailored_resume_path) if tailored_resume_path is not None else str(settings.resume_path)
        return cover_letter, resume_path

    def answer(question: str, job_id: str) -> str:
        if is_per_position_field(question):
            # No LLM call, nothing recorded or cached - see
            # is_per_position_field(). "" leaves it to the adapter's normal
            # unanswered-required-question handling.
            return ""
        faq_answers = resume_store.faq_answers()
        # An exact-text match against FAQ_PATH is already a curated,
        # confident, resume-grounded answer (see save_faq_answer() below and
        # qa_answerer.py's SYSTEM_PROMPT, which tells the LLM the same
        # thing) - asking the LLM to re-derive it is a guaranteed-redundant
        # round trip on every posting that repeats a question this exact,
        # near-universal on eligibility/sponsorship-style questions asked
        # near-verbatim across many postings. Skipping it matters
        # especially for a local model, where each such call can otherwise
        # cost many seconds for an answer already known. A near-miss
        # (different phrasing/whitespace) still falls through to the LLM
        # unaffected - this only ever short-circuits a literal match.
        cached = faq_answers.get(question)
        # An echoed entry (is_echoed_question()) already in FAQ_PATH from
        # before this check existed is ignored rather than replayed - the
        # LLM gets a fresh attempt instead.
        if cached is not None and not is_echoed_question(question, cached):
            tracker.record_qa(job_id, question, cached)
            return cached
        result = answer_question(
            provider,
            resume_text,
            faq_answers,
            question,
            recent_answers=tracker.recent_qa_pairs(),
        )
        if is_echoed_question(question, result.answer):
            # No answer, not a bad one: "" leaves the field unfilled, which
            # the adapter already reports as an unanswered required
            # question (recorded to answer_gaps for `job-bot
            # review-answers`) - never recorded or cached, so it can't be
            # replayed from FAQ_PATH or reused as a few-shot example.
            return ""
        tracker.record_qa(job_id, question, result.answer)
        if result.based_on_resume and result.confidence >= settings.faq_save_confidence:
            try:
                resume_store.save_faq_answer(question, result.answer)
            except CorruptDataFile as e:
                # Caching is an optimization - the answer itself is still
                # good, so the application goes ahead uncached rather than
                # failing over an unreadable FAQ file.
                print(f"Warning: answer not cached - {e}")
        return result.answer

    def apply_to(posting: JobPosting, cover_letter: CoverLetter, resume_path: str) -> bool | None:
        """Confirms, then submits for real (or stops right before the
        final click on --dry-run). Returns None if the user declined the
        confirmation prompt - not an error, the caller just moves on to
        the next posting silently. Raises on a real failure, including
        UnansweredRequiredQuestion - the caller handles logging/counting
        that exactly like a prep_error. Always closes an external-apply
        popup on the way out, success or failure: LinkedInAdapter hands it
        back (see open_external_application()'s docstring) and has no
        further involvement once it has, so leaving it open here would
        otherwise pile up one tab per external posting across a run.
        """
        active_confirmer = confirmer if posting.easy_apply else external_confirmer
        confirm_prompt = f"Apply to {posting.title} at {posting.company}?"
        if not posting.easy_apply:
            confirm_prompt += " (external site - EXPERIMENTAL)"
        if not active_confirmer.confirm(confirm_prompt):
            audit.log("user_declined", job_id=posting.job_id)
            return None

        def answer_for_this_posting(question: str) -> str:
            return answer(question, posting.job_id)

        external_page = None
        try:
            if posting.easy_apply:
                return adapter.fill_and_submit(
                    posting,
                    answer_question=answer_for_this_posting,
                    resume_path=resume_path,
                    cover_letter_text=cover_letter.body,
                    dry_run=args.dry_run,
                )
            external_page = adapter.open_external_application(posting)
            if external_page is None:
                raise RuntimeError(
                    'Could not find the "Apply on company website" button - the posting '
                    "may have turned out to be Easy Apply after all, or stopped accepting "
                    "applications since it was found."
                )
            return ExternalApplyAdapter(external_page).fill_and_submit(
                answer_question=answer_for_this_posting,
                resume_path=resume_path,
                cover_letter_text=cover_letter.body,
                dry_run=args.dry_run,
            )
        finally:
            if external_page is not None:
                external_page.close()

    applied = 0
    failed = 0
    fatal_error = False
    for posting in postings:
        if applied >= args.max_apps:
            break
        if rate_limiter.remaining_today() <= 0:
            print("Daily application cap reached.")
            break
        if should_skip(posting):
            continue
        existing = tracker.get_job(posting.job_id)
        if existing is not None and existing["status"] != "seen":
            # Already decided against in an earlier run (skipped by the
            # bot, or corrected to a terminal status by hand without
            # ever being applied to) - leave it alone rather than
            # re-scoring it every run.
            continue

        try:
            description = adapter.load_description(posting)
            if not clears_the_bar(posting, description, existing):
                continue
            cover_letter, resume_path = generate_materials(posting, description)
        except Exception as e:  # noqa: BLE001 - one bad posting shouldn't abort the whole run
            audit.log("prep_error", job_id=posting.job_id, error=str(e))
            failure_log.log(
                "prep_error",
                job_id=posting.job_id,
                title=posting.title,
                company=posting.company,
                url=posting.url,
                error=str(e),
            )
            print(f"Error preparing application for {posting.title} at {posting.company}: {e}")
            failed += 1
            if _is_ollama_unreachable(e):
                # Same reasoning as the page.is_closed() check below: every
                # remaining posting draws on the same unreachable Ollama
                # server and would fail identically on it - stop here
                # instead of burning a full LLM-call attempt (and an
                # identical error message) once per remaining posting.
                # Confirmed live: a run with Ollama down logged this same
                # "Could not reach Ollama" prep_error 6 times in a row
                # before this fix.
                print("Ollama is unreachable - stopping the run instead of repeating this for every posting.")
                fatal_error = True
                break
            if _is_claude_misconfigured(e):
                print(f"Claude provider is misconfigured ({e}) - stopping the run instead of repeating this for every posting.")
                fatal_error = True
                break
            if page.is_closed():
                # The browser itself is gone (closed, crashed, killed) -
                # every remaining posting shares this one page and would
                # fail identically on it, so stop here instead of
                # repeating the same failure once per remaining posting.
                print("Browser window was closed - stopping the run.")
                break
            continue

        try:
            submitted = apply_to(posting, cover_letter, resume_path)
        except Exception as e:  # noqa: BLE001 - surface and continue to the next job
            if isinstance(e, UnansweredRequiredQuestion):
                # Log this one specifically, not just as a generic
                # apply_error - see safety/answer_gaps.py: this is what
                # `job-bot review-answers` reads, and it's the whole point
                # of the "the bot should learn from its mistakes" loop -
                # answer this question once there and every future posting
                # that asks it gets answered automatically instead of
                # failing the same way again.
                try:
                    answer_gaps.record(
                        e.question, job_id=posting.job_id, company=posting.company, title=posting.title
                    )
                except CorruptDataFile as gap_error:
                    print(f"Warning: unanswered question not recorded - {gap_error}")
            audit.log("apply_error", job_id=posting.job_id, error=str(e))
            failure_log.log(
                "apply_error",
                job_id=posting.job_id,
                title=posting.title,
                company=posting.company,
                url=posting.url,
                error=str(e),
            )
            print(f"Error applying to {posting.title} at {posting.company}: {e}")
            failed += 1
            if _is_ollama_unreachable(e):
                print("Ollama is unreachable - stopping the run instead of repeating this for every posting.")
                fatal_error = True
                break
            if _is_claude_misconfigured(e):
                print(f"Claude provider is misconfigured ({e}) - stopping the run instead of repeating this for every posting.")
                fatal_error = True
                break
            if page.is_closed():
                print("Browser window was closed - stopping the run.")
                break
            continue

        if submitted is None:
            continue  # user declined the confirmation prompt

        if submitted:
            # The browser has already clicked Submit for real at this
            # point - mark_applied() must run before anything that could
            # raise, so a real submission is never lost from the tracker
            # (which would risk a duplicate real application on a future
            # run). record_application()'s own cap check is defense in
            # depth against a second concurrent `job-bot run` process
            # racing this one; if it loses that race, stop cleanly
            # rather than crash mid-loop.
            tracker.mark_applied(posting.job_id)
            audit.log("applied", job_id=posting.job_id, company=posting.company)
            applied += 1
            print(f"Applied: {posting.title} at {posting.company}")
            try:
                rate_limiter.record_application()
            except DailyCapReached:
                print("Daily application cap reached (possibly by a concurrent run). Stopping.")
                break
        else:
            audit.log("dry_run_stopped", job_id=posting.job_id)
            print(f"[dry-run] Would apply to {posting.title} at {posting.company}")

    return applied, failed, fatal_error


_STATUS_VIEW_FIELDS = (
    "title",
    "company",
    "status",
    "match_score",
    "url",
    "first_seen_at",
    "applied_at",
)


def cmd_status(settings: Settings, args: argparse.Namespace) -> None:
    """Record an outcome the bot has no way to observe on its own -
    `job-bot run` only ever writes seen/applied/skipped; everything past
    that (interviewing, offer, ...) is reported by the user by hand, or by
    `job-bot gmail-sync` reading a reply email. `--note` attaches a
    free-text note (see Tracker.set_note()) - independent of a status
    change, so either or both can be given in one call.

    With no <status> and no --note given, prints the job's current tracked
    record, the LLM's own match reasoning and eligibility-gate verdict (if
    it was scored, not just upserted - see Tracker.record_score()), note
    (if any), and answered-question history instead of changing anything -
    a quick, CLI-only way to check one application (e.g. before deciding what
    status to set, or to see *why* a job was skipped) without opening the
    dashboard. `--format json` prints that same view as one JSON object
    instead - every other read-oriented command here (doctor/report/
    export/faq/review-answers) already has this, and a monitoring script
    watching one specific application (e.g. to alert on a status change)
    otherwise has no structured way to read it short of `job-bot export`
    and searching the whole tracker for one job_id. Only applies to this
    view; --format is ignored when --status/--note change something,
    since those already print a one-line, trivially-parseable result.
    """
    tracker = Tracker(settings.db_path)

    if args.status is None and args.note is None:
        job = tracker.get_job(args.job_id)
        if job is None:
            if args.format == "json":
                print(json.dumps({"error": f"No tracked job with id {args.job_id!r}."}), file=sys.stderr)
            else:
                print(f"No tracked job with id {args.job_id!r}.", file=sys.stderr)
            sys.exit(1)
        generation = tracker.get_resume_generation(args.job_id)
        qa = tracker.list_qa(args.job_id)
        # Stored as a JSON array (see Tracker.record_score()'s docstring for
        # why, unlike the plain-text reasoning/eligibility_note columns) -
        # empty/None for a job scored before this column existed, or never
        # scored at all.
        missing_quals = json.loads(job["missing_qualifications"]) if job.get("missing_qualifications") else []
        if args.format == "json":
            payload = {
                **job,
                "missing_qualifications": missing_quals,
                "is_blacklisted": CompanyBlacklist(settings.blacklist_path).is_blocked(job["company"]),
                "resume_generation": generation,
                "qa_history": qa,
            }
            print(json.dumps(payload, indent=2))
            return
        width = max(len(field) for field in _STATUS_VIEW_FIELDS)
        for field in _STATUS_VIEW_FIELDS:
            print(f"{field:<{width}}  {job.get(field)}")
        if CompanyBlacklist(settings.blacklist_path).is_blocked(job["company"]):
            # A job tracked/applied to before its company was blacklisted
            # (or blacklisted afterward for an unrelated reason) - worth
            # surfacing here since nothing else flags the inconsistency:
            # job-bot run's own blacklist check only ever runs at search
            # time, never retroactively against what's already tracked.
            print(f"\n[!!] {job['company']} is on your blacklist.")
        if job.get("eligibility") in ("fail", "flag"):
            # Distinguishes a categorical eligibility-gate rejection (e.g.
            # a citizenship requirement the resume doesn't meet) from a
            # plain low score - status=skipped alone can't tell the two
            # apart. "flag" specifically means the model itself was
            # uncertain (a silent/ambiguous rule), worth a second look by
            # the human rather than a quiet skip.
            note = f" - {job['eligibility_note']}" if job.get("eligibility_note") else ""
            print(f"\n[!!] Eligibility: {job['eligibility']}{note}")
        if job.get("match_reasoning"):
            print(f"\nMatch reasoning: {job['match_reasoning']}")
        if missing_quals:
            print(f"\nMissing qualifications: {', '.join(missing_quals)}")
        if job.get("notes"):
            print(f"\nNote: {job['notes']}")
        if generation:
            print(f"\nTailored resume generated {generation['created_at']}:")
            print(f"  Summary: {generation['summary']}")
            print(f"  Skills:  {', '.join(generation['skills'])}")
        if qa:
            print(f"\nQ&A history ({len(qa)}):")
            for pair in qa:
                print(f"  Q: {pair['question']}\n  A: {pair['answer']}\n")
        return

    if args.note is not None:
        try:
            tracker.set_note(args.job_id, args.note)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Note set for {args.job_id}.")

    if args.status is not None:
        try:
            tracker.update_status(args.job_id, args.status)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"{args.job_id} -> {args.status}")


def cmd_review_answers(settings: Settings, args: argparse.Namespace) -> None:
    """Interactively answer the required questions Easy Apply couldn't
    confidently answer on its own (see safety/answer_gaps.py and
    browser/linkedin_adapter.py's UnansweredRequiredQuestion) - this is
    what "the bot should learn from its mistakes" looks like for a local
    model whose weights this project never fine-tunes: an answer given
    here is saved to FAQ_PATH and reused as context for every future
    posting that asks the same or a near-identical question (see
    generation/qa_answerer.py), so the same question doesn't keep failing
    the same way. Sorted most-frequently-seen first, since those are the
    ones worth the most to answer.

    `--search` narrows to gaps whose question contains this text (case-
    insensitive) - the same semantics `job-bot faq list --search` and
    `job-bot qa-history --search` already use - useful once there are
    enough recorded gaps that reviewing them all in frequency order isn't
    the fastest way to find a specific one (e.g. every sponsorship-related
    question, to answer them as one batch). `--format json` dumps the same
    (optionally --search-filtered) gaps (question/count/example) as one
    JSON array instead of prompting - for a monitoring script that wants
    to alert on e.g. a growing unanswered-questions count without an
    interactive terminal to answer from (input() would just hit EOF and
    stop immediately anyway, same as a piped/cron invocation of the
    interactive mode already does).

    `--dismiss` permanently discards one or more gaps without answering
    them. Real gap this closes: leaving the interactive prompt blank only
    ever skips a gap *for this run* - AnswerGapStore.resolve() (which
    actually removes one) was previously only ever called after saving a
    real answer, so a gap that's noise (a garbled/duplicate question, or
    one not worth caching an FAQ answer for) had no way to stop
    resurfacing every single time this command runs, forever, short of
    typing something into FAQ_PATH just to make it go away. Mirrors
    `job-bot blacklist remove`/`job-bot faq remove`'s own nargs="+" shape
    and per-item "removed/not found" feedback.
    """
    answer_gaps = AnswerGapStore(settings.answer_gaps_path)
    if args.dismiss:
        for question in args.dismiss:
            dismissed = answer_gaps.resolve(question)
            print(
                f'Dismissed: "{question}"' if dismissed else f'No unanswered gap matching: "{question}"'
            )
        return
    gaps = answer_gaps.list_unanswered()
    if args.search:
        needle = args.search.casefold()
        gaps = {question: info for question, info in gaps.items() if needle in question.casefold()}

    if args.format == "json":
        ordered = sorted(gaps.items(), key=lambda item: item[1].get("count", 0), reverse=True)
        payload = [{"question": question, **info} for question, info in ordered]
        print(json.dumps(payload, indent=2))
        return

    resume_store = ResumeStore(settings.resume_path, settings.faq_path)
    if not gaps:
        print(
            "No unanswered required questions recorded - nothing to review."
            if not args.search
            else f'No unanswered required questions matching "{args.search}".'
        )
        return

    ordered = sorted(gaps.items(), key=lambda item: item[1].get("count", 0), reverse=True)
    print(f"{len(ordered)} unanswered required question(s):\n")
    answered = 0
    for question, info in ordered:
        count = info.get("count", 1)
        times = "time" if count == 1 else "times"
        print(f'"{question}"')
        print(f"  seen {count} {times}, e.g. {info.get('example_title', '')} at {info.get('example_company', '')}")
        if is_per_position_field(question):
            # Offering to cache one answer here would put that date on every
            # position of every future form - see is_per_position_field().
            print(
                "  Not answerable here: this is a per-position work-history date, so one saved answer "
                "would be used for every position on every form. Dismiss it with "
                f'`job-bot review-answers --dismiss "{question}"`.\n'
            )
            continue
        try:
            answer = input("  Answer (blank to skip, Ctrl+C to stop): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if answer:
            resume_store.save_faq_answer(question, answer)
            answer_gaps.resolve(question)
            answered += 1
            print("  Saved - every future posting that asks this will be answered automatically.\n")
        else:
            print("  Skipped - still there next time you run this.\n")

    remaining = len(answer_gaps.list_unanswered())
    print(f"Answered {answered} question(s). {remaining} still unanswered.")


def cmd_faq(settings: Settings, args: argparse.Namespace) -> None:
    """view/remove/import/export cached FAQ answers - see resume/store.py's
    save_faq_answer()/faq_answers(). `job-bot review-answers` is the only
    way to *add* one one-at-a-time (it's tied to reviewing an actual
    unanswered question), but neither it nor anything else could view
    what's already cached, remove a wrong one, or back up/share a whole
    set short of hand-editing FAQ_PATH's JSON directly - this gives the
    FAQ cache the same view/remove/import/export shape `job-bot blacklist`
    already has, including `remove` taking one or more question texts
    (nargs="+", the same as `job-bot blacklist remove`'s companies) so
    cleaning up several wrong/stale answers spotted in one `faq list`
    review doesn't need a separate invocation per question. export's
    output is exactly what import reads back, so the two round-trip
    (e.g. to move a cache to another install).
    """
    resume_store = ResumeStore(settings.resume_path, settings.faq_path)
    if args.faq_action == "list":
        answers = resume_store.faq_answers()
        if args.search:
            # Case-insensitive substring match against either side of the
            # pair - the same "search" semantics job-bot export/the
            # dashboard already use, not just the question text, since a
            # cache large enough to need searching is exactly one where
            # you might remember the gist of an answer but not the exact
            # question wording that produced it.
            needle = args.search.casefold()
            answers = {
                question: answer
                for question, answer in answers.items()
                if needle in question.casefold() or needle in answer.casefold()
            }
        if args.format == "json":
            print(json.dumps(answers, indent=2, ensure_ascii=False))
        elif not answers:
            print("No cached FAQ answers." if not args.search else f'No cached FAQ answers matching "{args.search}".')
        else:
            for question, answer in answers.items():
                print(f'"{question}"\n  -> {answer}\n')
    elif args.faq_action == "remove":
        for question in args.question:
            removed = resume_store.remove_faq_answer(question)
            print(
                f'Removed cached answer for: "{question}"'
                if removed
                else f'No cached answer for: "{question}"'
            )
    elif args.faq_action == "clean":
        # Runtime already skips these (see resume/store.py's unusable_faq_reason()), so this is
        # housekeeping rather than a fix - but a skipped entry still shows
        # in `faq list`/the dashboard as if it were a real answer, and
        # `faq export` would carry it to another install.
        unusable = {
            question: reason
            for question, answer in resume_store.faq_answers().items()
            if (reason := unusable_faq_reason(question, answer)) is not None
        }
        if not unusable:
            print("No unusable cached FAQ answers found.")
            return
        verb = "Would remove" if args.dry_run else "Removed"
        for question, reason in unusable.items():
            if not args.dry_run:
                resume_store.remove_faq_answer(question)
            print(f'{verb} "{question}" ({reason})')
        if args.dry_run:
            print(f"{len(unusable)} unusable answer(s) found - re-run without --dry-run to remove them.")
    elif args.faq_action == "import":
        try:
            data = json.loads(args.file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as e:
            # UnicodeDecodeError alongside OSError - a --file saved with a
            # non-UTF-8 encoding previously crashed this with a raw,
            # uncaught traceback instead of the same clean "could not
            # read" message an unreadable/missing file already gets here.
            print(f"Error: could not read {args.file}: {e}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: {args.file} is not valid JSON: {e}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(data, dict):
            print(
                f'Error: {args.file} must contain a JSON object of {{"question": "answer"}} pairs, '
                "the same shape FAQ_PATH itself is stored in.",
                file=sys.stderr,
            )
            sys.exit(1)
        for question, answer in data.items():
            resume_store.save_faq_answer(str(question), str(answer))
        print(f"Imported {len(data)} FAQ answer(s) from {args.file}.")
    elif args.faq_action == "export":
        answers = resume_store.faq_answers()
        text = json.dumps(answers, indent=2, ensure_ascii=False) + "\n"
        if args.out is not None:
            args.out.write_text(text, encoding="utf-8")
            print(f"Exported {len(answers)} FAQ answer(s) to {args.out}.")
        else:
            sys.stdout.write(text)


def _no_history_message(
    kind: str, search: str | None, exact: str | None, *, exact_label: str = "company"
) -> str:
    """Shared "nothing found" wording for cmd_qa_history/cmd_resume_history/
    cmd_audit_log - `kind` is e.g. "Q&A history"/"resume generations"/
    "audit log entries". Preserves the exact "No {kind} matching
    \"{search}\"." wording qa-history/resume-history already had before
    --company existed (search-only is the common case, and changing
    established message text would be a needless breaking change for a
    script grepping it), while still describing the second, exact-match
    filter too when it's given, combined or alone. `exact_label` names
    that filter in the message ("company" by default, matching
    qa-history/resume-history's own --company; cmd_audit_log passes
    "action" for its own --action).
    """
    if not search and not exact:
        return f"No {kind} recorded yet."
    parts = []
    if search:
        parts.append(f'"{search}"')
    if exact:
        parts.append(f'{exact_label} "{exact}"')
    return f"No {kind} matching {' and '.join(parts)}."


def cmd_qa_history(settings: Settings, args: argparse.Namespace) -> None:
    """Every question/answer pair ever recorded, across every job, most
    recent first - unlike `job-bot status <job_id>` (one job at a time) or
    `job-bot faq list` (only the curated, promoted-to-cache subset), this
    is the full raw transcript (Tracker.search_qa()), with each pair's
    company/title for context. `--search` matches question or answer text,
    the same semantics `faq list --search` already uses. `--company` is an
    exact match (via normalize_company_name, the same rule `job-bot export
    --company` uses) rather than --search's fuzzy substring - pulling every
    pair for one company via --search risks also matching unrelated pairs
    whose question/answer text happens to mention that company in passing.
    """
    tracker = Tracker(settings.db_path)
    pairs = tracker.search_qa(search=args.search)
    if args.company is not None:
        normalized = normalize_company_name(args.company)
        pairs = [p for p in pairs if normalize_company_name(p["company"] or "") == normalized]
    if args.format == "json":
        print(json.dumps(pairs, indent=2))
        return
    if not pairs:
        print(_no_history_message("Q&A history", args.search, args.company))
        return
    for pair in pairs:
        print(f"[{pair['job_id']}] {pair['company']} - {pair['title']} ({pair['created_at']})")
        print(f"  Q: {pair['question']}\n  A: {pair['answer']}\n")


def cmd_resume_history(settings: Settings, args: argparse.Namespace) -> None:
    """Every tailor_resume() output ever recorded, across every job, most
    recent first, with each job's current status for outcome context - the
    resume-tailoring counterpart to `job-bot qa-history`. Unlike `job-bot
    status <job_id>` (one job's own generation only) or the internal,
    ranked-and-capped best_resume_examples() (few-shot prompt use only),
    this is the full raw history, for a human to review which tailoring
    approaches actually led somewhere. `--search` matches summary, company,
    or title text, the same semantics `faq list --search`/`job-bot
    qa-history --search` already use. `--company` is an exact match (same
    rule `job-bot export --company`/`job-bot qa-history --company` use)
    rather than --search's fuzzy substring.
    """
    tracker = Tracker(settings.db_path)
    generations = tracker.list_resume_generations(search=args.search)
    if args.company is not None:
        normalized = normalize_company_name(args.company)
        generations = [g for g in generations if normalize_company_name(g["company"] or "") == normalized]
    if args.format == "json":
        print(json.dumps(generations, indent=2))
        return
    if not generations:
        print(_no_history_message("resume generations", args.search, args.company))
        return
    for gen in generations:
        status = gen["status"] or "not tracked"
        print(f"[{gen['job_id']}] {gen['company']} - {gen['title']} ({status}, {gen['created_at']})")
        print(f"  Summary: {gen['summary']}")
        print(f"  Skills:  {', '.join(gen['skills'])}\n")


def cmd_audit_log(settings: Settings, args: argparse.Namespace) -> None:
    """Every logged action from the safety-critical audit trail
    (safety/audit_log.py's own module docstring: "Append-only, secret-
    redacted log of every action the bot takes"), most recent first -
    `job-bot doctor`'s "Audit log writable" check only ever proves this
    file can be written to; until this command existed there was no way
    to read it back short of grepping the raw JSONL file by hand.
    `--search` matches the whole entry (action and every detail value),
    the same semantics `qa-history`/`resume-history --search` already use.
    `--action` is the exact-match equivalent of those two commands' own
    `--company` - e.g. `--action applied`, `--action skip_blacklisted` (see
    AuditLogger.read_entries()'s own docstring for where to find the full
    action vocabulary).

    `--failures` points this at FAILED_APPLICATIONS_LOG_PATH instead of
    AUDIT_LOG_PATH - the very file `_print_cycle_summary()` below tells the
    user to go read by hand after a `job-bot run` that couldn't finish some
    postings (confirmed live: this exact message is what a real user saw).
    It's the same AuditLogger-backed JSONL shape (search_error/prep_error/
    apply_error entries - see cmd_run's own failure_log.log(...) call
    sites), so every other flag here works identically against it; the
    only difference is which file gets opened.
    """
    path = settings.failed_applications_log_path if args.failures else settings.audit_log_path
    audit = AuditLogger(path)
    entries = audit.read_entries(search=args.search, action=args.action)
    kind = "failed-application entries" if args.failures else "audit log entries"
    if args.format == "json":
        print(json.dumps(entries, indent=2))
        return
    if not entries:
        print(_no_history_message(kind, args.search, args.action, exact_label="action"))
        return
    for entry in entries:
        details = ", ".join(f"{k}={v}" for k, v in entry.get("details", {}).items())
        line = f"[{entry.get('timestamp', '?')}] {entry.get('action', '?')}"
        print(f"{line}  {details}" if details else line)


# Score buckets for `job-bot report --by-score`'s outcome breakdown, widest
# (worst-fit) first so a job with a null match_score never fits any bucket
# and is simply left out - it hasn't been through the LLM scorer yet.
SCORE_BUCKETS = ((0, 59), (60, 69), (70, 79), (80, 89), (90, 100))


def _score_bucket_label(score: int) -> str:
    for lo, hi in SCORE_BUCKETS:
        if lo <= score <= hi:
            return f"{lo}-{hi}"
    return "?"


def _score_breakdown(tracker: Tracker) -> dict[str, dict[str, int]]:
    """bucket label -> status -> count, for `job-bot report --by-score` -
    shared by both the text table (_print_score_breakdown below) and JSON
    output (cmd_report's --format json) so the two can never compute this
    differently.
    """
    scored_jobs = [job for job in tracker.list_jobs() if job["match_score"] is not None]
    buckets: dict[str, dict[str, int]] = {}
    for job in scored_jobs:
        by_status = buckets.setdefault(_score_bucket_label(job["match_score"]), {})
        by_status[job["status"]] = by_status.get(job["status"], 0) + 1
    return buckets


def _print_score_breakdown(tracker: Tracker) -> None:
    buckets = _score_breakdown(tracker)
    if not buckets:
        return

    statuses = sorted({status for counts in buckets.values() for status in counts})
    print("\nOutcomes by match score:")
    header = "score".ljust(8) + "".join(status.ljust(14) for status in statuses) + "total"
    print(header)
    for lo, hi in SCORE_BUCKETS:
        bucket_counts = buckets.get(f"{lo}-{hi}")
        if not bucket_counts:
            continue
        row = f"{lo}-{hi}".ljust(8) + "".join(str(bucket_counts.get(s, 0)).ljust(14) for s in statuses)
        print(row + str(sum(bucket_counts.values())))


# Printed first, in this fixed order rather than sorted alphabetically, so
# the common/expected case (pass) always reads first and the two verdicts
# most worth a second look (flag, fail) come right after it - "not scored"
# (Tracker.upsert_job() jobs, or ones tracked before this column existed;
# see Tracker.record_score()) last since it's not an eligibility verdict at
# all. Any other value found - there shouldn't be one, but eligibility is
# free-text on disk, not a DB-enforced enum - is appended after these four,
# so a future bug or a hand-edited row is still visible rather than lost.
_ELIGIBILITY_VERDICT_ORDER = ("pass", "flag", "fail", "not scored")


def _eligibility_breakdown(tracker: Tracker) -> dict[str, int]:
    """eligibility verdict -> count, for `job-bot report --by-eligibility` -
    how many tracked jobs were let through, flagged as ambiguous, or
    categorically disqualified by the eligibility gate (matching/scorer.py),
    versus never having been scored at all. Distinct from --by-score: a
    fail/flag verdict and a plain low score both end up status=skipped, so
    the score breakdown alone can't tell a citizenship-requirement
    rejection apart from a job that was simply a weak fit.
    """
    counts: dict[str, int] = {}
    for job in tracker.list_jobs():
        verdict = job.get("eligibility") or "not scored"
        counts[verdict] = counts.get(verdict, 0) + 1
    return counts


def _print_eligibility_breakdown(tracker: Tracker) -> None:
    """No `if not breakdown: return` guard here, unlike its sibling
    _print_score_breakdown: that one's buckets can genuinely be empty (every
    tracked job never scored, so none have a match_score to bucket), but
    _eligibility_breakdown always assigns a job to "not scored" instead of
    omitting it - it's only ever empty when there are zero tracked jobs at
    all, a case cmd_report's own `if not counts: return` already handles
    before this is ever called. A guard for a case that can't happen here
    would be untestable dead code, not real defense.
    """
    breakdown = _eligibility_breakdown(tracker)
    print("\nOutcomes by eligibility verdict:")
    width = max(len(verdict) for verdict in breakdown)
    ordered = [v for v in _ELIGIBILITY_VERDICT_ORDER if v in breakdown]
    ordered += sorted(v for v in breakdown if v not in _ELIGIBILITY_VERDICT_ORDER)
    for verdict in ordered:
        print(f"{verdict:<{width}}  {breakdown[verdict]}")


def _company_breakdown(tracker: Tracker) -> dict[str, dict[str, int]]:
    """company -> status -> count, for `job-bot report --by-company` - which
    companies you've actually applied to the most, and how those
    applications are trending (interviewing/offer vs rejected/no_response),
    without dropping to `job-bot export --search "<company>"` one company at
    a time. Unlike _score_breakdown, every tracked job has a company, so
    this is never partial the way the score breakdown is for unscored jobs.
    """
    buckets: dict[str, dict[str, int]] = {}
    for job in tracker.list_jobs():
        by_status = buckets.setdefault(job["company"], {})
        by_status[job["status"]] = by_status.get(job["status"], 0) + 1
    return buckets


def _print_company_breakdown(tracker: Tracker) -> None:
    """No `if not buckets: return` guard here, the same reasoning
    _print_eligibility_breakdown gives for lacking one: every tracked job
    has a non-null `company` (the jobs table's own NOT NULL constraint),
    so _company_breakdown can only be empty when there are zero tracked
    jobs at all - a case cmd_report's own `if not counts: return` already
    handles before this is ever called. A guard for a case that can't
    happen here would be untestable dead code, not real defense.
    """
    buckets = _company_breakdown(tracker)
    statuses = sorted({status for counts in buckets.values() for status in counts})
    # Most-applied company first, not alphabetical - that's the actually
    # useful read ("where have I put in the most effort"), with an
    # alphabetical tiebreak so equal-count companies still print in a
    # stable, predictable order.
    ordered = sorted(buckets, key=lambda company: (-sum(buckets[company].values()), company.casefold()))
    width = max(len("company"), max(len(company) for company in buckets))
    print("\nOutcomes by company:")
    header = "company".ljust(width + 2) + "".join(status.ljust(14) for status in statuses) + "total"
    print(header)
    for company in ordered:
        counts = buckets[company]
        row = company.ljust(width + 2) + "".join(str(counts.get(s, 0)).ljust(14) for s in statuses)
        print(row + str(sum(counts.values())))


def _weekly_breakdown(tracker: Tracker) -> dict[str, dict[str, int]]:
    """Thin wrapper around Tracker.applications_by_week() (see its own
    docstring), matching _missing_qualifications_breakdown's shape - the
    logic lives on Tracker so the dashboard's Weekly Activity panel can
    reuse it without importing from this module.
    """
    return tracker.applications_by_week()


def _print_weekly_breakdown(tracker: Tracker) -> None:
    """Guarded by `if not buckets: return` - unlike _print_company_breakdown,
    jobs can be tracked (seen/skipped) with none ever applied to, so this
    can be empty even when cmd_report got past its own `if not counts`.
    """
    buckets = _weekly_breakdown(tracker)
    if not buckets:
        return
    statuses = sorted({status for counts in buckets.values() for status in counts})
    print("\nApplications by week:")
    header = "week of".ljust(12) + "".join(status.ljust(14) for status in statuses) + "total"
    print(header)
    # Most recent week first - "how am I doing lately" is the useful read.
    for week in sorted(buckets, reverse=True):
        counts = buckets[week]
        row = week.ljust(12) + "".join(str(counts.get(s, 0)).ljust(14) for s in statuses)
        print(row + str(sum(counts.values())))


def _missing_qualifications_breakdown(tracker: Tracker, *, limit: int | None = None) -> dict[str, int]:
    """Thin wrapper around Tracker.missing_qualifications_counts() (see its
    own docstring for the exact-string-counting/limit reasoning) - kept
    here, matching _score_breakdown/_eligibility_breakdown/_company_breakdown's
    own shape, so `job-bot report --by-missing-qualifications` and its
    `--missing-qualifications-limit` flag read the same as every other
    breakdown in this file. The counting logic itself moved to Tracker so
    the dashboard's Missing Qualifications panel (dashboard/server.py)
    could reuse it too, without importing from this module.
    """
    return tracker.missing_qualifications_counts(limit=limit)


def _print_missing_qualifications_breakdown(tracker: Tracker, *, limit: int | None = None) -> None:
    """Guarded by `if not breakdown: return`, unlike _print_company_breakdown
    - every tracked job has a company, but a job can easily have zero
    missing_qualifications (never scored, or the LLM found no gaps), so
    this can be genuinely empty even with jobs tracked, the same reason
    _print_score_breakdown guards too.
    """
    breakdown = _missing_qualifications_breakdown(tracker, limit=limit)
    if not breakdown:
        return
    # Most commonly missing first - that's the actionable read ("what to
    # fix on the resume first") - alphabetical tiebreak for a stable order.
    ordered = sorted(breakdown, key=lambda qual: (-breakdown[qual], qual.casefold()))
    print("\nMost common missing qualifications:")
    for qual in ordered:
        print(f"  {breakdown[qual]:>3}  {qual}")


def _stale_applications(tracker: Tracker, days: int) -> list[dict]:
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    return [
        job
        for job in tracker.list_jobs(status="applied", sort="applied_at", direction="asc")
        if job["applied_at"] and job["applied_at"] < cutoff
    ]


def cmd_report(settings: Settings, args: argparse.Namespace) -> None:
    """Print status counts, plus three optional sections: `--stale-days`
    (applications with no reply worth a manual follow-up), `--by-score`
    (how match score correlates with actual outcomes, to sanity-check
    whether the LLM scorer's judgment tracks reality), and
    `--by-eligibility` (how many jobs were let through, flagged, or
    categorically disqualified by the eligibility gate - see
    _eligibility_breakdown), `--by-company` (which companies you've
    applied to the most, and how those applications are trending), and
    `--by-missing-qualifications` (which specific gaps the LLM scorer keeps
    flagging across postings - see _missing_qualifications_breakdown;
    `--missing-qualifications-limit` caps it to the N most common), and
    `--by-week` (applications sent per week - see _weekly_breakdown).
    `--format json` prints the same data as one
    JSON object instead - for a script or cron job that wants to alert on
    e.g. a growing stale-applications count without scraping the
    human-readable text layout.
    """
    tracker = Tracker(settings.db_path)
    counts = tracker.status_counts()
    stale_days = args.stale_days if args.stale_days is not None else settings.stale_after_days
    stale = _stale_applications(tracker, stale_days)

    if args.format == "json":
        payload: dict[str, Any] = {
            "counts": counts,
            "total": sum(counts.values()),
            "stale_days": stale_days,
            "stale": [
                {
                    "job_id": job["job_id"],
                    "company": job["company"],
                    "title": job["title"],
                    "applied_at": job["applied_at"],
                }
                for job in stale
            ],
        }
        if args.by_score:
            payload["by_score"] = _score_breakdown(tracker)
        if args.by_eligibility:
            payload["by_eligibility"] = _eligibility_breakdown(tracker)
        if args.by_company:
            payload["by_company"] = _company_breakdown(tracker)
        if args.by_week:
            payload["by_week"] = _weekly_breakdown(tracker)
        if args.by_missing_qualifications:
            payload["by_missing_qualifications"] = _missing_qualifications_breakdown(
                tracker, limit=args.missing_qualifications_limit
            )
        print(json.dumps(payload, indent=2))
        return

    if not counts:
        print("No jobs tracked yet.")
        return
    width = max(len(status) for status in counts)
    for status in sorted(counts):
        print(f"{status:<{width}}  {counts[status]}")
    print(f"{'total':<{width}}  {sum(counts.values())}")

    if stale:
        print(f"\nApplied {stale_days}+ days ago with no reply ({len(stale)}):")
        for job in stale:
            print(f"  {job['job_id']}  {job['company']} - {job['title']}  (applied {job['applied_at'][:10]})")

    if args.by_score:
        _print_score_breakdown(tracker)

    if args.by_eligibility:
        _print_eligibility_breakdown(tracker)

    if args.by_company:
        _print_company_breakdown(tracker)

    if args.by_week:
        _print_weekly_breakdown(tracker)

    if args.by_missing_qualifications:
        _print_missing_qualifications_breakdown(tracker, limit=args.missing_qualifications_limit)


def cmd_export(settings: Settings, args: argparse.Namespace) -> None:
    """Dump tracked jobs as CSV or JSON (`--format`) - to a file with
    `--out`, or stdout so it pipes straight into another tool. `--search`
    matches the dashboard's own search box (Tracker.list_jobs' `search`, a
    title/company/notes/match_reasoning/eligibility_note/
    missing_qualifications substring match) - the dashboard's
    /api/export.csv/.json already respected it, but the CLI command had no
    equivalent way to export a search result
    instead of a full status-filtered dump. `--eligibility` is the exact
    counterpart to `--status`, for pulling e.g. every job the eligibility
    gate categorically disqualified without guessing a search term that
    happens to match all of them. `--company` is an exact match (via
    normalize_company_name, the same rule CompanyBlacklist/gmail_sync use)
    rather than `--search`'s fuzzy substring - pulling every row for one
    company via `--search "Acme"` risks also matching unrelated rows whose
    notes/match_reasoning happen to mention "Acme" in passing, or missing
    ones whose stored company name is cased/spaced differently, exactly
    the kind of substring false-positive fixed in find_matching_job().
    `--stale-days` is the row-level counterpart to `job-bot report
    --stale-days`'s own list section (same "status=applied, no reply
    after N days" rule as _stale_applications() below) - the aggregate
    count there is useful for a glance, but a script that wants to
    actually act on the stale set (send follow-up reminders, say) needs
    the full exported rows, not just a count.
    """
    tracker = Tracker(settings.db_path)
    jobs = tracker.list_jobs(
        status=args.status,
        search=args.search,
        eligibility=args.eligibility,
        sort="first_seen_at",
        direction="asc",
    )
    if args.company is not None:
        normalized = normalize_company_name(args.company)
        jobs = [job for job in jobs if normalize_company_name(job["company"]) == normalized]
    if args.stale_days is not None:
        cutoff = (datetime.now(UTC) - timedelta(days=args.stale_days)).isoformat()
        jobs = [
            job for job in jobs if job["status"] == "applied" and job["applied_at"] and job["applied_at"] < cutoff
        ]
    write = write_export_json if args.format == "json" else write_export_csv

    if args.out:
        # newline="" so csv's own \r\n line terminator isn't doubled up by
        # universal-newline text-mode translation on write; harmless for json.
        with args.out.open("w", newline="", encoding="utf-8") as f:
            write(f, jobs)
        print(f"Exported {len(jobs)} job(s) to {args.out}")
    else:
        write(sys.stdout, jobs)


def cmd_gmail_sync(settings: Settings, args: argparse.Namespace) -> None:
    """Read recent Gmail, classify each message, and advance the matching
    tracked job's status - see gmail_sync.py's module docstring for the
    exact never-guess/never-downgrade rules this delegates to.

    `--format json` prints the same GmailSyncResult as one JSON object
    instead - every other command that hands back a structured result
    (report/export/doctor/faq/review-answers/status) already has this; a
    monitoring script running this on a schedule (e.g. `--dry-run --format
    json` to alert on what *would* change without applying it, or without
    --dry-run to log what actually did) otherwise has to scrape the
    human-formatted text.
    """
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

    if args.format == "json":
        payload = {
            "dry_run": args.dry_run,
            "total_emails": result.total_emails,
            "updated": [
                {"job_id": job_id, "company": company, "new_status": new_status}
                for job_id, company, new_status in result.updated
            ],
            "skipped_low_confidence": result.skipped_low_confidence,
            "unmatched_subjects": result.unmatched_subjects,
            "classification_errors": result.classification_errors,
        }
        print(json.dumps(payload, indent=2))
        return

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
    if result.classification_errors:
        print(
            f"{result.classification_errors} email(s) could not be classified (LLM provider error) - "
            "see `job-bot audit-log --action gmail_sync_classify_error` for what happened and why."
        )


def cmd_blacklist(settings: Settings, args: argparse.Namespace) -> None:
    """add/remove/check/list/import/export companies `job-bot run` will
    always skip - see build_parser()'s `blacklist` subparser for the six
    actions. add/remove each take one or more company names (nargs="+"),
    so blacklisting several past employers at once doesn't need a
    separate invocation per company. `add --reason` applies the same
    reason to every company in that call - blacklisting several companies
    for the same reason in one call is the common case; a different reason
    per company just means a separate `add` call each. `check` is the
    single-company lookup `list`'s own output otherwise has no shortcut
    for once the blacklist has more than a handful of entries - "is X
    blocked, and why" without scanning the whole list by eye or grepping
    `--format json`. export's output is exactly what import reads back
    (one company per line, no reason - see CompanyBlacklist's own
    docstring for why reasons don't round-trip through that plain-text
    format), so the two round-trip (e.g. to move a blacklist to another
    install).
    """
    blacklist = CompanyBlacklist(settings.blacklist_path)
    if args.blacklist_action == "add":
        tracker = Tracker(settings.db_path)
        for company in args.company:
            blacklist.add(company, reason=args.reason or "")
            print(f"Added to blacklist: {company}")
            # Blacklisting only stops future applications - it doesn't
            # touch anything already tracked - so this is purely a heads-up
            # in case the name typed here wasn't meant to catch an
            # application you're still actively in.
            in_progress = tracker.in_progress_jobs_at_company(company)
            if in_progress:
                statuses = ", ".join(sorted({job["status"] for job in in_progress}))
                print(
                    f"  Note: {len(in_progress)} tracked application(s) at {company} are still "
                    f"in progress ({statuses}) - blacklisting only stops future applications, "
                    "these aren't affected."
                )
    elif args.blacklist_action == "remove":
        for company in args.company:
            removed = blacklist.remove(company)
            print(
                f"Removed from blacklist: {company}" if removed else f"Not on the blacklist: {company}"
            )
    elif args.blacklist_action == "check":
        entry = blacklist.get_entry(args.company)
        if args.format == "json":
            print(
                json.dumps(
                    {
                        "company": entry["name"] if entry else args.company,
                        "blocked": entry is not None,
                        "reason": entry["reason"] if entry else None,
                    }
                )
            )
        elif entry is None:
            print(f"{args.company} is not blacklisted.")
        else:
            line = f"{entry['name']} is blacklisted"
            if entry["reason"]:
                line += f" - {entry['reason']}"
            print(f"{line}.")
    elif args.blacklist_action == "list":
        entries = blacklist.list_entries()
        if args.search:
            # Same case-insensitive substring-of-either-field semantics
            # `job-bot faq list --search` already uses (question or
            # answer) - here, a company name or its reason, since once
            # the list is long enough to need searching, remembering
            # *why* a company was blacklisted is often easier than its
            # exact name. `job-bot blacklist check` is the exact-match
            # counterpart for when the name itself is already known.
            needle = args.search.casefold()
            entries = [
                e for e in entries if needle in e["name"].casefold() or needle in e["reason"].casefold()
            ]
        if args.format == "json":
            print(json.dumps(entries, indent=2, ensure_ascii=False))
        elif not entries:
            print(
                "Blacklist is empty."
                if not args.search
                else f'No blacklisted companies matching "{args.search}".'
            )
        else:
            for entry in entries:
                line = entry["name"]
                if entry["reason"]:
                    line += f"  - {entry['reason']}"
                print(line)
    elif args.blacklist_action == "import":
        try:
            lines = args.file.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as e:
            # Same reasoning as `faq import`'s own read above - a --file
            # saved with a non-UTF-8 encoding previously crashed this with
            # a raw, uncaught traceback instead of this same clean
            # "could not read" message.
            print(f"Error: could not read {args.file}: {e}", file=sys.stderr)
            sys.exit(1)
        # One company per line - blank lines and "#"-prefixed comment lines
        # skipped, the same lightweight convention as a requirements.txt or
        # .gitignore, so a list exported from a spreadsheet (or hand-edited)
        # doesn't need any real structure beyond that.
        companies = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
        for company in companies:
            blacklist.add(company)
        print(f"Imported {len(companies)} compan{'y' if len(companies) == 1 else 'ies'} from {args.file}.")
    elif args.blacklist_action == "export":
        companies = blacklist.list_companies()
        text = "".join(f"{company}\n" for company in companies)
        if args.out is not None:
            args.out.write_text(text, encoding="utf-8")
            print(
                f"Exported {len(companies)} compan{'y' if len(companies) == 1 else 'ies'} to {args.out}."
            )
        else:
            sys.stdout.write(text)


def cmd_dashboard(settings: Settings, args: argparse.Namespace) -> None:
    """Serve the local tracker dashboard - see dashboard/server.py's module
    docstring for why it binds to localhost only and has no login.
    """
    port = args.port if args.port is not None else settings.dashboard_port
    run_dashboard(
        settings.db_path,
        settings.blacklist_path,
        settings.audit_log_path,
        settings.failed_applications_log_path,
        settings.answer_gaps_path,
        settings.resume_path,
        settings.faq_path,
        port=port,
        open_browser=not args.no_open,
        stale_after_days=settings.stale_after_days,
    )


def cmd_test_provider(settings: Settings, args: argparse.Namespace) -> None:
    """One real API call to the configured LLM provider, to confirm the key/
    model/local server actually work before trusting them to score real
    postings. See also `job-bot doctor`, which checks everything else
    (resume, LinkedIn session, ...) without making a network call.
    `--format json` prints the result as one JSON object instead - every
    other check-style command here (doctor, status, report, ...) already
    has this, for a monitoring script that wants to verify the LLM
    provider actually responds (not just that config looks sane, which
    `job-bot doctor --format json` alone can't confirm) without parsing
    a Pydantic model's default repr.
    """
    provider = get_provider(settings)
    result = provider.generate_structured(
        system="You are a test.",
        prompt=(
            "Respond as if evaluating a strong resume match: score 90, "
            "should_apply true, no missing qualifications, brief reasoning."
        ),
        schema=JobMatchScore,
    )
    if args.format == "json":
        print(json.dumps({"provider": settings.llm_provider, "result": result.model_dump()}, indent=2))
        return
    print(f"Provider OK: {settings.llm_provider}")
    print(result)


def _resume_check(settings: Settings) -> tuple[str, bool, str]:
    """Actually parses the resume, not just checks the file exists - a
    present-but-corrupted PDF, an unsupported extension, or an empty
    resume.txt (see resume/parser.py's ResumeParseError cases) previously
    only surfaced once `job-bot run` was already underway, well past
    `job-bot doctor` giving it a clean bill of health.
    """
    if not settings.resume_path.exists():
        return ("Resume file readable", False, str(settings.resume_path))
    try:
        parse_resume(settings.resume_path)
    except ResumeParseError as e:
        return ("Resume file readable", False, str(e))
    return ("Resume file readable", True, str(settings.resume_path))


def _probe_writable(directory: Path) -> str | None:
    """Actually creates `directory` and writes/removes a small probe file
    to prove it's writable, not just present - returns None if writable,
    or the OSError's message if not. Shared by every doctor check that
    needs this: _applications_dir_check/_blacklist_check/_faq_check/
    _answer_gaps_check below, for the case their own file doesn't exist
    yet (a fresh install, the normal case for all four) - a bare
    .exists() check, or the JSON-validity check _blacklist_check/
    _faq_check/_answer_gaps_check already run when the file IS present,
    says nothing about whether that file's directory could actually be
    written to the first time `job-bot run`/`blacklist add`/etc. needs
    to create it - the same "verify, not just check existence" reasoning
    _resume_check established first, for parsing rather than writing.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".job_bot_doctor_probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as e:
        return str(e)
    return None


def _blacklist_check(settings: Settings) -> tuple[str, bool, str]:
    """Neither missing (a fresh install has no blacklist file yet, which is
    fine) nor present-but-corrupted should look the same as "no companies
    blacklisted" - but CompanyBlacklist._load() (safety/blacklist.py)
    silently falls back to an empty blacklist on invalid JSON rather than
    raising. `job-bot run` now refuses to start on such a file and
    CompanyBlacklist._save() refuses to overwrite it (data_files.py), so
    neither blocking nor the file's contents are silently lost any more -
    this check reports the problem up front, before either of those, the
    same way _resume_check catches a corrupted resume before `job-bot run`
    gets there.

    When the file doesn't exist yet (the normal fresh-install case),
    proves the directory it would be created in is actually writable
    instead - see _probe_writable()'s own docstring for why a bare
    .exists() check alone isn't enough here either.
    """
    label = "Blacklist file valid (optional)"
    if not settings.blacklist_path.exists():
        error = _probe_writable(settings.blacklist_path.parent)
        if error is not None:
            return (label, False, f"{settings.blacklist_path.parent}: {error}")
        return (label, True, "")
    try:
        data = json.loads(settings.blacklist_path.read_text(encoding="utf-8"))
    except OSError as e:
        return (label, False, f"{settings.blacklist_path}: {e}")
    except UnicodeDecodeError as e:
        return (label, False, f"{settings.blacklist_path}: not valid UTF-8 ({e})")
    except json.JSONDecodeError as e:
        return (label, False, f"{settings.blacklist_path}: not valid JSON ({e})")
    if not isinstance(data, list):
        return (label, False, f"{settings.blacklist_path}: must be a JSON array of company name strings")
    return (label, True, "")


def _faq_check(settings: Settings) -> tuple[str, bool, str]:
    """Same corruption-hides-as-empty risk _blacklist_check guards against,
    for FAQ_PATH - ResumeStore.faq_answers() (resume/store.py) also falls
    back to {} on invalid JSON. Lower-stakes than the blacklist (a lost
    cache just means re-asking the LLM, not a lost safety guarantee), but
    still worth a heads-up before `job-bot review-answers`/`job-bot run`
    silently start treating every previously-learned answer as unknown.

    When the file doesn't exist yet, proves the directory it would be
    created in is actually writable instead - see _probe_writable()'s
    own docstring.
    """
    label = "FAQ cache valid (optional)"
    if not settings.faq_path.exists():
        error = _probe_writable(settings.faq_path.parent)
        if error is not None:
            return (label, False, f"{settings.faq_path.parent}: {error}")
        return (label, True, "")
    try:
        data = json.loads(settings.faq_path.read_text(encoding="utf-8"))
    except OSError as e:
        return (label, False, f"{settings.faq_path}: {e}")
    except UnicodeDecodeError as e:
        return (label, False, f"{settings.faq_path}: not valid UTF-8 ({e})")
    except json.JSONDecodeError as e:
        return (label, False, f"{settings.faq_path}: not valid JSON ({e})")
    if not isinstance(data, dict):
        return (label, False, f'{settings.faq_path}: must be a JSON object of {{"question": "answer"}} pairs')
    return (label, True, "")


def _faq_usability_check(settings: Settings) -> tuple[str, bool, str]:
    """Cached FAQ answers every run skips (unusable_faq_reason() - an
    echoed question or leaked reasoning). Harmless to a run, which already
    skips them, but otherwise only visible via `job-bot faq clean` or the
    dashboard's FAQ panel - doctor is the one place a user checks for "is
    anything off?". A corrupt or unreadable file passes here: _faq_check
    above already reports it, and flagging it twice would be noise.
    """
    label = "FAQ cache has no unusable answers (optional)"
    answers = ResumeStore(settings.resume_path, settings.faq_path).faq_answers()
    unusable = [question for question, answer in answers.items() if unusable_faq_reason(question, answer)]
    if not unusable:
        return (label, True, "")
    examples = ", ".join(f'"{question}"' for question in unusable[:3])
    more = f" and {len(unusable) - 3} more" if len(unusable) > 3 else ""
    return (label, False, f"{len(unusable)} skipped by every run ({examples}{more}) - run `job-bot faq clean`")


def _answer_gaps_check(settings: Settings) -> tuple[str, bool, str]:
    """Same corruption-hides-as-empty risk _blacklist_check/_faq_check
    guard against, for ANSWER_GAPS_PATH - AnswerGapStore._load()
    (safety/answer_gaps.py) also falls back to {} on invalid JSON. Not
    just a lost-cache inconvenience like FAQ_PATH: AnswerGapStore.record()
    loads, mutates, then saves the whole file on every single unanswered
    question `job-bot run` hits - the very next occurrence after
    corruption would silently overwrite the file with just that one new
    entry, permanently losing every previously recorded gap `job-bot
    review-answers` had queued up to review.

    When the file doesn't exist yet, proves the directory it would be
    created in is actually writable instead - see _probe_writable()'s
    own docstring.
    """
    label = "Answer-gaps file valid (optional)"
    if not settings.answer_gaps_path.exists():
        error = _probe_writable(settings.answer_gaps_path.parent)
        if error is not None:
            return (label, False, f"{settings.answer_gaps_path.parent}: {error}")
        return (label, True, "")
    try:
        data = json.loads(settings.answer_gaps_path.read_text(encoding="utf-8"))
    except OSError as e:
        return (label, False, f"{settings.answer_gaps_path}: {e}")
    except UnicodeDecodeError as e:
        return (label, False, f"{settings.answer_gaps_path}: not valid UTF-8 ({e})")
    except json.JSONDecodeError as e:
        return (label, False, f"{settings.answer_gaps_path}: not valid JSON ({e})")
    if not isinstance(data, dict):
        return (
            label,
            False,
            f'{settings.answer_gaps_path}: must be a JSON object of {{"question": {{...}}}} entries',
        )
    return (label, True, "")


def _applications_dir_check(settings: Settings) -> tuple[str, bool, str]:
    """generate_materials() (cmd_run) creates a new subdirectory under
    APPLICATIONS_DIR for every single job that clears the fit gate, and
    never checked it was actually writable first - a misconfigured path or
    a permission-denied directory previously only surfaced as a confusing
    prep_error on the very first posting worth applying to, well past
    `job-bot doctor` giving a clean bill of health. Actually creates the
    directory and writes/removes a small probe file (_probe_writable()
    above), the same "verify, not just check existence" reasoning
    _resume_check uses (an existing-but-unwritable directory would pass a
    bare .exists() check and still fail the first real run).
    """
    label = "Applications directory writable"
    error = _probe_writable(settings.applications_dir)
    if error is not None:
        return (label, False, f"{settings.applications_dir}: {error}")
    return (label, True, str(settings.applications_dir))


def _audit_log_check(settings: Settings) -> tuple[str, bool, str]:
    """AuditLogger.__init__() only creates AUDIT_LOG_PATH's parent
    directory (mkdir) - it never opens or writes the file itself, so an
    unwritable directory (permissions, a full disk, audit_log_path
    colliding with an existing directory of the same name) wasn't caught
    until the very first real action `job-bot run` took tried to log it,
    deep into a run, rather than upfront here the way
    _applications_dir_check already catches the equivalent problem for
    APPLICATIONS_DIR. Not optional the way the Gmail checks are: every
    apply/skip decision cmd_run makes is meant to go through here (see
    safety/audit_log.py's own module docstring), so a silently-unwritable
    audit log is a real gap in the safety trail, not a missing nice-to-have.

    Opens the file in append mode and immediately closes it without
    writing anything - proves it's writable (and, on a fresh install,
    creates the empty file the same way the very first real log() call
    would) without adding a stray probe entry to an otherwise meaningful
    audit trail.
    """
    label = "Audit log writable"
    try:
        settings.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        with settings.audit_log_path.open("a", encoding="utf-8"):
            pass
    except OSError as e:
        return (label, False, f"{settings.audit_log_path}: {e}")
    return (label, True, str(settings.audit_log_path))


def _failed_applications_log_check(settings: Settings) -> tuple[str, bool, str]:
    """Same gap as _audit_log_check above, for FAILED_APPLICATIONS_LOG_PATH -
    cmd_run constructs an AuditLogger for it unconditionally on every
    `job-bot run` (alongside the main audit log), so the identical
    unwritable-directory risk applies: previously uncaught until the very
    first search_error/prep_error/apply_error tried to log to it, deep into
    a run, rather than upfront here. Now readable via `job-bot audit-log
    --failures` too - same reasoning as _audit_log_check for why this isn't
    optional: it's the one place a user is told to look for what happened
    and why (see _print_cycle_summary()) when postings couldn't be
    completed, so a silently-unwritable copy of it defeats that entirely.
    """
    label = "Failed-applications log writable"
    try:
        settings.failed_applications_log_path.parent.mkdir(parents=True, exist_ok=True)
        with settings.failed_applications_log_path.open("a", encoding="utf-8"):
            pass
    except OSError as e:
        return (label, False, f"{settings.failed_applications_log_path}: {e}")
    return (label, True, str(settings.failed_applications_log_path))


def _tracker_db_check(settings: Settings) -> tuple[str, bool, str]:
    """Every other command opens the tracker eagerly (Tracker.__init__()'s
    _init_db() runs a schema migration on every construction), so a
    present-but-corrupted db.sqlite3 (truncated by a killed process, disk
    corruption, or simply not a SQLite file) doesn't fail quietly the way
    a malformed JSON store does - it already raises loudly (main() also
    turns that into a clean message for whichever command hits it first,
    see its own sqlite3.DatabaseError handler). This check exists to give
    that diagnosis upfront, before a real run gets partway through and
    hits it. A missing file is fine - `job-bot run`/every other command
    creates it fresh - so only a present-but-unreadable one is a real
    problem.
    """
    label = "Tracker database readable"
    if not settings.db_path.exists():
        return (label, True, "not created yet - job-bot run will create it")
    try:
        Tracker(settings.db_path)
    except sqlite3.DatabaseError as e:
        return (label, False, f"{settings.db_path}: {e}")
    return (label, True, str(settings.db_path))


def _daily_cap_usage_check(settings: Settings) -> tuple[str, bool, str]:
    """Today's application count against the effective daily cap - directly
    actionable in a way a pure file/config check can't be: someone running
    `job-bot doctor` to understand why `job-bot run` isn't applying to
    anything (or won't get far into a fresh --loop) previously had no way
    to see "you've already used today's cap" without starting a real run
    and watching it apply to nothing. Flagged (not just informational) once
    the cap is actually reached, since that's the specific, common reason
    a run does nothing today - same "ok=False for a real, actionable
    reason" treatment every other check here gets, not a status this one
    alone is exempt from.

    RateLimiter shares db_path with Tracker but doesn't guard against a
    corrupted file the way _tracker_db_check() above does - same
    sqlite3.DatabaseError catch here, or a corrupted db.sqlite3 crashed
    doctor entirely on this check instead of giving the clean diagnosis
    _tracker_db_check() already exists to provide.
    """
    label = "Daily application cap available"
    try:
        rate_limiter = RateLimiter(settings.db_path, settings.effective_daily_cap())
        used = rate_limiter.count_today()
    except sqlite3.DatabaseError:
        return (label, True, "skipped - tracker database is corrupted, see the check above")
    cap = settings.effective_daily_cap()
    remaining = rate_limiter.remaining_today()
    detail = f"{used}/{cap} applications submitted today"
    if remaining <= 0:
        detail += " - job-bot run will apply to nothing more until tomorrow"
    return (label, remaining > 0, detail)


def cmd_doctor(settings: Settings, args: argparse.Namespace) -> None:
    """Check local setup for the common ways `job-bot run` fails partway
    through rather than up front - deliberately file/config checks only, no
    network calls, so it's fast and safe to run anytime. LLM connectivity
    itself (which does make a real API call) is `job-bot test-provider`'s job.
    `--format json` prints the same checks as one JSON object instead - for
    a setup script or health-check cron job that wants to act on pass/fail
    programmatically rather than parsing the human-readable [OK]/[!!] lines.
    """
    checks: list[tuple[str, bool, str]] = [
        _resume_check(settings),
        _blacklist_check(settings),
        _faq_check(settings),
        _faq_usability_check(settings),
        _answer_gaps_check(settings),
        _applications_dir_check(settings),
        _audit_log_check(settings),
        _failed_applications_log_check(settings),
        _tracker_db_check(settings),
    ]

    if settings.llm_provider == "claude":
        checks.append(("Anthropic API key (ANTHROPIC_API_KEY)", bool(settings.anthropic_api_key), ""))
    else:
        checks.append(
            ("Ollama base URL configured", bool(settings.ollama_base_url), settings.ollama_base_url)
        )

    session_ready = settings.browser_profile_dir.exists() and any(settings.browser_profile_dir.iterdir())
    checks.append(
        (
            "LinkedIn session saved",
            session_ready,
            str(settings.browser_profile_dir) if session_ready else "run `job-bot login` first",
        )
    )

    gmail_ready = settings.gmail_credentials_path.exists()
    checks.append(
        (
            "Gmail credentials (optional, for gmail-sync)",
            gmail_ready,
            "" if gmail_ready else str(settings.gmail_credentials_path),
        )
    )

    # The parallel to "LinkedIn session saved" above: credentials.json alone
    # only means gmail-sync *can* be set up, not that the one-time OAuth
    # consent (which writes gmail_token.json - see gmail_client.py) was ever
    # actually completed. Without this, doctor gave no way to tell "not
    # configured at all" apart from "configured but never authorized" -
    # both looked identical (silence) until the first `job-bot gmail-sync`
    # either opened a browser tab for consent or, on a headless/CI box with
    # no browser available, failed outright.
    gmail_authorized = settings.gmail_token_path.exists()
    checks.append(
        (
            "Gmail authorized (optional, run `job-bot gmail-sync` once)",
            gmail_authorized,
            "" if gmail_authorized else "run `job-bot gmail-sync` once to complete the one-time OAuth consent",
        )
    )

    cap_ok = settings.daily_application_cap <= HARD_DAILY_APPLICATION_CEILING
    checks.append(
        (
            "Daily application cap within hard ceiling",
            cap_ok,
            ""
            if cap_ok
            else (
                f"{settings.daily_application_cap} > {HARD_DAILY_APPLICATION_CEILING}; "
                "the ceiling will be used instead"
            ),
        )
    )
    checks.append(_daily_cap_usage_check(settings))

    passed = sum(ok for _, ok, _ in checks)

    if args.format == "json":
        payload = {
            "checks": [{"label": label, "ok": ok, "detail": detail} for label, ok, detail in checks],
            "passed": passed,
            "total": len(checks),
        }
        print(json.dumps(payload, indent=2))
        return

    print("job-bot doctor")
    print("-" * 40)
    for label, ok, detail in checks:
        line = f"[{'OK' if ok else '!!'}] {label}"
        if detail:
            line += f" - {detail}"
        print(line)

    print("-" * 40)
    print(f"{passed}/{len(checks)} checks passed.")
    print("Run `job-bot test-provider` to verify LLM connectivity (makes one real API call).")


def build_parser() -> argparse.ArgumentParser:
    """Defines every `job-bot <command>` and its flags. Each subparser here
    pairs with one `cmd_*` function above and one branch in main()'s
    dispatch - run `job-bot <command> --help` for the flags themselves
    rather than reading this as documentation.
    """
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
    run_p.add_argument(
        "--loop",
        action="store_true",
        help=(
            "Keep running all day instead of stopping after one search batch: re-searches and "
            "applies in cycles, back-to-back as long as postings keep turning up, until today's "
            "application cap is reached or you stop it with Ctrl+C. --loop-interval-minutes only "
            "paces the cycles where nothing was applied. --max-apps then caps applications per "
            "cycle, not for the whole day - the daily cap is what bounds the day as a whole."
        ),
    )
    run_p.add_argument(
        "--loop-interval-minutes",
        type=int,
        default=20,
        help=(
            "Minutes to wait before retrying in --loop mode, but only after a cycle that applied "
            "to nothing (default: 20) - a cycle that did apply to something starts the next one "
            "immediately."
        ),
    )
    run_p.add_argument("--provider", choices=["claude", "ollama"], default=None)
    run_p.add_argument("--model", default=None)
    run_p.add_argument(
        "--yes-i-understand-the-risk",
        action="store_true",
        help="Skip the per-application confirmation prompt. The daily cap still applies.",
    )
    run_p.add_argument(
        "--min-score",
        type=int,
        default=None,
        help=(
            "Extra floor on top of the model's own should_apply verdict - a posting is only "
            "applied to if should_apply is True AND its score clears this too "
            "(default: from .env, MIN_MATCH_SCORE)."
        ),
    )
    run_p.add_argument(
        "--exclude-title-keywords",
        default=None,
        help=(
            "Comma-separated, case-insensitive substrings - a posting whose title contains any "
            "of these is skipped before it's scored (default: from .env, EXCLUDE_TITLE_KEYWORDS)."
        ),
    )
    run_p.add_argument(
        "--experience-level",
        default=None,
        help=(
            "Comma-separated LinkedIn seniority levels to restrict the search to: "
            f"{', '.join(sorted(EXPERIENCE_LEVEL_CODES))} (default: from .env, DEFAULT_EXPERIENCE_LEVELS)."
        ),
    )
    run_p.add_argument(
        "--max-years-experience",
        type=int,
        default=None,
        help=(
            "Treat a posting explicitly requiring more years of experience than this, or "
            "explicitly Senior/Staff/Principal/Lead/Director-or-higher, as ineligible regardless "
            "of match score (default: from .env, MAX_YEARS_EXPERIENCE; unset means no check)."
        ),
    )
    run_p.add_argument(
        "--require-w2",
        action="store_true",
        help=(
            "Treat a posting explicitly stated as Corp-to-Corp (C2C), 1099, or otherwise not "
            "offered as direct W2 employment as ineligible (default: from .env, REQUIRE_W2). "
            "Only adds the restriction for this run - it can't be used to turn REQUIRE_W2=true "
            "off for one run."
        ),
    )
    run_p.add_argument(
        "--include-external-apply",
        action="store_true",
        help=(
            "EXPERIMENTAL: also apply to postings with no Easy Apply, on the employer's own site, "
            "via a best-effort generic form filler. Always confirms before submitting, regardless "
            "of --yes-i-understand-the-risk. See README.md's \"Applying on company websites "
            '(experimental)" section first. (default: from .env, ENABLE_EXTERNAL_APPLY)'
        ),
    )

    test_provider_p = sub.add_parser(
        "test-provider", help="Sanity-check the configured LLM provider with one call."
    )
    test_provider_p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Print the result as one JSON object instead.",
    )

    doctor_p = sub.add_parser(
        "doctor", help="Check local setup (resume, API key, LinkedIn session, Gmail creds) for problems."
    )
    doctor_p.add_argument(
        "--format", choices=["text", "json"], default="text", help="Print as one JSON object instead."
    )

    status_p = sub.add_parser(
        "status",
        help="Record an application outcome by hand, or view one job's record with no <status>.",
    )
    status_p.add_argument(
        "job_id", help="The LinkedIn job id, as shown in `job-bot report` or the audit log."
    )
    status_p.add_argument(
        "status",
        nargs="?",
        default=None,
        choices=sorted(TRACKER_STATUSES),
        help="New status to record. Omit (with no --note either) to print the job's current record instead.",
    )
    status_p.add_argument(
        "--note", default=None, help="Set a free-text note on this job (e.g. salary info from a call)."
    )
    status_p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Print the view (no <status>/--note given) as one JSON object instead.",
    )

    review_answers_p = sub.add_parser(
        "review-answers",
        help=(
            "Answer required questions Easy Apply couldn't confidently answer on its own - "
            "saved answers are reused automatically on every future posting that asks the same question."
        ),
    )
    review_answers_p.add_argument(
        "--search",
        default=None,
        help="Only review/print gaps whose question contains this text (case-insensitive).",
    )
    review_answers_p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Print the unanswered questions as one JSON array instead of prompting for answers.",
    )
    review_answers_p.add_argument(
        "--dismiss",
        nargs="+",
        default=None,
        metavar="QUESTION",
        help="Permanently discard one or more gaps (exact question text(s), as shown by this "
        "command's own output) without answering them - unlike leaving the prompt blank, which "
        "re-queues a gap for next time, this removes it for good. Doesn't touch FAQ_PATH.",
    )

    faq_p = sub.add_parser(
        "faq", help="View, remove, import, or export cached FAQ answers (see `job-bot review-answers`)."
    )
    faq_sub = faq_p.add_subparsers(dest="faq_action", required=True)
    faq_list_p = faq_sub.add_parser("list", help="Print every cached question/answer pair.")
    faq_list_p.add_argument(
        "--search",
        default=None,
        help="Only print pairs whose question or answer contains this text (case-insensitive).",
    )
    faq_list_p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Print the (optionally --search-filtered) pairs as one JSON object instead.",
    )
    faq_remove_p = faq_sub.add_parser(
        "remove", help="Remove one or more cached answers, e.g. to fix a wrong one."
    )
    faq_remove_p.add_argument(
        "question",
        nargs="+",
        help="The exact question text(s), as shown by `job-bot faq list`. Quote each separately.",
    )
    faq_clean_p = faq_sub.add_parser(
        "clean",
        help="Remove cached answers a run would skip anyway (echoing the question, or leaked reasoning).",
    )
    faq_clean_p.add_argument(
        "--dry-run", action="store_true", help="Only list what would be removed; change nothing."
    )
    faq_import_p = faq_sub.add_parser(
        "import", help="Merge in answers from a JSON file (a backup, or another install's FAQ_PATH)."
    )
    faq_import_p.add_argument(
        "file", type=Path, help='JSON object of {"question": "answer"} pairs - the same shape FAQ_PATH is.'
    )
    faq_export_p = faq_sub.add_parser(
        "export", help="Write cached FAQ answers out as JSON (the same shape `faq import` reads back)."
    )
    faq_export_p.add_argument(
        "--out", type=Path, default=None, help="Write to this file instead of stdout."
    )

    qa_history_p = sub.add_parser(
        "qa-history",
        help="Print every question/answer pair ever recorded, across every job, most recent first.",
    )
    qa_history_p.add_argument(
        "--search",
        default=None,
        help="Only print pairs whose question or answer contains this text (case-insensitive).",
    )
    qa_history_p.add_argument(
        "--company",
        default=None,
        help="Only print pairs for this exact company (case/spacing-insensitive), not a substring "
        "match like --search.",
    )
    qa_history_p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Print the pairs as one JSON array instead.",
    )

    resume_history_p = sub.add_parser(
        "resume-history",
        help="Print every tailored-resume generation ever recorded, across every job, most recent first.",
    )
    resume_history_p.add_argument(
        "--search",
        default=None,
        help="Only print generations whose summary, company, or title contains this text "
        "(case-insensitive).",
    )
    resume_history_p.add_argument(
        "--company",
        default=None,
        help="Only print generations for this exact company (case/spacing-insensitive), not a "
        "substring match like --search.",
    )
    resume_history_p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Print the generations as one JSON array instead.",
    )

    audit_log_p = sub.add_parser(
        "audit-log",
        help="Print every logged action from the safety audit trail, most recent first.",
    )
    audit_log_p.add_argument(
        "--search",
        default=None,
        help="Only print entries whose action or details contain this text (case-insensitive).",
    )
    audit_log_p.add_argument(
        "--action",
        default=None,
        help="Only print entries with this exact action (e.g. applied, skip_blacklisted, "
        "gmail_sync_update).",
    )
    audit_log_p.add_argument(
        "--failures",
        action="store_true",
        help="Read FAILED_APPLICATIONS_LOG_PATH instead of AUDIT_LOG_PATH - the postings a run "
        "couldn't finish and why, the same file a failed run tells you to check by hand.",
    )
    audit_log_p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Print the entries as one JSON array instead.",
    )

    report_p = sub.add_parser("report", help="Print a count of tracked jobs by status.")
    report_p.add_argument(
        "--stale-days",
        type=int,
        default=None,
        help="Flag applications with no reply after this many days (default: from .env).",
    )
    report_p.add_argument(
        "--by-score", action="store_true", help="Break outcomes down by match-score bucket."
    )
    report_p.add_argument(
        "--by-eligibility",
        action="store_true",
        help="Break outcomes down by eligibility-gate verdict (pass/flag/fail/not scored).",
    )
    report_p.add_argument(
        "--by-company",
        action="store_true",
        help="Break outcomes down by company, most-applied first.",
    )
    report_p.add_argument(
        "--by-week",
        action="store_true",
        help="Count applications sent per week (local time, most recent first), by current status.",
    )
    report_p.add_argument(
        "--by-missing-qualifications",
        action="store_true",
        help="Show which missing qualifications the LLM scorer flags most often across postings.",
    )
    report_p.add_argument(
        "--missing-qualifications-limit",
        type=int,
        default=None,
        help=(
            "With --by-missing-qualifications, keep only its N most common phrases (default: no "
            "limit) - useful once most distinct phrases have count=1 and bury the ones that "
            "actually repeat."
        ),
    )
    report_p.add_argument(
        "--format", choices=["text", "json"], default="text", help="Print as one JSON object instead."
    )

    export_p = sub.add_parser("export", help="Export tracked jobs as CSV or JSON.")
    export_p.add_argument("--status", choices=sorted(TRACKER_STATUSES), default=None)
    export_p.add_argument(
        "--eligibility",
        choices=["pass", "fail", "flag"],
        default=None,
        help="Only export jobs with this eligibility-gate verdict.",
    )
    export_p.add_argument(
        "--search",
        default=None,
        help="Only export jobs whose title, company, note, match reasoning, eligibility note, or "
        "missing qualifications contains this text.",
    )
    export_p.add_argument(
        "--company",
        default=None,
        help="Only export jobs at this exact company (case/spacing-insensitive), not a substring "
        "match like --search.",
    )
    export_p.add_argument(
        "--stale-days",
        type=int,
        default=None,
        help="Only export applied jobs with no reply after this many days "
        "(same rule `job-bot report --stale-days` uses).",
    )
    export_p.add_argument("--format", choices=["csv", "json"], default="csv")
    export_p.add_argument(
        "--out", type=Path, default=None, help="Write to this file instead of stdout."
    )

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
    gmail_p.add_argument(
        "--format", choices=["text", "json"], default="text", help="Print the result as one JSON object instead."
    )

    dashboard_p = sub.add_parser("dashboard", help="Serve a live one-page view of the tracker at localhost.")
    dashboard_p.add_argument("--port", type=int, default=None, help="default: from .env (8765)")
    dashboard_p.add_argument("--no-open", action="store_true", help="Don't auto-open a browser tab.")

    blacklist_p = sub.add_parser("blacklist", help="Manage the list of companies to never apply to.")
    blacklist_sub = blacklist_p.add_subparsers(dest="blacklist_action", required=True)
    add_p = blacklist_sub.add_parser("add", help="Add one or more companies to the blacklist.")
    add_p.add_argument("company", nargs="+", help="One or more company names, each quoted separately.")
    add_p.add_argument(
        "--reason",
        default=None,
        help="Why these are blacklisted (optional, e.g. 'no H1B sponsorship') - applies to every "
        "company in this call, shown by `job-bot blacklist list`.",
    )
    remove_p = blacklist_sub.add_parser("remove", help="Remove one or more companies from the blacklist.")
    remove_p.add_argument("company", nargs="+", help="One or more company names, each quoted separately.")
    check_p = blacklist_sub.add_parser(
        "check", help="Check whether one company is blacklisted, and why."
    )
    check_p.add_argument("company", help="Company name to check (case/spacing-insensitive).")
    check_p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Print the result as one JSON object instead.",
    )
    blacklist_list_p = blacklist_sub.add_parser("list", help="List blacklisted companies.")
    blacklist_list_p.add_argument(
        "--search",
        default=None,
        help="Only print companies whose name or reason contains this text (case-insensitive).",
    )
    blacklist_list_p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Print the (optionally --search-filtered) companies as one JSON array instead.",
    )
    import_p = blacklist_sub.add_parser(
        "import", help="Add every company listed in a text file (one per line)."
    )
    import_p.add_argument(
        "file", type=Path, help="Plain text file, one company per line. Blank and '#'-comment lines skipped."
    )
    blacklist_export_p = blacklist_sub.add_parser(
        "export", help="Write the blacklist out as text, one company per line (the same shape `import` reads)."
    )
    blacklist_export_p.add_argument(
        "--out", type=Path, default=None, help="Write to this file instead of stdout."
    )

    return parser


def main() -> None:
    """Entry point registered as the `job-bot` console script (see
    pyproject.toml's [project.scripts]). Parses argv, builds one Settings
    for the whole invocation, dispatches to the matching cmd_*, and turns
    any EXPECTED_ERRORS into a clean one-line message instead of a
    traceback - anything else raised here is a bug, not user error.
    """
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
            cmd_test_provider(settings, args)
        elif args.command == "doctor":
            cmd_doctor(settings, args)
        elif args.command == "status":
            cmd_status(settings, args)
        elif args.command == "review-answers":
            cmd_review_answers(settings, args)
        elif args.command == "report":
            cmd_report(settings, args)
        elif args.command == "export":
            cmd_export(settings, args)
        elif args.command == "gmail-sync":
            cmd_gmail_sync(settings, args)
        elif args.command == "dashboard":
            cmd_dashboard(settings, args)
        elif args.command == "blacklist":
            cmd_blacklist(settings, args)
        elif args.command == "faq":
            cmd_faq(settings, args)
        elif args.command == "qa-history":
            cmd_qa_history(settings, args)
        elif args.command == "resume-history":
            cmd_resume_history(settings, args)
        elif args.command == "audit-log":
            cmd_audit_log(settings, args)
    except EXPECTED_ERRORS as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except sqlite3.DatabaseError as e:
        # Every command that touches the tracker (Tracker.__init__()'s
        # _init_db() runs a schema migration on every construction) opens
        # it eagerly, so a present-but-corrupted db.sqlite3 (truncated by a
        # killed process, disk corruption, or simply not a SQLite file)
        # previously raised this raw, uncaught - sqlite3.DatabaseError isn't
        # in EXPECTED_ERRORS, so it fell all the way through main() as a
        # traceback instead of the clean, actionable message every other
        # expected failure here gets. `job-bot doctor` already diagnoses
        # exactly this (_tracker_db_check's own docstring calls out this
        # gap by name), but only when doctor itself is the command run -
        # every other command still hit the raw crash until this handler.
        print(
            f"Error: The tracker database at {settings.db_path} looks corrupted ({e}). "
            "Run `job-bot doctor` for a clean diagnosis, or restore/replace it from a backup.",
            file=sys.stderr,
        )
        sys.exit(1)
    except KeyboardInterrupt:
        # Without this, Ctrl+C during a real browser action (mid Easy Apply
        # form, waiting on Ollama, ...) propagated a raw traceback through
        # browser_session()'s own cleanup - confusing on its own, and it
        # also left the Chromium profile in a state where the *next* run
        # failed outright with "profile is already in use by another
        # instance of Chromium" (seen live) since the interrupted close()
        # never finished. A clean stop here doesn't fix that underlying
        # profile-lock risk, but it does stop dumping a scary traceback for
        # what is a completely normal way to end a run.
        print("\nStopped.")
        sys.exit(1)


if __name__ == "__main__":
    main()
