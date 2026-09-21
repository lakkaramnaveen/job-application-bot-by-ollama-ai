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
    UnansweredRequiredQuestion,
)
from job_bot.browser.session import BrowserSessionError, browser_session
from job_bot.config import HARD_DAILY_APPLICATION_CEILING, Settings, SettingsError, get_settings
from job_bot.dashboard.server import run_dashboard
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
from job_bot.resume.parser import ResumeParseError
from job_bot.resume.store import ResumeStore
from job_bot.safety.answer_gaps import AnswerGapStore
from job_bot.safety.audit_log import AuditLogger
from job_bot.safety.blacklist import CompanyBlacklist
from job_bot.safety.confirm import SubmitConfirmer
from job_bot.safety.rate_limiter import DailyCapReached, RateLimiter
from job_bot.tracker.db import TRACKER_STATUSES, InvalidStatus, Tracker, write_export_csv

EXPECTED_ERRORS = (
    ClaudeProviderError,
    OllamaProviderError,
    ResumeParseError,
    SettingsError,
    InvalidStatus,
    GmailClientError,
    UnsafeJobId,
    BrowserSessionError,
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

        def run_one_cycle() -> tuple[int, int]:
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
            applied, failed = run_one_cycle()
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
                applied, failed = run_one_cycle()
                _print_cycle_summary(applied, failed, rate_limiter, settings)
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
            f"{failed} posting(s) could not be completed - see "
            f"{settings.failed_applications_log_path} for what happened and why."
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
) -> tuple[int, int]:
    """One search -> score -> tailor -> apply pass over a fresh batch of
    postings. Called once for a plain `job-bot run`, or repeatedly for
    `--loop` (back-to-back with no sleep as long as each cycle keeps
    applying to something; only a cycle that applies to nothing pauses
    before the next one) - re-running search() each cycle is what lets loop
    mode pick up postings that appeared after the previous cycle, not just
    the ones visible at process start. Returns (applied, failed) for that
    cycle only, not a running total across cycles.
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
        return 0, 1
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
            posting.job_id, posting.title, posting.company, posting.url, match.score, should_apply
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
        if cached is not None:
            tracker.record_qa(job_id, question, cached)
            return cached
        result = answer_question(
            provider,
            resume_text,
            faq_answers,
            question,
            recent_answers=tracker.recent_qa_pairs(),
        )
        tracker.record_qa(job_id, question, result.answer)
        if result.based_on_resume and result.confidence >= settings.faq_save_confidence:
            resume_store.save_faq_answer(question, result.answer)
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
                answer_gaps.record(
                    e.question, job_id=posting.job_id, company=posting.company, title=posting.title
                )
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

    return applied, failed


def cmd_status(settings: Settings, args: argparse.Namespace) -> None:
    """Record an outcome the bot has no way to observe on its own -
    `job-bot run` only ever writes seen/applied/skipped; everything past
    that (interviewing, offer, ...) is reported by the user by hand, or by
    `job-bot gmail-sync` reading a reply email.
    """
    tracker = Tracker(settings.db_path)
    try:
        tracker.update_status(args.job_id, args.status)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"{args.job_id} -> {args.status}")


def cmd_review_answers(settings: Settings) -> None:
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
    """
    resume_store = ResumeStore(settings.resume_path, settings.faq_path)
    answer_gaps = AnswerGapStore(settings.answer_gaps_path)
    gaps = answer_gaps.list_unanswered()
    if not gaps:
        print("No unanswered required questions recorded - nothing to review.")
        return

    ordered = sorted(gaps.items(), key=lambda item: item[1].get("count", 0), reverse=True)
    print(f"{len(ordered)} unanswered required question(s):\n")
    answered = 0
    for question, info in ordered:
        count = info.get("count", 1)
        times = "time" if count == 1 else "times"
        print(f'"{question}"')
        print(f"  seen {count} {times}, e.g. {info.get('example_title', '')} at {info.get('example_company', '')}")
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


# Score buckets for `job-bot report --by-score`'s outcome breakdown, widest
# (worst-fit) first so a job with a null match_score never fits any bucket
# and is simply left out - it hasn't been through the LLM scorer yet.
SCORE_BUCKETS = ((0, 59), (60, 69), (70, 79), (80, 89), (90, 100))


def _score_bucket_label(score: int) -> str:
    for lo, hi in SCORE_BUCKETS:
        if lo <= score <= hi:
            return f"{lo}-{hi}"
    return "?"


def _print_score_breakdown(tracker: Tracker) -> None:
    scored_jobs = [job for job in tracker.list_jobs() if job["match_score"] is not None]
    if not scored_jobs:
        return

    buckets: dict[str, dict[str, int]] = {}
    for job in scored_jobs:
        by_status = buckets.setdefault(_score_bucket_label(job["match_score"]), {})
        by_status[job["status"]] = by_status.get(job["status"], 0) + 1

    statuses = sorted({job["status"] for job in scored_jobs})
    print("\nOutcomes by match score:")
    header = "score".ljust(8) + "".join(status.ljust(14) for status in statuses) + "total"
    print(header)
    for lo, hi in SCORE_BUCKETS:
        bucket_counts = buckets.get(f"{lo}-{hi}")
        if not bucket_counts:
            continue
        row = f"{lo}-{hi}".ljust(8) + "".join(str(bucket_counts.get(s, 0)).ljust(14) for s in statuses)
        print(row + str(sum(bucket_counts.values())))


def _stale_applications(tracker: Tracker, days: int) -> list[dict]:
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    return [
        job
        for job in tracker.list_jobs(status="applied", sort="applied_at", direction="asc")
        if job["applied_at"] and job["applied_at"] < cutoff
    ]


def cmd_report(settings: Settings, args: argparse.Namespace) -> None:
    """Print status counts, plus two optional sections: `--stale-days`
    (applications with no reply worth a manual follow-up) and `--by-score`
    (how match score correlates with actual outcomes, to sanity-check
    whether the LLM scorer's judgment tracks reality).
    """
    tracker = Tracker(settings.db_path)
    counts = tracker.status_counts()
    if not counts:
        print("No jobs tracked yet.")
        return
    width = max(len(status) for status in counts)
    for status in sorted(counts):
        print(f"{status:<{width}}  {counts[status]}")
    print(f"{'total':<{width}}  {sum(counts.values())}")

    stale_days = args.stale_days if args.stale_days is not None else settings.stale_after_days
    stale = _stale_applications(tracker, stale_days)
    if stale:
        print(f"\nApplied {stale_days}+ days ago with no reply ({len(stale)}):")
        for job in stale:
            print(f"  {job['job_id']}  {job['company']} - {job['title']}  (applied {job['applied_at'][:10]})")

    if args.by_score:
        _print_score_breakdown(tracker)


def cmd_export(settings: Settings, args: argparse.Namespace) -> None:
    """Dump tracked jobs as CSV - to a file with `--out`, or stdout so it
    pipes straight into another tool.
    """
    tracker = Tracker(settings.db_path)
    jobs = tracker.list_jobs(status=args.status, sort="first_seen_at", direction="asc")

    if args.out:
        # newline="" so csv's own \r\n line terminator isn't doubled up by
        # universal-newline text-mode translation on write.
        with args.out.open("w", newline="", encoding="utf-8") as f:
            write_export_csv(f, jobs)
        print(f"Exported {len(jobs)} job(s) to {args.out}")
    else:
        write_export_csv(sys.stdout, jobs)


def cmd_gmail_sync(settings: Settings, args: argparse.Namespace) -> None:
    """Read recent Gmail, classify each message, and advance the matching
    tracked job's status - see gmail_sync.py's module docstring for the
    exact never-guess/never-downgrade rules this delegates to.
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


def cmd_blacklist(settings: Settings, args: argparse.Namespace) -> None:
    """add/remove/list companies `job-bot run` will always skip - see
    build_parser()'s `blacklist` subparser for the three actions.
    """
    blacklist = CompanyBlacklist(settings.blacklist_path)
    if args.blacklist_action == "add":
        blacklist.add(args.company)
        print(f"Added to blacklist: {args.company}")
    elif args.blacklist_action == "remove":
        removed = blacklist.remove(args.company)
        print(
            f"Removed from blacklist: {args.company}" if removed else f"Not on the blacklist: {args.company}"
        )
    elif args.blacklist_action == "list":
        companies = blacklist.list_companies()
        if not companies:
            print("Blacklist is empty.")
        else:
            for company in companies:
                print(company)


def cmd_dashboard(settings: Settings, args: argparse.Namespace) -> None:
    """Serve the local tracker dashboard - see dashboard/server.py's module
    docstring for why it binds to localhost only and has no login.
    """
    port = args.port if args.port is not None else settings.dashboard_port
    run_dashboard(settings.db_path, port=port, open_browser=not args.no_open)


def cmd_test_provider(settings: Settings) -> None:
    """One real API call to the configured LLM provider, to confirm the key/
    model/local server actually work before trusting them to score real
    postings. See also `job-bot doctor`, which checks everything else
    (resume, LinkedIn session, ...) without making a network call.
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
    print(f"Provider OK: {settings.llm_provider}")
    print(result)


def cmd_doctor(settings: Settings) -> None:
    """Check local setup for the common ways `job-bot run` fails partway
    through rather than up front - deliberately file/config checks only, no
    network calls, so it's fast and safe to run anytime. LLM connectivity
    itself (which does make a real API call) is `job-bot test-provider`'s job.
    """
    print("job-bot doctor")
    print("-" * 40)

    checks: list[tuple[str, bool, str]] = [
        ("Resume file", settings.resume_path.exists(), str(settings.resume_path)),
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

    passed = 0
    for label, ok, detail in checks:
        line = f"[{'OK' if ok else '!!'}] {label}"
        if detail:
            line += f" - {detail}"
        print(line)
        passed += ok

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

    sub.add_parser("test-provider", help="Sanity-check the configured LLM provider with one call.")

    sub.add_parser(
        "doctor", help="Check local setup (resume, API key, LinkedIn session, Gmail creds) for problems."
    )

    status_p = sub.add_parser(
        "status", help="Record an application outcome (interviewing, offer, rejected, ...) by hand."
    )
    status_p.add_argument(
        "job_id", help="The LinkedIn job id, as shown in `job-bot report` or the audit log."
    )
    status_p.add_argument("status", choices=sorted(TRACKER_STATUSES))

    sub.add_parser(
        "review-answers",
        help=(
            "Answer required questions Easy Apply couldn't confidently answer on its own - "
            "saved answers are reused automatically on every future posting that asks the same question."
        ),
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

    export_p = sub.add_parser("export", help="Export tracked jobs as CSV.")
    export_p.add_argument("--status", choices=sorted(TRACKER_STATUSES), default=None)
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

    dashboard_p = sub.add_parser("dashboard", help="Serve a live one-page view of the tracker at localhost.")
    dashboard_p.add_argument("--port", type=int, default=None, help="default: from .env (8765)")
    dashboard_p.add_argument("--no-open", action="store_true", help="Don't auto-open a browser tab.")

    blacklist_p = sub.add_parser("blacklist", help="Manage the list of companies to never apply to.")
    blacklist_sub = blacklist_p.add_subparsers(dest="blacklist_action", required=True)
    add_p = blacklist_sub.add_parser("add", help="Add a company to the blacklist.")
    add_p.add_argument("company")
    remove_p = blacklist_sub.add_parser("remove", help="Remove a company from the blacklist.")
    remove_p.add_argument("company")
    blacklist_sub.add_parser("list", help="List blacklisted companies.")

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
            cmd_test_provider(settings)
        elif args.command == "doctor":
            cmd_doctor(settings)
        elif args.command == "status":
            cmd_status(settings, args)
        elif args.command == "review-answers":
            cmd_review_answers(settings)
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
    except EXPECTED_ERRORS as e:
        print(f"Error: {e}", file=sys.stderr)
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
