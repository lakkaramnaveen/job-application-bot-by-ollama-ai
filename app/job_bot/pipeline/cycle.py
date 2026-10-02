"""One search -> score -> tailor -> apply pass: the body of `job-bot run`.

Step 5 of docs/architecture.md. This was cli.py's _run_apply_cycle() -
moved here whole, with the policies it composes already extracted
(failures.py, skip.py, answers.py) and its shared dependencies in one
RunContext (context.py). cli.py stays the composition root: it builds the
context and the browser session, drives --loop, and passes in the two
things tests substitute (the external-site adapter class and the form
time limit) as arguments rather than this module importing them.
"""

from __future__ import annotations

import contextlib
import dataclasses
import signal
import threading
from collections.abc import Callable, Iterator
from typing import Any

from playwright.sync_api import Page

from job_bot.browser.base_adapter import JobPosting
from job_bot.browser.linkedin_adapter import (
    FieldsRejected,
    LinkedInAdapter,
    LinkedInSignedOut,
    NavigationFailed,
    UnansweredRequiredQuestion,
)
from job_bot.data_files import CorruptDataFile
from job_bot.generation.artifacts import write_cover_letter, write_tailored_resume, write_tailored_resume_docx
from job_bot.generation.cover_letter import generate_cover_letter
from job_bot.generation.qa_answerer import answer_question
from job_bot.generation.resume_tailor import tailor_resume
from job_bot.matching.scorer import score_job_match
from job_bot.models.schemas import CoverLetter, JobMatchScore, TailoredResume
from job_bot.pipeline.answers import AnswerService
from job_bot.pipeline.context import RunContext
from job_bot.pipeline.failures import BROWSER_GONE_MESSAGE, _browser_is_gone, classify_failure
from job_bot.pipeline.skip import SkipPolicy, SkipReason
from job_bot.safety.rate_limiter import DailyCapReached

# Consecutive job-page load failures (NavigationFailed, i.e. already past
# _goto_with_retry()'s own retries) after which a cycle stops - see its use
# in _run_apply_cycle(). Low on purpose: one or two can be a genuine blip,
# three in a row has only ever been LinkedIn refusing every load.
NAVIGATION_FAILURE_STREAK_LIMIT = 3


FORM_TIME_LIMIT_SECONDS = 600


class FormTimedOut(RuntimeError):
    pass


@contextlib.contextmanager
def _form_time_limit(seconds: float) -> Iterator[None]:
    """Abandon the application form (FormTimedOut) if it runs past
    `seconds`.

    Real case (2026-10-02): a --loop run sat at 99% CPU for over an hour
    after "Writing a tailored resume and cover letter..." - its own browser
    and Playwright driver were gone, nothing was logged, and it never moved
    on. SIGALRM interrupts Python code wherever it's spinning, so the
    posting fails with a clear error and the run carries on (or stops, if
    the browser is gone - see _browser_is_gone()).

    A no-op where SIGALRM can't be used (Windows, or off the main thread),
    or for seconds <= 0.
    """
    usable = seconds > 0 and hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    if not usable:
        yield
        return

    def expired(signum: int, frame: object) -> None:
        raise FormTimedOut(f"Gave up on the application form after {seconds:.0f}s - it stopped making progress.")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _score_verdict(match: JobMatchScore, should_apply: bool, min_score: int) -> str:
    """One line on why a freshly scored posting is or isn't being applied to,
    for the per-posting progress output in _run_apply_cycle()."""
    if should_apply:
        return f"Scored {match.score} - a fit."
    if match.eligibility == "fail":
        note = f": {match.eligibility_note}" if match.eligibility_note else ""
        return f"Skipped: not eligible{note}"
    if match.score < min_score:
        return f"Skipped: scored {match.score} (below {min_score})."
    return f"Skipped: scored {match.score}, but the model judged it not a fit."


@dataclasses.dataclass(frozen=True)
class CycleResult:
    """What one cycle did, for the caller's summary and --loop decisions.

    fatal: every later cycle would fail the same way (provider down, the
        browser gone, signed out) - stop the loop.
    throttled: LinkedIn was refusing page loads - back off before the next
        search, even if something was applied.
    """

    applied: int
    failed: int
    fatal: bool = False
    throttled: bool = False


def run_cycle(
    ctx: RunContext,
    *,
    adapter: LinkedInAdapter,
    page: Page,
    resume_text: str,
    external_adapter_factory: Callable[[Page], Any],
    form_time_limit_seconds: float = FORM_TIME_LIMIT_SECONDS,
) -> CycleResult:
    """One search -> score -> tailor -> apply pass over a fresh batch of
    postings. Called once for a plain `job-bot run`, or repeatedly for
    `--loop` (back-to-back with no sleep as long as each cycle keeps
    applying to something; only a cycle that applies to nothing pauses
    before the next one) - re-running search() each cycle is what lets loop
    mode pick up postings that appeared after the previous cycle, not just
    the ones visible at process start. Returns a CycleResult for that
    cycle only, not a running total across cycles. Its `throttled` is True
    when LinkedIn refused the search page, or NAVIGATION_FAILURE_STREAK_LIMIT
    job-page loads in a row - --loop then backs off (see
    throttle_backoff_minutes()) even if the cycle applied to something,
    instead of searching again immediately.

    Its `fatal` is True when this cycle stopped early because the
    LLM provider itself was unreachable/misconfigured (Ollama down, Claude
    misconfigured, the browser gone - see pipeline/failures.py's
    classify_failure()) or LinkedIn's session had expired (LinkedInSignedOut), not just
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
    # Local names for the run's shared context, so the body below reads
    # the same as before RunContext existed (docs/architecture.md, step 4);
    # step 5 (pipeline/cycle.py) consumes ctx directly.
    settings, args, provider = ctx.settings, ctx.args, ctx.provider
    resume_store, tracker, rate_limiter = ctx.resume_store, ctx.tracker, ctx.rate_limiter
    blacklist, confirmer, external_confirmer = ctx.blacklist, ctx.confirmer, ctx.external_confirmer
    audit, failure_log, answer_gaps = ctx.audit, ctx.failure_log, ctx.answer_gaps
    min_score, exclude_keywords = ctx.min_score, ctx.exclude_keywords
    experience_levels, max_years_experience = ctx.experience_levels, ctx.max_years_experience
    require_w2, include_external = ctx.require_w2, ctx.include_external

    requested_ids: list[str] = getattr(args, "job_id", None) or []

    skip_policy = SkipPolicy(
        tracker,
        blacklist,
        max_applications_per_company=settings.max_applications_per_company,
        max_apply_attempts=settings.max_apply_attempts,
        exclude_keywords=exclude_keywords,
        requested_ids=requested_ids,
    )
    is_dead_end = skip_policy.is_dead_end

    if requested_ids:
        # `--job-id`: exactly these previously seen postings, instead of a
        # search - to retry one that failed (e.g. after a fix) or dry-run
        # it to watch the form. Only tracked postings: their title, company,
        # and URL come from the tracker, the same source cmd_run's skip
        # logic already trusts.
        postings: list[JobPosting] = []
        for job_id in requested_ids:
            job = tracker.get_job(job_id)
            if job is None:
                print(f"Job {job_id} isn't in the tracker - `job-bot run` hasn't seen it in a search yet.")
                continue
            postings.append(
                JobPosting(job_id=job_id, title=job["title"], company=job["company"], url=job["url"], description="")
            )
        audit.log("requested_jobs", job_ids=requested_ids, found=len(postings))
    else:
        try:
            postings = adapter.search(
                args.keywords,
                args.location,
                max_results=args.search_pool,
                experience_levels=experience_levels,
                include_external=include_external,
                skip=is_dead_end,
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
            # signed_out is what doctor and the dashboard key off (AuditLogger.last_search_signed_out_at())
            # - a structured flag, not a match on the error message's wording.
            signed_out = isinstance(e, LinkedInSignedOut)
            audit.log(
                "search_error", keywords=args.keywords, location=args.location, error=str(e), signed_out=signed_out
            )
            failure_log.log(
                "search_error", keywords=args.keywords, location=args.location, error=str(e), signed_out=signed_out
            )
            print(f"Error searching for postings: {e}")
            if _browser_is_gone(e, page):
                print(BROWSER_GONE_MESSAGE)
                return CycleResult(applied=0, failed=1, fatal=True)
            # A signed-out session fails every search identically until the
            # user runs `job-bot login` - fatal the same way a down provider is.
            # A refused search page (NavigationFailed, after its retries) is
            # LinkedIn rate-limiting the session, the same as a streak of
            # refused job pages - report it as throttled so --loop backs off
            # progressively instead of probing every interval (2026-10-02:
            # searches kept being refused for hours).
            return CycleResult(applied=0, failed=1, fatal=signed_out, throttled=isinstance(e, NavigationFailed))
        audit.log("search", keywords=args.keywords, location=args.location, results=len(postings))

    def should_skip(posting: JobPosting) -> bool:
        """Cheap, deterministic reasons to pass over this posting before
        spending an LLM call on it - SkipPolicy decides, this records it.
        Doesn't cover the cap/--max-apps checks - those stop the whole
        cycle, not just this one posting, so the main loop below handles
        them directly.
        """
        reason = skip_policy.reason(posting)
        if reason is None:
            return False
        if reason is SkipReason.ALREADY_APPLIED:
            if posting.job_id in requested_ids:
                print(f"Skipping {posting.title} at {posting.company}: already applied.")
        elif reason is SkipReason.TOO_MANY_FAILURES:
            audit.log("skip_too_many_failures", job_id=posting.job_id, limit=skip_policy.max_attempts)
        elif reason is SkipReason.BLACKLISTED:
            audit.log("skip_blacklisted", job_id=posting.job_id, company=posting.company)
        elif reason is SkipReason.COMPANY_LIMIT:
            # Printed (unlike the other cheap skips) since it's a deliberate
            # user setting the user will want to see working - in a live
            # run, two roles at one recruiter went out back to back.
            limit = skip_policy.company_limit
            audit.log("skip_company_limit", job_id=posting.job_id, company=posting.company, limit=limit)
            print(f"Skipping {posting.title} at {posting.company}: already applied there (MAX_APPLICATIONS_PER_COMPANY={limit}).")
        elif reason is SkipReason.EXCLUDED_KEYWORD:
            # Not persisted to the tracker (unlike a real score/skip
            # decision), since the exclude list is expected to change
            # between runs and a posting excluded today should still be
            # re-evaluated normally if it's removed later.
            audit.log("skip_excluded_keyword", job_id=posting.job_id, title=posting.title)
        return True

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
                print(f"  Skipped: scored {existing['match_score']} earlier (below {min_score}).")
                return False
            audit.log("reused_score", job_id=posting.job_id, score=existing["match_score"])
            print(f"  Scored {existing['match_score']} earlier - still a fit.")
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
        print(f"  {_score_verdict(match, should_apply, min_score)}")
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

    answers = AnswerService(
        provider=provider,
        resume_text=resume_text,
        resume_store=resume_store,
        tracker=tracker,
        answer_gaps=answer_gaps,
        save_confidence=settings.faq_save_confidence,
        answer_fn=answer_question,
    )
    answer = answers.answer

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
        # Only the form itself is timed - the confirmation prompt above is
        # already answered, however long that took.
        limit = form_time_limit_seconds
        try:
            if posting.easy_apply:
                with _form_time_limit(limit):
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
            return external_adapter_factory(external_page).fill_and_submit(
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
    throttled = False
    consecutive_navigation_failures = 0
    for position, posting in enumerate(postings, start=1):
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

        # One line per posting actually worked on, so a quiet stretch of
        # LLM calls reads as progress rather than a hang - in a live run,
        # skipped postings printed nothing at all and the run was stopped
        # with Ctrl+C because it "seemed stuck".
        print(f"[{position}/{len(postings)}] {posting.title} at {posting.company}")
        try:
            description = adapter.load_description(posting)
            consecutive_navigation_failures = 0
            if not clears_the_bar(posting, description, existing):
                continue
            print("  Writing a tailored resume and cover letter...")
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
            verdict = classify_failure(e, page)
            if verdict.fatal:
                print(verdict.message)
                fatal_error = True
                break
            if verdict.navigation_refused:
                consecutive_navigation_failures += 1
                if consecutive_navigation_failures >= NAVIGATION_FAILURE_STREAK_LIMIT:
                    # Confirmed in data/failed_applications.log: once LinkedIn
                    # starts refusing job-page loads it refuses all of them -
                    # bursts of 18, 16, 15 and 11 consecutive failures within
                    # seconds. Continuing through the rest of the postings is
                    # exactly the request pattern that gets an automated
                    # account restricted, so stop this cycle instead.
                    print(
                        f"LinkedIn refused {consecutive_navigation_failures} job page loads in a row - "
                        "it's likely rate-limiting this session. Stopping this cycle instead of loading "
                        "the rest."
                    )
                    throttled = True
                    break
            continue

        print("  Filling in the application..." if not args.dry_run else "  Filling in the application (dry run)...")
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
            if isinstance(e, FieldsRejected):
                answers.learn_from_rejection(e, posting)
            if not isinstance(e, UnansweredRequiredQuestion | FieldsRejected):
                attempts = tracker.record_apply_failure(posting.job_id)
                if settings.max_apply_attempts > 0 and attempts >= settings.max_apply_attempts:
                    print(
                        f"  Giving up on this posting after {attempts} failed attempts "
                        f"(MAX_APPLY_ATTEMPTS={settings.max_apply_attempts})."
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
            verdict = classify_failure(e, page)
            if verdict.fatal:
                print(verdict.message)
                fatal_error = True
                break
            continue

        if submitted is None:
            continue  # user declined the confirmation prompt

        if submitted:
            # The browser has already clicked Submit for real at this
            # point - both safety records for it must be written before
            # anything that could raise: mark_applied() so a real
            # submission is never lost from the tracker (a duplicate real
            # application on a future run), and record_application() so it
            # always counts toward the daily cap (an audit write failing
            # here - disk full mid-run - used to skip it, letting the next
            # run exceed the cap). record_application()'s own cap check is
            # defense in depth against a second concurrent `job-bot run`
            # process racing this one; if it loses that race, stop cleanly
            # (after logging this submission) rather than crash mid-loop.
            tracker.mark_applied(posting.job_id)
            try:
                rate_limiter.record_application()
                cap_reached_concurrently = False
            except DailyCapReached:
                cap_reached_concurrently = True
            audit.log("applied", job_id=posting.job_id, company=posting.company)
            applied += 1
            print(f"Applied: {posting.title} at {posting.company}")
            if cap_reached_concurrently:
                print("Daily application cap reached (possibly by a concurrent run). Stopping.")
                break
        else:
            audit.log("dry_run_stopped", job_id=posting.job_id)
            print(f"[dry-run] Would apply to {posting.title} at {posting.company}")

    return CycleResult(applied=applied, failed=failed, fatal=fatal_error, throttled=throttled)
