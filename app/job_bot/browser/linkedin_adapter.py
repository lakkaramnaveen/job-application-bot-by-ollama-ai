"""LinkedIn Easy Apply adapter.

Deterministic, code-controlled navigation (search, click Easy Apply, page
through the multi-step form, submit) - the LLM is only invoked to answer
free-text/radio/select questions the code can't fill from context. This
keeps the flow reliable and reviewable instead of letting a model drive
clicks directly.

LinkedIn's DOM is not publicly documented and changes over time, so the
selectors below are centralized in `SELECTORS` and favor stable accessible
attributes (role, aria-label) over brittle class names. If a run stops
finding buttons/fields it used to find, this is the first place to look and
adjust - `python -m job_bot.cli run --dry-run` is the fastest way to verify
selector changes without submitting anything.
"""

import logging
import re
import time
from collections.abc import Callable
from urllib.parse import quote, urljoin

from playwright.sync_api import Locator, Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from job_bot.browser.base_adapter import JobBoardAdapter, JobPosting

logger = logging.getLogger(__name__)

SELECTORS = {
    # LinkedIn currently renders this control as an <a> (an
    # href="…/apply/?openSDUIApplyFlow=true…" link intercepted by JS to open
    # the modal inline, confirmed against a live job posting), not a
    # <button> - matching only button: here made every real run time out
    # after 30s on every single posting, since the locator matched nothing
    # at all. Match both tags rather than assume which one LinkedIn uses for
    # any given account/job/rollout.
    "easy_apply_button": 'button:has-text("Easy Apply"), a:has-text("Easy Apply")',
    "dialog": 'div[role="dialog"]',
    "dismiss_safety_reminder": 'button[aria-label*="Dismiss" i]',
    "job_cards": "div[data-job-id]",
    "applied_badge": "text=/^\\s*Applied\\s*$/i",
    # A job card shows this small text badge only when the posting supports
    # Easy Apply; its absence is how an external-apply card looks instead.
    "easy_apply_badge": "text=/Easy Apply/i",
    # The external-apply control (confirmed against a live posting): a
    # <button>, not a link - it has no href at all, unlike easy_apply_button
    # above. Clicking it opens the employer's own site in a new tab/popup
    # rather than navigating in place, which is why
    # open_external_application() below has to handle it via
    # page.expect_popup() instead of a plain click + wait.
    "external_apply_button": 'button[aria-label*="on company website" i]',
}

# Priority-ordered fallback chains for the three progress-button roles in
# fill_and_submit()'s loop: LinkedIn's aria-label (confirmed against a live
# posting) is tried first; a looser visible-text match is a fallback for a
# button that turns out not to carry that exact aria-label (an A/B-tested
# rollout, a differently-generated form, ...) - "stuck on a step with no
# Next/Review/Submit button found" was, by a wide margin, this adapter's
# single most common real-world failure, and a button simply not matching
# by aria-label alone is the most plausible cause available without a
# reproduction to confirm against. Checked as separate, ordered locators
# rather than one combined comma-selector + .first: see
# external_apply_adapter.py's _find_submit_button() docstring for why that
# pattern can silently pick the wrong element when more than one candidate
# is present, not necessarily in the order written.
NEXT_BUTTON_SELECTORS = (
    'button[aria-label*="next step" i]',
    'button[aria-label*="Continue" i]',
    'button:has-text("Next")',
    'button:has-text("Continue")',
)
REVIEW_BUTTON_SELECTORS = (
    'button[aria-label*="Review" i]',
    'button:has-text("Review")',
)
SUBMIT_BUTTON_SELECTORS = (
    'button[aria-label*="Submit application" i]',
    'button:has-text("Submit application")',
    'button:has-text("Submit")',
)

# Small, human-scale pauses between UI actions - not an attempt to evade
# detection, just to let LinkedIn's client-side rendering keep up so we don't
# race the DOM. Real users don't click at machine speed either.
ACTION_DELAY_SECONDS = 1.0

RESULTS_PER_PAGE = 25
MAX_SEARCH_PAGES = 8  # hard cap so a huge search can't page forever
NAVIGATION_RETRIES = 2

# Job-card anchors on the search page carry root-relative hrefs
# ("/jobs/view/4012345678/?refId=..."), so every scraped href is resolved
# against this before being stored. A relative URL would be unusable
# everywhere it's later consumed: page.goto() rejects it, and the dashboard
# would render it as a link to the dashboard's own localhost origin.
LINKEDIN_BASE_URL = "https://www.linkedin.com"

# LinkedIn's own `f_E` search filter (seniority level), as documented by its
# search URL params. Filtering here - at the source - is cheaper and more
# reliable than scoring every posting and relying on the LLM to reject
# mismatched seniority after the fact: it costs zero LLM calls for postings
# that never should have been in the pool, and it can't be fooled by a
# posting whose body text doesn't clearly state its own level.
EXPERIENCE_LEVEL_CODES = {
    "internship": "1",
    "entry": "2",
    "associate": "3",
    "mid-senior": "4",
    "director": "5",
    "executive": "6",
}

# LinkedIn's f_TPR ("posted date") search filter, as r<seconds-ago>. Its own
# UI only exposes day/week/month buttons, but confirmed live that an
# arbitrary value like 3 days is still honored server-side, not silently
# rounded up to the nearest UI bucket - see search()'s docstring.
DATE_POSTED_24H = "r86400"
DATE_POSTED_3_DAYS = "r259200"


class UnansweredRequiredQuestion(RuntimeError):
    """A required text/radio/select question fill_and_submit() couldn't
    answer confidently and deliberately left unanswered rather than guess -
    see _first_unanswered_required_text_field_label()/
    _first_unanswered_required_choice_label(). Carries the question as a
    structured `question` attribute (not just embedded in the message) so
    a caller can offer to save an answer for it - see
    safety/answer_gaps.py and cli.py's cmd_review_answers() - closing the
    loop so the same question doesn't silently fail the same way on every
    future posting that asks it.
    """

    def __init__(self, job_id: str, question: str, reason: str):
        self.job_id = job_id
        self.question = question
        super().__init__(
            f"Could not complete the Easy Apply form for job {job_id}: "
            f"a required question has no answer ({question!r}). {reason} - consider adding it "
            "to your FAQ answers (job-bot review-answers) or trying a different provider/model."
        )


class LinkedInAdapter(JobBoardAdapter):
    def __init__(self, page: Page):
        self._page = page

    def search(
        self,
        keywords: str,
        location: str,
        max_results: int = 25,
        experience_levels: list[str] | None = None,
        include_external: bool = False,
    ) -> list[JobPosting]:
        """Searches postings from the last 24 hours first, and only widens
        to the last 3 days if that isn't enough to fill max_results -
        never further back than 3 days, so a run is always looking at
        genuinely fresh postings rather than ones that have likely already
        collected plenty of applicants. Confirmed live that LinkedIn's
        f_TPR filter honors an arbitrary r<seconds> value (not just its own
        UI's day/week/month buttons): r259200 (3 days) returns a real
        subset of r604800 (week)'s results, not the same set relabeled.
        """
        experience_filter = ""
        if experience_levels:
            codes = [EXPERIENCE_LEVEL_CODES[level] for level in experience_levels]
            experience_filter = f"&f_E={quote(','.join(codes), safe=',')}"
        # f_AL=true is LinkedIn's own Easy Apply filter - omitting it (only
        # when the caller actually wants external-apply postings too) is
        # what makes those postings show up in results at all; every result
        # is still classified per-card below rather than assumed, since
        # dropping the filter returns a mix, not just external ones.
        easy_apply_filter = "" if include_external else "&f_AL=true"

        postings = self._search_one_window(
            keywords, location, max_results, experience_filter, easy_apply_filter, include_external, DATE_POSTED_24H
        )
        if len(postings) >= max_results:
            return postings
        # The 3-day window is always a superset of the 24h one, so a fresh
        # search under it supersedes rather than merges with the narrower
        # results above - nothing from the first pass is lost.
        return self._search_one_window(
            keywords,
            location,
            max_results,
            experience_filter,
            easy_apply_filter,
            include_external,
            DATE_POSTED_3_DAYS,
        )

    def _search_one_window(
        self,
        keywords: str,
        location: str,
        max_results: int,
        experience_filter: str,
        easy_apply_filter: str,
        include_external: bool,
        date_filter: str,
    ) -> list[JobPosting]:
        postings: list[JobPosting] = []
        seen_ids: set[str] = set()

        for page_num in range(MAX_SEARCH_PAGES):
            if len(postings) >= max_results:
                break

            start = page_num * RESULTS_PER_PAGE
            url = (
                "https://www.linkedin.com/jobs/search/"
                f"?keywords={quote(keywords, safe='')}"
                f"&location={quote(location, safe='')}"
                f"&start={start}"
                f"&f_TPR={date_filter}"
                f"{easy_apply_filter}"
                f"{experience_filter}"
            )
            self._goto_with_retry(url)
            try:
                self._page.wait_for_selector(SELECTORS["job_cards"], timeout=15000)
            except PlaywrightTimeoutError:
                break  # no more results

            cards = self._page.locator(SELECTORS["job_cards"]).all()
            if not cards:
                break

            # Counts job ids not seen on an earlier page, NOT postings kept -
            # a page can legitimately yield zero postings while still being
            # real progress through the results (e.g. every card on it is
            # already marked Applied). Ending the search on "kept nothing
            # here" would stop at the first such page and never reach the
            # applicable jobs behind it.
            new_ids_on_this_page = 0
            for card in cards:
                job_id = card.get_attribute("data-job-id") or ""
                if not job_id or job_id in seen_ids:
                    continue
                seen_ids.add(job_id)
                new_ids_on_this_page += 1

                posting = self._parse_job_card(card, job_id, include_external)
                if posting is not None:
                    postings.append(posting)
                    if len(postings) >= max_results:
                        break

            if new_ids_on_this_page == 0:
                # Every card here was already seen on an earlier page, which
                # is how LinkedIn behaves when you page past the last result.
                break

        return postings

    def _parse_job_card(self, card: Locator, job_id: str, include_external: bool) -> JobPosting | None:
        """Builds a JobPosting from one search-result card, or returns None
        if the card isn't usable (no title) or is already marked Applied on
        LinkedIn itself - the caller has already deduped job_id against
        seen_ids before calling this.
        """
        if card.locator(SELECTORS["applied_badge"]).count() > 0:
            logger.info("Skipping job %s: already marked Applied on LinkedIn", job_id)
            return None

        title_el = card.locator("a").first
        title = (title_el.inner_text() or "").strip()
        if not title:
            return None
        href = title_el.get_attribute("href") or ""
        subtitle = card.locator("[class*=subtitle]").first
        company = subtitle.inner_text().strip() if subtitle.count() else ""

        # Without include_external, f_AL=true already guarantees every
        # result is Easy Apply server-side - trust that instead of checking
        # the card for a badge that may not be decorated identically
        # everywhere. Only a mixed page (include_external=True dropped that
        # filter) needs the per-card check to actually distinguish the two.
        easy_apply = True
        if include_external:
            easy_apply = card.locator(SELECTORS["easy_apply_badge"]).count() > 0

        return JobPosting(
            job_id=job_id,
            title=title,
            company=company,
            url=urljoin(LINKEDIN_BASE_URL, href) if href else "",
            description="",
            easy_apply=easy_apply,
        )

    def load_description(self, posting: JobPosting) -> str:
        self._goto_with_retry(posting.url)
        self._page.wait_for_load_state("domcontentloaded")
        body = self._page.locator('div[class*="description"]').first
        return body.inner_text() if body.count() else ""

    def open_external_application(self, posting: JobPosting) -> Page | None:
        """For an easy_apply=False posting: open its external application
        on the employer's own site and return the resulting Page, or None
        if the external-apply control isn't there (the posting may have
        turned out to be Easy Apply after all, or stopped accepting
        applications since it was found).

        The control is a <button> with no href (confirmed against a live
        posting) that opens the destination in a new tab/popup rather than
        navigating in place - page.expect_popup() is how Playwright catches
        that, not a plain click + wait_for_load_state on self._page.
        Callers own the returned page's lifecycle (close it when done);
        this adapter has no further involvement once it's returned, since
        everything past this point happens on a site this project doesn't
        control the structure of - see external_apply_adapter.py.
        """
        self._goto_with_retry(posting.url)
        self._page.wait_for_load_state("domcontentloaded")
        button = self._page.locator(SELECTORS["external_apply_button"])
        if button.count() == 0:
            return None
        try:
            with self._page.expect_popup(timeout=15000) as popup_info:
                button.first.click()
        except PlaywrightTimeoutError as e:
            # Same "diagnostic, not a bare Playwright timeout" treatment
            # as fill_and_submit()'s dialog wait: the click can fail to
            # open a popup at all if the destination site navigates in
            # place instead of via window.open() (confirmed live
            # elsewhere that LinkedIn's own control does use window.open,
            # but nothing guarantees every employer's redirect does too),
            # or its own loading was just unusually slow this time.
            raise RuntimeError(
                f"Clicking the external-apply button for job {posting.job_id} didn't open a new "
                f"tab/popup within 15s (page was at {self._page.url!r}, titled "
                f"{self._page.title()!r}) - the destination site may navigate in place instead of "
                "opening a popup, or its own loading was unusually slow this time."
            ) from e
        external_page = popup_info.value
        external_page.wait_for_load_state("domcontentloaded")
        return external_page

    def fill_and_submit(
        self,
        posting: JobPosting,
        *,
        answer_question: Callable[[str], str],
        resume_path: str | None,
        cover_letter_text: str | None,
        dry_run: bool,
    ) -> bool:
        self._goto_with_retry(posting.url)
        self._page.wait_for_load_state("domcontentloaded")
        self._page.locator(SELECTORS["easy_apply_button"]).first.click()
        time.sleep(ACTION_DELAY_SECONDS)

        dialog = self._page.locator(SELECTORS["dialog"]).first
        try:
            dialog.wait_for(timeout=10000)
        except PlaywrightTimeoutError as e:
            # Diagnostic, not just a bare Playwright timeout: confirmed
            # live in a real run (twice in one cycle), Playwright's own
            # message here names only the selector, never what the page
            # actually showed instead - giving no lead on whether the
            # click silently opened nothing, LinkedIn presented a re-
            # authentication checkpoint instead of the dialog, the posting
            # stopped accepting applications since it was found, or
            # client-side rendering was just unusually slow this once. The
            # current URL/title at least narrows that down the next time
            # this shows up in failed_applications.log.
            raise RuntimeError(
                f"Easy Apply dialog never appeared for job {posting.job_id} within 10s of "
                f"clicking Easy Apply (page was at {self._page.url!r}, titled "
                f"{self._page.title()!r}) - the posting may require re-authentication, have "
                "stopped accepting applications since it was found, or LinkedIn's own "
                "rendering was unusually slow this time."
            ) from e

        max_steps = 20  # hard cap so a stuck form can't loop forever
        for _ in range(max_steps):
            self._upload_resume_if_requested(dialog, resume_path)
            self._fill_visible_fields(dialog, answer_question, cover_letter_text)
            self._raise_if_unanswered_required_field(dialog, posting)

            submit_btn = self._find_button(dialog, SUBMIT_BUTTON_SELECTORS)
            if submit_btn is not None:
                if dry_run:
                    return False
                submit_btn.click()
                self._dismiss_safety_reminder_if_present()
                return True

            review_btn = self._find_button(dialog, REVIEW_BUTTON_SELECTORS)
            if review_btn is not None:
                review_btn.click()
                time.sleep(ACTION_DELAY_SECONDS)
                continue
            next_btn = self._find_button(dialog, NEXT_BUTTON_SELECTORS)
            if next_btn is not None:
                next_btn.click()
                time.sleep(ACTION_DELAY_SECONDS)
                continue

            # No progress button found and no submit button - the form is
            # stuck (e.g. a required field we couldn't resolve). Stop rather
            # than guess.
            break

        # Diagnostic, not just "stuck": lists whatever button text actually
        # was on screen at the point of giving up. This ends up in
        # failed_applications.log via cli.py's apply_error handling
        # (audit.log(..., error=str(e))) - without it, every occurrence of
        # this error looked identical regardless of cause, giving no lead on
        # whether NEXT_BUTTON_SELECTORS/REVIEW_BUTTON_SELECTORS/
        # SUBMIT_BUTTON_SELECTORS above are missing a real button label or
        # the form is stuck for some other reason (e.g. a field type this
        # adapter doesn't fill at all).
        visible_button_texts = [t.strip() for t in dialog.locator("button:visible").all_inner_texts()]
        buttons_seen = ", ".join(repr(t) for t in visible_button_texts if t) or "none"
        raise RuntimeError(
            f"Could not complete the Easy Apply form for job {posting.job_id} (stuck on a step with "
            f"no Next/Review/Submit button found - buttons visible on this step: {buttons_seen})."
        )

    def _raise_if_unanswered_required_field(self, dialog: Locator, posting: JobPosting) -> None:
        """Fails fast, naming the specific question, if a required field
        was left empty rather than let fill_and_submit()'s loop either
        submit an incomplete form or spin uselessly until it gives up.

        Checked before the submit/progress-button checks in that loop, not
        after, for two reasons: on a form with everything on one step
        (many real Easy Apply forms are exactly that), Submit is already
        reachable the moment this runs - without this check the code
        would click it anyway, since LinkedIn's own client-side validation
        blocks the submission employer-side but fill_and_submit() has no
        way to know that; it would report success and the caller would
        wrongly record the job as applied. On a multi-step form, the same
        empty field would otherwise have every one of max_steps's
        iterations re-ask answer_question() for it (its value never
        changes, so _fill_visible_fields()'s "already has a value" skip
        never kicks in) before eventually giving up with a generic
        "stuck" message - wasting up to 19 redundant LLM calls on a
        question already known to be unanswerable.

        Covers a plain text/number/textarea field
        (_first_unanswered_required_text_field_label()), a required radio
        group/dropdown/checkbox deliberately left unanswered rather than
        guessed (_first_unanswered_required_choice_label() - see its own
        and _select_best_radio()'s "never guess" docstrings), and a
        required file upload _upload_resume_if_requested() didn't fill
        (_first_unanswered_required_file_field_label() - most commonly its
        "ambiguous file field" case, deliberately not guessed either). The
        radio/select case is the dominant one in practice: audit.log
        showed ~33 generic "stuck" errors against a single specific one
        before this check existed, since LinkedIn's own eligibility/
        sponsorship-style questions are overwhelmingly radio groups, not
        free text.
        """
        unanswered = self._first_unanswered_required_text_field_label(dialog)
        if unanswered is not None:
            raise UnansweredRequiredQuestion(
                posting.job_id, unanswered, "The LLM couldn't produce a usable answer for it"
            )

        unanswered_choice = self._first_unanswered_required_choice_label(dialog)
        if unanswered_choice is not None:
            raise UnansweredRequiredQuestion(
                posting.job_id,
                unanswered_choice,
                "The LLM's answer didn't clearly match any option, so this was "
                "deliberately left unanswered rather than guessed",
            )

        unanswered_file = self._first_unanswered_required_file_field_label(dialog)
        if unanswered_file is not None:
            raise UnansweredRequiredQuestion(
                posting.job_id,
                unanswered_file,
                "No file was uploaded for it - either the resume path isn't configured, or "
                "multiple file fields on this step made it impossible to confidently identify "
                "which one to use (see _upload_resume_if_requested)",
            )

    def _goto_with_retry(self, url: str) -> None:
        """Navigate with a couple of retries - LinkedIn's client-side
        rendering occasionally times out on a cold load with no real error
        in the page itself, and a bare retry resolves it almost every time.
        """
        last_error: Exception | None = None
        for attempt in range(NAVIGATION_RETRIES + 1):
            try:
                self._page.goto(url, timeout=20000)
                return
            except PlaywrightTimeoutError as e:
                last_error = e
                logger.warning("Navigation to %s timed out (attempt %d), retrying", url, attempt + 1)
                time.sleep(ACTION_DELAY_SECONDS)
        raise RuntimeError(f"Failed to load {url} after {NAVIGATION_RETRIES + 1} attempts") from last_error

    def _upload_resume_if_requested(self, dialog: Locator, resume_path: str | None) -> None:
        if not resume_path:
            return
        # LinkedIn's "Resume" step usually shows a card list of previously
        # uploaded resumes (one radio-toggle per card, id prefixed
        # "jobsDocumentCardToggle") with one already selected - confirmed
        # against a live job posting's DOM. A hidden input[type="file"]
        # still exists on this step regardless (behind the "Upload resume"
        # button next to the cards), so without this check the code below
        # would call set_input_files() on it every single application
        # regardless of whether anything actually needed to change -
        # silently piling up duplicate copies of the same document in the
        # user's LinkedIn resume library (5 identical entries were found
        # there from repeated runs before this fix).
        if dialog.locator('input[id^="jobsDocumentCardToggle"]:checked').count() > 0:
            return
        file_inputs = dialog.locator('input[type="file"]')
        # Skip file inputs that already have a resume selected (LinkedIn
        # often pre-fills with a previously uploaded resume).
        pending = [
            file_inputs.nth(i)
            for i in range(file_inputs.count())
            if file_inputs.nth(i).get_attribute("data-job-bot-uploaded") != "1"
        ]
        for file_input in pending:
            label = self._label_for(file_input)
            if len(pending) > 1:
                # Multiple file fields on this step - only upload into one we
                # can positively identify as the resume field. An unlabeled
                # field, or one whose label isn't recognized either way, is
                # left alone rather than guessed: with several file inputs
                # present, defaulting to "upload unless denylisted" risks
                # silently landing the resume in a cover-letter/portfolio/
                # transcript field whose label just isn't on the denylist.
                if not (label and self._looks_like_resume_file_field(label)):
                    logger.warning(
                        "Skipping ambiguous file field (label=%r): multiple file inputs are "
                        "present on this step and this one isn't confidently a resume field.",
                        label,
                    )
                    continue
            elif label and self._looks_like_non_resume_file_field(label):
                # The single file field on this step is explicitly labeled
                # for something else - never guess a resume upload into a
                # field meant for a different document.
                continue
            file_input.set_input_files(resume_path)
            file_input.evaluate("el => el.setAttribute('data-job-bot-uploaded', '1')")

    _RESUME_FILE_LABEL_MARKERS = ("resume", "cv")
    _NON_RESUME_FILE_LABEL_MARKERS = (
        "cover letter",
        "portfolio",
        "writing sample",
        "transcript",
        "certificate",
        "license",
    )

    @classmethod
    def _looks_like_resume_file_field(cls, label: str) -> bool:
        normalized = label.casefold()
        return any(marker in normalized for marker in cls._RESUME_FILE_LABEL_MARKERS)

    @classmethod
    def _looks_like_non_resume_file_field(cls, label: str) -> bool:
        normalized = label.casefold()
        return any(marker in normalized for marker in cls._NON_RESUME_FILE_LABEL_MARKERS)

    def _fill_visible_fields(
        self,
        dialog: Locator,
        answer_question: Callable[[str], str],
        cover_letter_text: str | None = None,
    ) -> None:
        for text_input in dialog.locator('input[type="text"], input[type="number"], textarea').all():
            if (text_input.input_value() or "").strip():
                continue
            label = self._label_for(text_input)
            if cover_letter_text and label and self._looks_like_cover_letter_field(label):
                text_input.fill(cover_letter_text)
                continue
            answer = answer_question(label) if label else ""
            if answer:
                if (text_input.get_attribute("type") or "").casefold() == "number":
                    answer = self._numeric_value(answer) or answer
                text_input.fill(answer)

        for group in dialog.locator("fieldset").all():
            radios = group.locator('input[type="radio"]')
            if radios.count() == 0:
                continue
            if any(radios.nth(i).is_checked() for i in range(radios.count())):
                continue
            label = self._label_for(group)
            answer = answer_question(label) if label else ""
            self._select_best_radio(group, answer)

        for select in dialog.locator("select").all():
            # A select with no blank placeholder option has its first real
            # option auto-selected by the browser with nothing chosen by
            # anyone - indistinguishable from a real answer by value/text
            # alone (there may be no "Select an option" placeholder to
            # compare against). selectedIndex == 0 covers both that case and
            # an explicit blank placeholder, and is always safe to treat as
            # "unanswered": _select_best_option() below never guesses either,
            # so at worst this re-confirms whatever was already selected.
            selected_index = select.evaluate("el => el.selectedIndex")
            if selected_index > 0:
                continue
            options = select.locator("option").all_inner_texts()
            label = self._label_for(select)
            answer = answer_question(label) if label else ""
            self._select_best_option(select, options, answer)

    def _first_unanswered_required_text_field_label(self, dialog: Locator) -> str | None:
        """A required text/number/textarea field _fill_visible_fields()
        left empty - either answer_question() returned "" for it, or it has
        no id and so no _label_for() could resolve at all. Radio/select
        fields are deliberately not checked here: leaving those unselected
        on a non-match is already the intended, documented behavior (see
        _select_best_radio()'s docstring) - this only targets the case
        that's cheap and reliable to detect (a plain empty required value)
        and where the caller can say something more specific than "stuck".

        Checks required via _is_marked_required() (both the `required`
        attribute and `aria-required="true"`), not a bare `required`
        attribute check - real bug this guards against: a text field
        marked required only via aria-required="true" was invisible to
        this check entirely, so it was never flagged as unanswered and
        fill_and_submit()'s loop went on to find and click Submit anyway,
        the exact "submit an incomplete form and wrongly record it as
        applied" failure mode this whole check exists to prevent (see
        _raise_if_unanswered_required_field()'s docstring) - confirmed
        live before this fix.
        """
        for text_input in dialog.locator('input[type="text"], input[type="number"], textarea').all():
            if not self._is_marked_required(text_input):
                continue
            if (text_input.input_value() or "").strip():
                continue
            return self._label_for(text_input) or "(unlabeled required field)"
        return None

    def _first_unanswered_required_choice_label(self, dialog: Locator) -> str | None:
        """Same purpose as _first_unanswered_required_text_field_label(),
        for a required radio group, <select>, or standalone checkbox left
        unanswered - detected via `required`/`aria-required="true"` on the
        individual radio inputs (fieldset itself has no `required`
        attribute in HTML), the select element, or the checkbox itself,
        confirmed against a real LinkedIn Easy Apply form's DOM. Only ever
        reports a field this attribute actually marks as required; a
        genuinely required field LinkedIn doesn't mark this way still
        falls through to the generic "stuck" message unchanged, exactly as
        before this check existed - this can only add diagnostic detail,
        never new false positives on an optional field.
        """
        for group in dialog.locator("fieldset").all():
            radios = group.locator('input[type="radio"]')
            if radios.count() == 0:
                continue
            if any(radios.nth(i).is_checked() for i in range(radios.count())):
                continue
            if not self._is_marked_required(radios.first):
                continue
            return self._label_for(group) or "(unlabeled required choice)"

        for select in dialog.locator("select").all():
            if not self._is_marked_required(select):
                continue
            selected_index = select.evaluate("el => el.selectedIndex")
            if selected_index > 0:
                continue
            return self._label_for(select) or "(unlabeled required dropdown)"

        # A standalone required checkbox (a consent/agreement box, most
        # commonly) - never auto-checked on the user's behalf, same
        # "never guess" stance as a radio/select non-match (see
        # _select_best_radio()'s docstring), so left unchecked here is
        # correct. Real bug this guards against: nothing else in this
        # adapter mentions checkboxes at all, unlike
        # external_apply_adapter.py's equivalent check, which explicitly
        # handles this case - a required checkbox was invisible to every
        # check here, so fill_and_submit() went on to find and click
        # Submit with it still unchecked. Confirmed live before this fix.
        for checkbox in dialog.locator('input[type="checkbox"]').all():
            if not self._is_marked_required(checkbox):
                continue
            if checkbox.is_checked():
                continue
            return self._label_for(checkbox) or "(unlabeled required checkbox)"

        return None

    def _first_unanswered_required_file_field_label(self, dialog: Locator) -> str | None:
        """A required file input _upload_resume_if_requested() left empty
        - most commonly its "ambiguous file field" case (multiple file
        inputs on this step, none confidently identifiable as the resume
        field - deliberately not guessed, see that method's docstring), or
        resume_path itself not being configured for this run. Real bug
        this guards against: nothing checked required file inputs at all,
        so this case fell through to fill_and_submit() finding and
        clicking Submit anyway with the field still empty - confirmed live
        before this fix.
        """
        for file_input in dialog.locator('input[type="file"]').all():
            if not self._is_marked_required(file_input):
                continue
            has_file = file_input.evaluate("el => el.files && el.files.length > 0")
            if has_file:
                continue
            return self._label_for(file_input) or "(unlabeled required file upload)"
        return None

    @staticmethod
    def _is_marked_required(el: Locator) -> bool:
        return el.get_attribute("required") is not None or el.get_attribute("aria-required") == "true"

    @staticmethod
    def _looks_like_cover_letter_field(label: str) -> bool:
        return "cover letter" in label.casefold()

    @staticmethod
    def _label_for(el: Locator) -> str:
        try:
            el_id = el.get_attribute("id")
            if el_id:
                label = el.page.locator(f'label[for="{el_id}"]')
                if label.count() > 0:
                    return label.first.inner_text().strip()
            legend = el.locator("legend")
            if legend.count() > 0:
                return legend.first.inner_text().strip()
            aria = el.get_attribute("aria-label")
            if aria:
                return aria.strip()
        except PlaywrightTimeoutError:
            pass
        return ""

    @staticmethod
    def _numeric_value(answer: str) -> str | None:
        """Extracts a plain number from a free-text LLM answer (e.g. "5+
        years", "5 years of experience") for filling an
        input[type="number"] field. Playwright's .fill() sets a number
        input's value the same way a real browser would - the input
        rejects anything that isn't a valid number and silently resets to
        empty, so a field like "years of experience" was ending up blank
        even though the LLM's answer was substantively correct (qwen
        answers these with a trailing "+"/"years" qualifier rather than a
        bare digit; see _best_match_index()'s docstring for the same
        pattern on radio/select fields). Returns the first digit sequence
        found (with an optional decimal part), or None if the answer has
        no digits at all - the caller then falls back to the raw answer,
        which will be rejected the same way but at least isn't silently
        swapped for something the LLM never said.
        """
        match = re.search(r"\d+(?:\.\d+)?", answer)
        return match.group() if match else None

    @staticmethod
    def _select_best_radio(group: Locator, answer: str) -> None:
        radios = group.locator('input[type="radio"]')
        labels = [LinkedInAdapter._label_for_id(group.page, radios.nth(i)) for i in range(radios.count())]
        idx = LinkedInAdapter._best_match_index(labels, answer)
        if idx is None:
            # Leave unselected rather than guess on a field that may be
            # sponsorship/authorization/eligibility-shaped. If the field is
            # required, LinkedIn's own validation blocks the Next/Review click
            # and fill_and_submit's stuck-form detection surfaces that as a
            # clear error instead of a silently wrong high-stakes answer.
            return
        radio = radios.nth(idx)
        radio_id = radio.get_attribute("id")
        label = group.page.locator(f'label[for="{radio_id}"]') if radio_id else None
        if label is not None and label.count() > 0:
            # LinkedIn commonly styles these as custom pill/card radios with
            # the native <input> visually hidden behind its own <label> -
            # checking the input directly then fails Playwright's
            # actionability check ("label intercepts pointer events"),
            # observed live timing out after ~30s on a real application.
            # Click the label instead, exactly like a real user does.
            label.first.click()
        else:
            radio.check()

    @staticmethod
    def _select_best_option(select: Locator, options: list[str], answer: str) -> None:
        idx = LinkedInAdapter._best_match_index(options, answer)
        if idx is not None:
            select.select_option(index=idx)

    @staticmethod
    def _label_for_id(page: Page, input_el: Locator) -> str:
        input_id = input_el.get_attribute("id")
        if not input_id:
            return ""
        label = page.locator(f'label[for="{input_id}"]')
        return label.first.inner_text().strip() if label.count() > 0 else ""

    @staticmethod
    def _best_match_index(options: list[str], answer: str) -> int | None:
        """Pick the option whose text best matches the LLM's answer, or
        None if nothing matches at all. Never falls back to an arbitrary
        option (e.g. "the first one") - on a field like work
        authorization/sponsorship, silently selecting a guessed answer is
        worse than leaving it blank and letting the caller detect a stuck
        form. See _select_best_radio()/_select_best_option().
        """
        answer_norm = answer.strip().casefold()
        if not answer_norm:
            return None
        for i, opt in enumerate(options):
            if opt.strip().casefold() == answer_norm:
                return i
        # Fallback: the answer as a whole word within the option's text
        # (e.g. answer "yes" matching option "Yes, I am authorized"). Plain
        # substring containment here would also match e.g. answer "no"
        # against option "None" or "Notice period" - wrong option, silently
        # selected, on a field this sensitive - so require that the match
        # isn't butted up against surrounding word characters.
        #
        # These lookarounds rather than \b: \b is defined relative to the
        # adjacent character in the *pattern* too, so an answer that starts
        # or ends with a non-word character ("5+", "C++", "100%") could never
        # match anything - r"\b5\+\b" doesn't match "5+ years", because the
        # position after "+" sits between two non-word characters. That left
        # such fields blank and stalled the form. (?<!\w)/(?!\w) still reject
        # "no" inside "None" while matching "5+" in "5+ years".
        pattern = re.compile(rf"(?<!\w){re.escape(answer_norm)}(?!\w)")
        for i, opt in enumerate(options):
            if pattern.search(opt.strip().casefold()):
                return i
        # Second fallback, the reverse direction: a short option (most
        # commonly a plain "Yes"/"No" radio pair) as a whole word within a
        # longer answer. qa_answerer.py's own prompt asks for a direct,
        # sometimes explanatory answer ("No, I am currently located in..."),
        # not a bare "yes"/"no" - the check above alone can then never
        # match ANY yes/no-shaped question, no matter how clearly the
        # answer states its position, since it only ever looks for the
        # (long) answer inside the (short) option. Confirmed live: this
        # exact case - a real, reasonable answer explicitly starting "No,
        # ..." - left blank and recorded as the same unanswerable gap 27
        # times in one real user's answer_gaps.json, despite the model
        # clearly knowing and stating the answer every time.
        for i, opt in enumerate(options):
            opt_norm = opt.strip().casefold()
            if not opt_norm:
                continue
            opt_pattern = re.compile(rf"(?<!\w){re.escape(opt_norm)}(?!\w)")
            if opt_pattern.search(answer_norm):
                return i
        return None

    @staticmethod
    def _find_button(dialog: Locator, selectors: tuple[str, ...]) -> Locator | None:
        """Tries each selector in order, returning the first that matches
        anything - never a combined comma-selector + .first, which matches
        in DOCUMENT order across every alternative rather than the order
        written (see NEXT_BUTTON_SELECTORS/REVIEW_BUTTON_SELECTORS/
        SUBMIT_BUTTON_SELECTORS' module-level comment).
        """
        for selector in selectors:
            candidate = dialog.locator(selector)
            if candidate.count() > 0:
                return candidate.first
        return None

    def _dismiss_safety_reminder_if_present(self) -> None:
        try:
            dismiss = self._page.locator(SELECTORS["dismiss_safety_reminder"])
            if dismiss.count() > 0:
                dismiss.first.click(timeout=3000)
        except PlaywrightTimeoutError:
            pass
