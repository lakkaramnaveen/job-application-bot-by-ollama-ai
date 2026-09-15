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
    "next_button": 'button[aria-label*="next step" i], button[aria-label*="Continue" i]',
    "review_button": 'button[aria-label*="Review" i]',
    "submit_button": 'button[aria-label*="Submit application" i]',
    "dismiss_safety_reminder": 'button[aria-label*="Dismiss" i]',
    "job_cards": "div[data-job-id]",
    "applied_badge": "text=/^\\s*Applied\\s*$/i",
}

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


class LinkedInAdapter(JobBoardAdapter):
    def __init__(self, page: Page):
        self._page = page

    def search(
        self,
        keywords: str,
        location: str,
        max_results: int = 25,
        experience_levels: list[str] | None = None,
    ) -> list[JobPosting]:
        postings: list[JobPosting] = []
        seen_ids: set[str] = set()

        experience_filter = ""
        if experience_levels:
            codes = [EXPERIENCE_LEVEL_CODES[level] for level in experience_levels]
            experience_filter = f"&f_E={quote(','.join(codes), safe=',')}"

        for page_num in range(MAX_SEARCH_PAGES):
            if len(postings) >= max_results:
                break

            start = page_num * RESULTS_PER_PAGE
            url = (
                "https://www.linkedin.com/jobs/search/"
                f"?keywords={quote(keywords, safe='')}"
                f"&location={quote(location, safe='')}"
                f"&start={start}"
                "&f_AL=true"  # Easy Apply filter
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

                if card.locator(SELECTORS["applied_badge"]).count() > 0:
                    logger.info("Skipping job %s: already marked Applied on LinkedIn", job_id)
                    continue

                title_el = card.locator("a").first
                title = (title_el.inner_text() or "").strip()
                href = title_el.get_attribute("href") or ""
                subtitle = card.locator("[class*=subtitle]").first
                company = subtitle.inner_text().strip() if subtitle.count() else ""

                if job_id and title:
                    postings.append(
                        JobPosting(
                            job_id=job_id,
                            title=title,
                            company=company,
                            url=urljoin(LINKEDIN_BASE_URL, href) if href else "",
                            description="",
                        )
                    )
                    if len(postings) >= max_results:
                        break

            if new_ids_on_this_page == 0:
                # Every card here was already seen on an earlier page, which
                # is how LinkedIn behaves when you page past the last result.
                break

        return postings

    def load_description(self, posting: JobPosting) -> str:
        self._goto_with_retry(posting.url)
        self._page.wait_for_load_state("domcontentloaded")
        body = self._page.locator('div[class*="description"]').first
        return body.inner_text() if body.count() else ""

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
        dialog.wait_for(timeout=10000)

        max_steps = 20  # hard cap so a stuck form can't loop forever
        for _ in range(max_steps):
            self._upload_resume_if_requested(dialog, resume_path)
            self._fill_visible_fields(dialog, answer_question, cover_letter_text)

            # A required text/number/textarea field _fill_visible_fields()
            # couldn't fill (answer_question returned "" - the LLM couldn't
            # produce a usable answer, e.g. a genuinely hard compound
            # question) will never let LinkedIn's own client-side validation
            # let a real submission actually go through - and this check
            # has to happen before the submit-button check just below, not
            # only the next/review ones: on a form with everything on one
            # step (many real Easy Apply forms are exactly that), Submit is
            # already reachable right now, and without this check the code
            # would click it anyway - validation blocks the real submission
            # employer-side, but fill_and_submit() has no way to know that;
            # it only knows it clicked something, so it would report success
            # and the caller would record a job as applied that never really
            # went through. On a multi-step form, the same empty field would
            # instead have every one of the max_steps iterations below
            # re-call answer_question for it (its value never changes, so
            # _fill_visible_fields's own "already has a value" skip never
            # kicks in) before giving up with a generic "stuck" message -
            # wasting up to 19 redundant LLM calls on a question already
            # known to be unanswerable. Fail fast instead, naming the
            # question, before either failure mode can happen.
            unanswered = self._first_unanswered_required_text_field_label(dialog)
            if unanswered is not None:
                raise RuntimeError(
                    f"Could not complete the Easy Apply form for job {posting.job_id}: "
                    f"a required question has no answer ({unanswered!r}). The LLM couldn't "
                    "produce a usable answer for it - consider adding it to your FAQ answers "
                    "(job_bot.resume.store.ResumeStore) or trying a different provider/model."
                )

            # Same reasoning, for a required radio group or dropdown left
            # unanswered - which _select_best_option()/_select_best_radio()
            # leave deliberately unanswered rather than guess (see their own
            # docstrings) whenever the LLM's answer doesn't clearly match an
            # option. Before this check existed, that case fell all the way
            # through to the generic "stuck on a step" RuntimeError below
            # with no indication of which question was actually the
            # problem - in practice this was the dominant real-world
            # failure (audit.log showed ~33 generic "stuck" errors against
            # a single specific one, across weeks of real runs), because
            # LinkedIn's own eligibility/sponsorship-style questions are
            # overwhelmingly radio groups, not free text.
            unanswered_choice = self._first_unanswered_required_choice_label(dialog)
            if unanswered_choice is not None:
                raise RuntimeError(
                    f"Could not complete the Easy Apply form for job {posting.job_id}: "
                    f"a required question has no answer ({unanswered_choice!r}). The LLM's "
                    "answer didn't clearly match any option, so this was deliberately left "
                    "unanswered rather than guessed - consider adding it to your FAQ answers "
                    "(job_bot.resume.store.ResumeStore) or trying a different provider/model."
                )

            submit_btn = dialog.locator(SELECTORS["submit_button"])
            if submit_btn.count() > 0:
                if dry_run:
                    return False
                submit_btn.first.click()
                self._dismiss_safety_reminder_if_present()
                return True

            if self._click_if_present(dialog, SELECTORS["review_button"]):
                time.sleep(ACTION_DELAY_SECONDS)
                continue
            if self._click_if_present(dialog, SELECTORS["next_button"]):
                time.sleep(ACTION_DELAY_SECONDS)
                continue

            # No progress button found and no submit button - the form is
            # stuck (e.g. a required field we couldn't resolve). Stop rather
            # than guess.
            break

        raise RuntimeError(
            f"Could not complete the Easy Apply form for job {posting.job_id} "
            "(stuck on a step with no Next/Review/Submit button found)."
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
        """
        for text_input in dialog.locator('input[type="text"], input[type="number"], textarea').all():
            if text_input.get_attribute("required") is None:
                continue
            if (text_input.input_value() or "").strip():
                continue
            return self._label_for(text_input) or "(unlabeled required field)"
        return None

    def _first_unanswered_required_choice_label(self, dialog: Locator) -> str | None:
        """Same purpose as _first_unanswered_required_text_field_label(),
        for a required radio group or <select> left unanswered - detected
        via `required`/`aria-required="true"` on the individual radio
        inputs (fieldset itself has no `required` attribute in HTML) or on
        the select element, confirmed against a real LinkedIn Easy Apply
        form's DOM. Only ever reports a field this attribute actually marks
        as required; a genuinely required field LinkedIn doesn't mark this
        way still falls through to the generic "stuck" message unchanged,
        exactly as before this check existed - this can only add
        diagnostic detail, never new false positives on an optional field.
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
    def _select_best_radio(group: Locator, answer: str) -> None:
        radios = group.locator('input[type="radio"]')
        labels = [LinkedInAdapter._label_for_id(group.page, radios.nth(i)) for i in range(radios.count())]
        idx = LinkedInAdapter._best_match_index(labels, answer)
        if idx is not None:
            radios.nth(idx).check()
        # else: leave unselected rather than guess on a field that may be
        # sponsorship/authorization/eligibility-shaped. If the field is
        # required, LinkedIn's own validation blocks the Next/Review click
        # and fill_and_submit's stuck-form detection surfaces that as a
        # clear error instead of a silently wrong high-stakes answer.

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
        return None

    def _click_if_present(self, dialog: Locator, selector: str) -> bool:
        loc = dialog.locator(selector)
        if loc.count() > 0:
            loc.first.click()
            return True
        return False

    def _dismiss_safety_reminder_if_present(self) -> None:
        try:
            dismiss = self._page.locator(SELECTORS["dismiss_safety_reminder"])
            if dismiss.count() > 0:
                dismiss.first.click(timeout=3000)
        except PlaywrightTimeoutError:
            pass
