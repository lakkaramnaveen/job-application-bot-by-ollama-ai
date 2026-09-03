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
import time
from collections.abc import Callable

from playwright.sync_api import Locator, Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from job_bot.browser.base_adapter import JobBoardAdapter, JobPosting

logger = logging.getLogger(__name__)

SELECTORS = {
    "easy_apply_button": 'button:has-text("Easy Apply")',
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


class LinkedInAdapter(JobBoardAdapter):
    def __init__(self, page: Page):
        self._page = page

    def search(self, keywords: str, location: str, max_results: int = 25) -> list[JobPosting]:
        postings: list[JobPosting] = []
        seen_ids: set[str] = set()

        for page_num in range(MAX_SEARCH_PAGES):
            if len(postings) >= max_results:
                break

            start = page_num * RESULTS_PER_PAGE
            url = (
                "https://www.linkedin.com/jobs/search/"
                f"?keywords={keywords.replace(' ', '%20')}"
                f"&location={location.replace(' ', '%20')}"
                f"&start={start}"
                "&f_AL=true"  # Easy Apply filter
            )
            self._goto_with_retry(url)
            try:
                self._page.wait_for_selector(SELECTORS["job_cards"], timeout=15000)
            except PlaywrightTimeoutError:
                break  # no more results

            cards = self._page.locator(SELECTORS["job_cards"]).all()
            if not cards:
                break

            new_on_this_page = 0
            for card in cards:
                job_id = card.get_attribute("data-job-id") or ""
                if not job_id or job_id in seen_ids:
                    continue
                seen_ids.add(job_id)

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
                        JobPosting(job_id=job_id, title=title, company=company, url=href, description="")
                    )
                    new_on_this_page += 1
                    if len(postings) >= max_results:
                        break

            if new_on_this_page == 0:
                break  # this page had nothing new; further pages won't either

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
        for i in range(file_inputs.count()):
            file_input = file_inputs.nth(i)
            # Skip file inputs that already have a resume selected (LinkedIn
            # often pre-fills with a previously uploaded resume).
            if file_input.get_attribute("data-job-bot-uploaded") == "1":
                continue
            file_input.set_input_files(resume_path)
            file_input.evaluate("el => el.setAttribute('data-job-bot-uploaded', '1')")

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
            selected = select.input_value()
            options = select.locator("option").all_inner_texts()
            if selected and selected.strip() and selected not in ("", "Select an option"):
                continue
            label = self._label_for(select)
            answer = answer_question(label) if label else ""
            self._select_best_option(select, options, answer)

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
        radios.nth(idx).check()

    @staticmethod
    def _select_best_option(select: Locator, options: list[str], answer: str) -> None:
        idx = LinkedInAdapter._best_match_index(options, answer)
        select.select_option(index=idx)

    @staticmethod
    def _label_for_id(page: Page, input_el: Locator) -> str:
        input_id = input_el.get_attribute("id")
        if not input_id:
            return ""
        label = page.locator(f'label[for="{input_id}"]')
        return label.first.inner_text().strip() if label.count() > 0 else ""

    @staticmethod
    def _best_match_index(options: list[str], answer: str) -> int:
        """Pick the option whose text best matches the LLM's answer.
        Falls back to the first non-empty option if nothing matches -
        never guesses on high-stakes fields like sponsorship/authorization
        without at least attempting a real text match first.
        """
        answer_norm = answer.strip().casefold()
        for i, opt in enumerate(options):
            if opt.strip().casefold() == answer_norm:
                return i
        for i, opt in enumerate(options):
            if answer_norm and answer_norm in opt.strip().casefold():
                return i
        for i, opt in enumerate(options):
            if opt.strip():
                return i
        return 0

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
