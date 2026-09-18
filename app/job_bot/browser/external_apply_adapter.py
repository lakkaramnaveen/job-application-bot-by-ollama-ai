"""EXPERIMENTAL: best-effort form filler for job applications handled
entirely off LinkedIn, on the employer's own career site - opened via
LinkedInAdapter.open_external_application(). Not on by default; see
Settings.enable_external_apply and README.md's "Applying on company
websites (experimental)" section.

Unlike linkedin_adapter.py, there is no single site's DOM to learn and
adjust to here - every employer's career site (Greenhouse, Workday, Lever,
a fully custom form, ...) is structured differently, so this will get forms
wrong sometimes. The goal is to fail safely when that happens (leave a
clear, specific reason in the failure log - see cli.py's cmd_run - rather
than guess or submit something incomplete), not to never fail at all.

What this never does, regardless of what a form asks for or what
answer_question might be willing to produce: solve or bypass a CAPTCHA,
create an account or enter a password, or fill a field asking for a
government ID/SSN, passport, or financial account number. Each of those is
detected and stops the application cleanly - see CaptchaEncountered,
AccountCreationRequired, and SENSITIVE_FIELD_MARKERS below.
"""

import re
import time
from collections.abc import Callable

from playwright.sync_api import Locator, Page

ACTION_DELAY_SECONDS = 1.0
MAX_STEPS = 10

# Label/placeholder substrings that mean "never fill this field, no matter
# what answer_question might return for it" - these ask for exactly the
# categories this project will never enter into a form. A required field
# matching this list is therefore always left blank, which the required-
# field check below turns into a clear, specific error rather than either
# silently skipping it (risking an incomplete submission the caller
# wouldn't know about) or guessing.
SENSITIVE_FIELD_MARKERS = (
    "social security",
    "ssn",
    "passport number",
    "driver's license number",
    "driver license number",
    "national id",
    "bank account",
    "routing number",
    "credit card",
    "debit card",
    "cvv",
    "tax id",
    "ein",
)

# Structural, low-false-positive DOM markers for the major CAPTCHA
# providers - deliberately not a body-text scan (e.g. matching on the word
# "captcha" would also match legitimate page copy like "no captcha
# required"), since a false negative here is the failure mode that actually
# matters to avoid.
_CAPTCHA_SELECTORS = (
    'iframe[src*="recaptcha" i]',
    'iframe[src*="hcaptcha" i]',
    'iframe[title*="captcha" i]',
    "div.g-recaptcha",
    "div.h-captcha",
    'iframe[src*="challenges.cloudflare.com" i]',
)

_RESUME_FILE_LABEL_MARKERS = ("resume", "cv")
_NON_RESUME_FILE_LABEL_MARKERS = ("cover letter", "portfolio", "writing sample", "transcript")


class CaptchaEncountered(RuntimeError):
    """Raised when the external site presents a CAPTCHA. job-bot never
    attempts to solve or bypass one - this stops the application cleanly.
    """


class AccountCreationRequired(RuntimeError):
    """Raised when the external site has a password field anywhere on the
    page - job-bot never creates accounts or enters passwords, even ones
    the user would choose themselves.
    """


class ExternalApplyAdapter:
    """Fills and submits one external application on an already-open Page
    (see LinkedInAdapter.open_external_application()). Reuses the same
    never-guess philosophy as LinkedInAdapter (leave a field blank rather
    than guess, fail on an unanswered required field instead of clicking
    Submit anyway - see linkedin_adapter.py's
    _first_unanswered_required_text_field_label() for the LinkedIn-specific
    version of the same idea) but with much looser, best-effort selectors,
    since there's no single site's DOM to have learned.

    Deliberately does not attempt radio buttons or checkboxes at all -
    consent/agreement checkboxes in particular are not something to check on
    a user's behalf without their actual review. A required one left
    unchecked (or a required radio group left unselected) is caught by the
    same required-field check as an empty text field, and stops the
    application rather than guessing or silently submitting without it.
    """

    def __init__(self, page: Page):
        self._page = page

    def fill_and_submit(
        self,
        *,
        answer_question: Callable[[str], str],
        resume_path: str | None,
        cover_letter_text: str | None,
        dry_run: bool,
    ) -> bool:
        self._raise_if_unsafe_to_proceed()

        for _ in range(MAX_STEPS):
            self._upload_resume_if_requested(resume_path)
            self._fill_visible_fields(answer_question, cover_letter_text)
            self._raise_if_unsafe_to_proceed()

            unanswered = self._first_unanswered_required_field_label()
            if unanswered is not None:
                raise RuntimeError(
                    "Could not complete the external application: a required question has no "
                    f"answer ({unanswered!r}). Either the LLM couldn't produce a usable answer for "
                    "it, it's a checkbox/radio group (never auto-filled here), or it's a field "
                    "job-bot never fills automatically (see SENSITIVE_FIELD_MARKERS)."
                )

            submit_btn = self._find_submit_button()
            if submit_btn is not None:
                if dry_run:
                    return False
                submit_btn.click()
                return True

            if self._click_progress_button():
                time.sleep(ACTION_DELAY_SECONDS)
                self._raise_if_unsafe_to_proceed()
                continue

            # No progress button found and no submit button - stuck rather
            # than guess, same philosophy as linkedin_adapter.py.
            break

        raise RuntimeError(
            "Could not complete the external application (stuck on a step with no "
            "Next/Continue/Submit button found)."
        )

    def _raise_if_unsafe_to_proceed(self) -> None:
        for selector in _CAPTCHA_SELECTORS:
            if self._page.locator(selector).count() > 0:
                raise CaptchaEncountered(
                    "This application requires solving a CAPTCHA - job-bot never attempts this."
                )
        if self._page.locator('input[type="password"]').count() > 0:
            raise AccountCreationRequired(
                "This application requires creating an account/password - job-bot never does this."
            )

    def _upload_resume_if_requested(self, resume_path: str | None) -> None:
        if not resume_path:
            return
        file_inputs = self._page.locator('input[type="file"]:visible')
        pending = [
            file_inputs.nth(i)
            for i in range(file_inputs.count())
            if file_inputs.nth(i).get_attribute("data-job-bot-uploaded") != "1"
        ]
        for file_input in pending:
            label = self._label_for(file_input).casefold()
            if len(pending) > 1:
                if not any(marker in label for marker in _RESUME_FILE_LABEL_MARKERS):
                    continue
            elif any(marker in label for marker in _NON_RESUME_FILE_LABEL_MARKERS):
                continue
            file_input.set_input_files(resume_path)
            file_input.evaluate("el => el.setAttribute('data-job-bot-uploaded', '1')")

    def _fill_visible_fields(
        self, answer_question: Callable[[str], str], cover_letter_text: str | None
    ) -> None:
        # :visible everywhere in this file is load-bearing, not cosmetic: a
        # multi-step form's later steps commonly already exist in the DOM,
        # just CSS-hidden until reached - .fill()/.click() on a hidden
        # element blocks waiting for it to become actionable (it never
        # does) instead of raising, which would hang the run rather than
        # fail cleanly. Only ever touch what's actually on screen right now.
        text_selector = (
            'input[type="text"]:visible, input[type="email"]:visible, input[type="tel"]:visible, '
            'input[type="number"]:visible, input:not([type]):visible, textarea:visible'
        )
        for text_input in self._page.locator(text_selector).all():
            if (text_input.input_value() or "").strip():
                continue
            label = self._label_for(text_input)
            normalized = label.casefold()
            if any(marker in normalized for marker in SENSITIVE_FIELD_MARKERS):
                continue  # never even ask - see SENSITIVE_FIELD_MARKERS
            if cover_letter_text and "cover letter" in normalized:
                text_input.fill(cover_letter_text)
                continue
            answer = answer_question(label) if label else ""
            if answer:
                if (text_input.get_attribute("type") or "").casefold() == "number":
                    answer = self._numeric_value(answer) or answer
                text_input.fill(answer)

        for select in self._page.locator("select:visible").all():
            if select.evaluate("el => el.selectedIndex") > 0:
                continue
            label = self._label_for(select)
            if any(marker in label.casefold() for marker in SENSITIVE_FIELD_MARKERS):
                continue
            options = select.locator("option").all_inner_texts()
            answer = answer_question(label) if label else ""
            idx = self._best_match_index(options, answer)
            if idx is not None:
                select.select_option(index=idx)

    def _first_unanswered_required_field_label(self) -> str | None:
        required_selector = (
            "input[required]:visible, textarea[required]:visible, select[required]:visible, "
            'input[aria-required="true"]:visible, textarea[aria-required="true"]:visible, '
            'select[aria-required="true"]:visible'
        )
        for field in self._page.locator(required_selector).all():
            tag = field.evaluate("el => el.tagName")
            field_type = (field.get_attribute("type") or "").casefold()
            if tag == "SELECT":
                if field.evaluate("el => el.selectedIndex") > 0:
                    continue
            elif field_type == "radio":
                # A real bug this guards against: some sites mark every
                # radio in a group `required` (redundant but common - e.g.
                # framework-generated accessibility markup), not just one.
                # Checking only *this* input's .checked would then report
                # every unchecked sibling as its own unanswered required
                # field, even when the group is genuinely answered by
                # another option already checked (a page-supplied default,
                # e.g. "Willing to relocate? Yes / No" defaulting to "No") -
                # confirmed live before this fix. getElementsByName (a
                # native DOM lookup, not a CSS-attribute-value selector) is
                # used instead of querying by [name="..."] so a name
                # containing a quote or other CSS-special character can
                # never break the lookup.
                if field.evaluate(
                    "el => el.name ? "
                    "Array.from(document.getElementsByName(el.name)).some(r => r.checked) : "
                    "el.checked"
                ):
                    continue
            elif field_type == "checkbox":
                if field.is_checked():
                    continue
            elif field_type == "file":
                continue  # handled by _upload_resume_if_requested, not here
            elif (field.input_value() or "").strip():
                continue
            return self._label_for(field) or "(unlabeled required field)"
        return None

    def _find_submit_button(self) -> Locator | None:
        # Checked as separate, priority-ordered locators, NOT one combined
        # comma-selector + .first: a comma-separated CSS selector list
        # matches the union of every alternative in DOCUMENT order, not in
        # the order the alternatives are written - so a page with an
        # unrelated "Apply Now" button earlier in the DOM (e.g. a "similar
        # jobs" sidebar advertising a different posting) would have been
        # clicked instead of the real button[type="submit"], despite
        # type="submit" being listed first in the string. Only fall back to
        # a looser text match when nothing more specific exists at all.
        for selector in (
            'button[type="submit"]:visible, input[type="submit"]:visible',
            'button:has-text("Send Application"):visible',
            'button:has-text("Submit"):visible',
            'button:has-text("Apply Now"):visible',
        ):
            candidate = self._page.locator(selector)
            if candidate.count() > 0:
                return candidate.first
        return None

    def _click_progress_button(self) -> bool:
        candidate = self._page.locator(
            'button:has-text("Next"):visible, button:has-text("Continue"):visible'
        )
        if candidate.count() > 0:
            candidate.first.click()
            return True
        return False

    @staticmethod
    def _label_for(el: Locator) -> str:
        el_id = el.get_attribute("id")
        if el_id:
            label = el.page.locator(f'label[for="{el_id}"]')
            if label.count() > 0:
                return label.first.inner_text().strip()
        wrapping_label = el.locator("xpath=ancestor::label[1]")
        if wrapping_label.count() > 0:
            return wrapping_label.first.inner_text().strip()
        aria = el.get_attribute("aria-label")
        if aria:
            return aria.strip()
        placeholder = el.get_attribute("placeholder")
        if placeholder:
            return placeholder.strip()
        return ""

    @staticmethod
    def _numeric_value(answer: str) -> str | None:
        """Same purpose as linkedin_adapter.py's _numeric_value() -
        extracts a plain number from a free-text LLM answer (e.g. "5+
        years") for filling an input[type="number"] field, since the
        browser silently resets such a field to empty on anything that
        isn't a valid number. Returns the first digit sequence found (with
        an optional decimal part), or None if the answer has no digits.
        """
        match = re.search(r"\d+(?:\.\d+)?", answer)
        return match.group() if match else None

    @staticmethod
    def _best_match_index(options: list[str], answer: str) -> int | None:
        """Same bidirectional, word-boundary matching as
        linkedin_adapter.py's _best_match_index() - never falls back to an
        arbitrary option. Checks both directions: a short canonical answer
        (e.g. "Referral") naming one of several longer option labels needs
        the answer found within the option; a short option (most commonly
        a plain "Yes"/"No" pair) named by a longer, explanatory answer -
        qa_answerer.py's own prompt allows elaboration, it doesn't require
        a bare "yes"/"no" - needs the reverse, or the match would never
        succeed for any yes/no-shaped question no matter how clearly the
        answer states its position.
        """
        answer_norm = answer.strip().casefold()
        if not answer_norm:
            return None
        for i, opt in enumerate(options):
            if opt.strip().casefold() == answer_norm:
                return i
        pattern = re.compile(rf"(?<!\w){re.escape(answer_norm)}(?!\w)")
        for i, opt in enumerate(options):
            if pattern.search(opt.strip().casefold()):
                return i
        for i, opt in enumerate(options):
            opt_norm = opt.strip().casefold()
            if not opt_norm:
                continue
            opt_pattern = re.compile(rf"(?<!\w){re.escape(opt_norm)}(?!\w)")
            if opt_pattern.search(answer_norm):
                return i
        return None
