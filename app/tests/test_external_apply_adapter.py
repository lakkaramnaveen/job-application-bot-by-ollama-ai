"""Exercises ExternalApplyAdapter against local static HTML fixtures - never
touches a real employer site. The safety-boundary tests here (CAPTCHA,
account creation, sensitive fields) are the ones that matter most: this
module is EXPERIMENTAL and will get many real forms wrong, but it must
never cross these specific lines regardless of what a form asks for.
"""

from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from job_bot.browser.external_apply_adapter import (
    AccountCreationRequired,
    CaptchaEncountered,
    ExternalApplyAdapter,
)

FIXTURES = Path(__file__).parent / "fixtures"
FORM_FIXTURE = FIXTURES / "external_apply_form.html"
CAPTCHA_FIXTURE = FIXTURES / "external_apply_form_captcha.html"
PASSWORD_FIXTURE = FIXTURES / "external_apply_form_password.html"
SENSITIVE_FIELD_FIXTURE = FIXTURES / "external_apply_form_sensitive_field.html"
MULTI_STEP_FIXTURE = FIXTURES / "external_apply_form_multi_step.html"
UNRELATED_APPLY_NOW_FIXTURE = FIXTURES / "external_apply_form_unrelated_apply_now_button.html"
RADIO_DEFAULT_CHECKED_FIXTURE = FIXTURES / "external_apply_form_radio_group_default_checked.html"
RADIO_UNANSWERED_FIXTURE = FIXTURES / "external_apply_form_radio_group_unanswered.html"


@pytest.fixture
def playwright_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        yield page
        browser.close()


def test_fills_and_stops_before_submit_on_dry_run(playwright_page):
    playwright_page.goto(f"file://{FORM_FIXTURE}")
    adapter = ExternalApplyAdapter(playwright_page)

    submitted = adapter.fill_and_submit(
        answer_question=lambda label: "Jane Doe" if "name" in label.casefold() else "",
        resume_path=None,
        cover_letter_text="Dear Hiring Manager, ...",
        dry_run=True,
    )

    assert submitted is False
    assert playwright_page.locator("#full-name").input_value() == "Jane Doe"
    assert playwright_page.locator("#cover-note").input_value() == "Dear Hiring Manager, ..."


def test_submits_for_real_when_not_a_dry_run(playwright_page):
    playwright_page.goto(f"file://{FORM_FIXTURE}")
    adapter = ExternalApplyAdapter(playwright_page)

    submitted = adapter.fill_and_submit(
        answer_question=lambda label: "Jane Doe" if "name" in label.casefold() else "",
        resume_path=None,
        cover_letter_text=None,
        dry_run=False,
    )

    assert submitted is True


def test_selects_the_best_matching_option(playwright_page):
    playwright_page.goto(f"file://{FORM_FIXTURE}")
    adapter = ExternalApplyAdapter(playwright_page)

    def answer_question(label: str) -> str:
        if "name" in label.casefold():
            return "Jane Doe"
        if "hear about" in label.casefold():
            return "Referral"
        return ""

    adapter.fill_and_submit(
        answer_question=answer_question, resume_path=None, cover_letter_text=None, dry_run=True
    )

    assert playwright_page.locator("#how-heard").input_value() == "referral"


# --- _best_match_index (no browser needed - pure string matching) ---


def test_best_match_index_matches_a_short_option_named_by_a_long_explanatory_answer():
    """Same bug/fix as linkedin_adapter.py's equivalent test: qa_answerer.py's
    own prompt allows an explanatory answer rather than a bare "yes"/"no" -
    the original single-direction check (answer found within option) could
    never match a yes/no-shaped select option, no matter how clearly the
    answer states its position.
    """
    options = ["Yes", "No"]
    answer = "No, I am currently located in St Louis, MO and would need to relocate."
    assert ExternalApplyAdapter._best_match_index(options, answer) == 1


def test_best_match_index_reverse_direction_respects_word_boundaries():
    options = ["Yes", "No"]
    answer = "I know the role well and am a normal full-time candidate."
    assert ExternalApplyAdapter._best_match_index(options, answer) is None


def test_never_solves_a_captcha_and_raises_a_clear_error(playwright_page):
    playwright_page.goto(f"file://{CAPTCHA_FIXTURE}")
    adapter = ExternalApplyAdapter(playwright_page)

    with pytest.raises(CaptchaEncountered, match="CAPTCHA"):
        adapter.fill_and_submit(
            answer_question=lambda label: "Jane Doe",
            resume_path=None,
            cover_letter_text=None,
            dry_run=True,
        )

    # Confirms this fails *before* ever touching the form, not after a
    # partial fill attempt.
    assert playwright_page.locator("#full-name").input_value() == ""


def test_never_creates_an_account_or_enters_a_password(playwright_page):
    playwright_page.goto(f"file://{PASSWORD_FIXTURE}")
    adapter = ExternalApplyAdapter(playwright_page)

    with pytest.raises(AccountCreationRequired, match="account"):
        adapter.fill_and_submit(
            answer_question=lambda label: "Jane Doe" if "name" in label.casefold() else "hunter2",
            resume_path=None,
            cover_letter_text=None,
            dry_run=True,
        )

    assert playwright_page.locator("#account-password").input_value() == ""


def test_never_fills_a_field_asking_for_an_ssn_and_fails_on_it_being_required(playwright_page):
    """The SSN field is required and never filled (SENSITIVE_FIELD_MARKERS),
    so it's indistinguishable from an unanswerable required field - correct,
    since either way the application can't safely proceed.
    """
    playwright_page.goto(f"file://{SENSITIVE_FIELD_FIXTURE}")
    adapter = ExternalApplyAdapter(playwright_page)
    questions_asked: list[str] = []

    def answer_question(label: str) -> str:
        questions_asked.append(label)
        return "123-45-6789"  # even if the LLM were willing to answer, never used

    with pytest.raises(RuntimeError, match="Social Security"):
        adapter.fill_and_submit(
            answer_question=answer_question, resume_path=None, cover_letter_text=None, dry_run=True
        )

    assert playwright_page.locator("#ssn").input_value() == ""
    assert questions_asked == ["Full name"]  # the SSN field was never even asked about


def test_stops_when_a_required_field_cannot_be_answered(playwright_page):
    playwright_page.goto(f"file://{FORM_FIXTURE}")
    adapter = ExternalApplyAdapter(playwright_page)

    with pytest.raises(RuntimeError, match="Full name"):
        adapter.fill_and_submit(
            answer_question=lambda label: "",  # can't answer anything
            resume_path=None,
            cover_letter_text=None,
            dry_run=True,
        )


def test_multi_step_form_advances_past_next_and_completes(playwright_page):
    playwright_page.goto(f"file://{MULTI_STEP_FIXTURE}")
    adapter = ExternalApplyAdapter(playwright_page)

    def answer_question(label: str) -> str:
        if "name" in label.casefold():
            return "Jane Doe"
        if "years" in label.casefold():
            return "5"
        return ""

    submitted = adapter.fill_and_submit(
        answer_question=answer_question, resume_path=None, cover_letter_text=None, dry_run=True
    )

    assert submitted is False  # dry-run: reached Submit but stopped before clicking it
    assert playwright_page.locator("#full-name").input_value() == "Jane Doe"
    assert playwright_page.locator("#years-exp").input_value() == "5"


def test_does_not_block_on_a_required_sibling_when_the_radio_group_has_a_checked_default(
    playwright_page,
):
    """Real bug this guards against: _first_unanswered_required_field_label()
    used to check only the individual radio input's own .checked state, not
    whether any radio sharing its `name` (i.e. its actual group) was
    checked. A form that marks every radio in a group `required` (common -
    e.g. framework-generated accessibility markup) and defaults one option
    to checked (e.g. "Willing to relocate? Yes / No" defaulting to "No")
    then had every *other*, unchecked sibling misreported as its own
    unanswered required field, blocking an application that was actually
    complete. Confirmed live before this fix: this fixture raised
    RuntimeError("...('Yes')...") even though "No" was already checked.
    """
    playwright_page.goto(f"file://{RADIO_DEFAULT_CHECKED_FIXTURE}")
    adapter = ExternalApplyAdapter(playwright_page)

    submitted = adapter.fill_and_submit(
        answer_question=lambda label: "Jane Doe" if "name" in label.casefold() else "",
        resume_path=None,
        cover_letter_text=None,
        dry_run=True,
    )

    assert submitted is False  # dry-run: reached Submit but stopped before clicking it


def test_still_stops_on_a_genuinely_unanswered_required_radio_group(playwright_page):
    """Non-regression for the fix above: when no option in the group is
    checked at all, it must still block - the fix only changes how an
    *already-answered* group's other required siblings are treated, not
    whether an unanswered group is still caught.
    """
    playwright_page.goto(f"file://{RADIO_UNANSWERED_FIXTURE}")
    adapter = ExternalApplyAdapter(playwright_page)

    with pytest.raises(RuntimeError, match="required question has no answer"):
        adapter.fill_and_submit(
            answer_question=lambda label: "Jane Doe" if "name" in label.casefold() else "",
            resume_path=None,
            cover_letter_text=None,
            dry_run=True,
        )


def test_find_submit_button_prefers_type_submit_over_an_earlier_unrelated_button(playwright_page):
    """Real bug this guards against: a comma-separated CSS selector list
    (the old `'button[type="submit"], ..., button:has-text("Apply Now")'`
    + `.first`) matches in DOCUMENT order across every alternative
    combined, not in the order the alternatives are written - so a page
    with an unrelated "Apply Now" button earlier in the DOM (a "similar
    jobs" sidebar advertising a different posting is a realistic real-
    world shape for this) would have .first pick that one instead of the
    real button[type="submit"], despite type="submit" being listed first
    in the selector string. Confirmed live before this fix: .first
    resolved to the sidebar's "Apply Now" button, not the real submit.
    """
    playwright_page.goto(f"file://{UNRELATED_APPLY_NOW_FIXTURE}")
    adapter = ExternalApplyAdapter(playwright_page)

    button = adapter._find_submit_button()

    assert button is not None
    assert button.get_attribute("id") == "real-submit"
