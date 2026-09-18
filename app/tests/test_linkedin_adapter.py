"""Exercises the LinkedIn adapter's form-filling logic against a local static
HTML fixture that mimics the accessible structure of the Easy Apply modal
(role="dialog", label/input pairs, a fieldset radio group, a select, and an
aria-labeled submit button). This never touches the real linkedin.com - it
verifies the field-detection and answer-matching logic in isolation.
"""

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from playwright.sync_api import sync_playwright

from job_bot.browser.base_adapter import JobPosting
from job_bot.browser.linkedin_adapter import (
    DATE_POSTED_3_DAYS,
    DATE_POSTED_24H,
    RESULTS_PER_PAGE,
    LinkedInAdapter,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "easy_apply_form.html"
LINK_ENTRY_POINT_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "easy_apply_form_link_entry_point.html"
SEARCH_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "search_results.html"
RELATIVE_HREF_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "search_results_relative_hrefs.html"
ALL_APPLIED_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "search_results_all_applied.html"
PAGE_TWO_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "search_results_page_two.html"
RESUME_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_resume.txt"
AMBIGUOUS_FILE_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "easy_apply_form_ambiguous_file_field.html"
SELECT_NO_PLACEHOLDER_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "easy_apply_form_select_no_placeholder.html"
)
REQUIRED_FIELD_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "easy_apply_form_required_field.html"
RESUME_ALREADY_SELECTED_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "easy_apply_form_resume_already_selected.html"
)
RADIO_COVERED_BY_LABEL_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "easy_apply_form_radio_covered_by_label.html"
)
REQUIRED_RADIO_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "easy_apply_form_required_radio.html"
REQUIRED_SELECT_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "easy_apply_form_required_select.html"
MIXED_APPLY_TYPES_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "search_results_mixed_apply_types.html"
EXTERNAL_APPLY_POSTING_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "job_posting_external_apply.html"
EXTERNAL_APPLY_NO_POPUP_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "job_posting_external_apply_no_popup.html"
)
SUBMIT_BUTTON_TEXT_ONLY_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "easy_apply_form_submit_button_text_only.html"
)
NO_PROGRESS_BUTTON_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "easy_apply_form_no_progress_button.html"
ARIA_REQUIRED_TEXT_FIELD_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "easy_apply_form_aria_required_text_field.html"
)
REQUIRED_CHECKBOX_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "easy_apply_form_required_checkbox.html"
CHECKED_AND_OPTIONAL_CHECKBOXES_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "easy_apply_form_checked_and_optional_checkboxes.html"
)
REQUIRED_FILE_AMBIGUOUS_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "easy_apply_form_required_file_ambiguous.html"
)
DIALOG_NEVER_APPEARS_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "easy_apply_form_dialog_never_appears.html"
)


@pytest.fixture
def playwright_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        yield page
        browser.close()


def test_dry_run_fills_fields_and_stops_before_submit(playwright_page):
    posting = JobPosting(
        job_id="1",
        title="Backend Engineer",
        company="Acme",
        url=f"file://{FIXTURE_PATH}",
        description="",
    )
    adapter = LinkedInAdapter(playwright_page)

    answers = {
        "Years of Python experience": "5",
        "Are you authorized to work in the US?": "Yes",
        "Preferred start date": "Immediately",
    }

    def answer_question(label: str) -> str:
        return answers.get(label, "")

    submitted = adapter.fill_and_submit(
        posting,
        answer_question=answer_question,
        resume_path=None,
        cover_letter_text=None,
        dry_run=True,
    )

    assert submitted is False
    assert playwright_page.locator("#years-python").input_value() == "5"
    assert playwright_page.locator("#auth-yes").is_checked()
    assert not playwright_page.locator("#auth-no").is_checked()
    assert playwright_page.locator("#start-date").input_value() == "immediately"


def test_easy_apply_entry_point_matches_an_anchor_tag_not_just_a_button(playwright_page):
    """LinkedIn's actual current markup for the "Easy Apply" control is an
    <a> (confirmed against a live job posting - an href to an /apply/ URL,
    intercepted by JS to open the modal inline), not a <button>. Matching
    only button: here made every real run time out after 30s on every
    single posting, since the locator matched nothing at all.
    """
    posting = JobPosting(
        job_id="1",
        title="Backend Engineer",
        company="Acme",
        url=f"file://{LINK_ENTRY_POINT_FIXTURE_PATH}",
        description="",
    )
    adapter = LinkedInAdapter(playwright_page)

    submitted = adapter.fill_and_submit(
        posting,
        answer_question=lambda label: "5" if "Python" in label else "",
        resume_path=None,
        cover_letter_text=None,
        dry_run=True,
    )

    assert submitted is False
    assert playwright_page.locator("#years-python").input_value() == "5"


def test_unanswered_label_gets_empty_string_not_a_crash(playwright_page):
    posting = JobPosting(job_id="2", title="X", company="Y", url=f"file://{FIXTURE_PATH}", description="")
    adapter = LinkedInAdapter(playwright_page)

    submitted = adapter.fill_and_submit(
        posting,
        answer_question=lambda label: "",
        resume_path=None,
        cover_letter_text=None,
        dry_run=True,
    )

    assert submitted is False
    assert playwright_page.locator("#years-python").input_value() == ""
    # A high-stakes radio group (work authorization) with no matching
    # answer must be left unselected, never guessed - see
    # _best_match_index()'s docstring.
    assert not playwright_page.locator("#auth-yes").is_checked()
    assert not playwright_page.locator("#auth-no").is_checked()


def test_unanswered_required_field_fails_fast_with_a_specific_message(playwright_page):
    """Real failure this guards against: a required text field (LinkedIn's
    compound "Additional Questions" render as required type="text" inputs,
    not type="number") that answer_question() can't answer stays empty
    forever - LinkedIn's own client-side validation would then never let a
    real submission actually go through no matter how many times Next or
    Submit is clicked.

    Verified this is worse than just a wasted-LLM-calls inefficiency: on a
    fixture shaped like this one, where Submit is reachable on the same
    step as the unanswered field (many real Easy Apply forms are exactly
    one step), the pre-fix code found and clicked Submit anyway - `DID NOT
    RAISE` when this test's fix was reverted - which on a real non-dry-run
    form would submit incomplete and still get recorded as `applied`
    (validation blocks the actual employer-side submission, but
    fill_and_submit() has no way to know that; it only knows it clicked
    something). The "stuck after 20 Next clicks" message only ever
    surfaced on forms with more steps between the empty field and Submit.
    """
    posting = JobPosting(
        job_id="1", title="X", company="Y", url=f"file://{REQUIRED_FIELD_FIXTURE_PATH}", description=""
    )
    adapter = LinkedInAdapter(playwright_page)
    questions_asked: list[str] = []

    def answer_question(label: str) -> str:
        questions_asked.append(label)
        return "5" if "Python experience" in label else ""  # the compound question is unanswerable

    with pytest.raises(RuntimeError, match="Golang"):
        adapter.fill_and_submit(
            posting,
            answer_question=answer_question,
            resume_path=None,
            cover_letter_text=None,
            dry_run=True,
        )

    # Failed on the first pass through the loop - the unanswerable question
    # was asked once, not up to max_steps (20) times.
    assert questions_asked.count("How many years with any two of Golang, Java, Node.js, or Python?") == 1


def test_unanswered_aria_required_text_field_fails_fast_instead_of_submitting_incomplete(playwright_page):
    """Real bug this guards against: _first_unanswered_required_text_field_
    label() checked only the native `required` attribute, not
    aria-required="true" - unlike its radio/select counterpart,
    _first_unanswered_required_choice_label(), which already checked both
    via _is_marked_required(). A text field required only via aria-required
    was therefore invisible to this check entirely: on this fixture (Submit
    reachable on the same step, no other required field to catch it first),
    fill_and_submit() found and clicked Submit anyway with the field still
    empty - confirmed live before this fix, `dry_run=True` still returned
    False (reached Submit) instead of raising.
    """
    posting = JobPosting(
        job_id="1", title="X", company="Y", url=f"file://{ARIA_REQUIRED_TEXT_FIELD_FIXTURE_PATH}", description=""
    )
    adapter = LinkedInAdapter(playwright_page)

    with pytest.raises(RuntimeError, match="Desired salary"):
        adapter.fill_and_submit(
            posting,
            answer_question=lambda label: "",  # can't answer - field stays empty
            resume_path=None,
            cover_letter_text=None,
            dry_run=True,
        )


def test_unanswered_required_checkbox_fails_fast_instead_of_submitting_incomplete(playwright_page):
    """Real bug this guards against: nothing in this adapter mentioned
    checkboxes at all - unlike external_apply_adapter.py's equivalent
    required-field check, which explicitly handles them. A required
    consent/agreement checkbox is correctly never auto-checked (same
    "never guess/never act on the user's behalf" stance as a radio/select
    non-match), but with no detection for it either, fill_and_submit()
    went on to find and click Submit with it still unchecked - confirmed
    live before this fix, `dry_run=True` returned False (reached Submit)
    instead of raising.
    """
    posting = JobPosting(
        job_id="1", title="X", company="Y", url=f"file://{REQUIRED_CHECKBOX_FIXTURE_PATH}", description=""
    )
    adapter = LinkedInAdapter(playwright_page)

    with pytest.raises(RuntimeError, match="background check"):
        adapter.fill_and_submit(
            posting,
            answer_question=lambda label: "",
            resume_path=None,
            cover_letter_text=None,
            dry_run=True,
        )
    assert not playwright_page.locator("#consent").is_checked()


def test_checked_required_checkbox_does_not_block_an_unchecked_optional_one(playwright_page):
    """Non-regression for the fix above: a required checkbox already
    checked (a page-supplied default) must not block, and an unrelated
    unchecked *optional* checkbox alongside it must not be mistaken for a
    required one either.
    """
    posting = JobPosting(
        job_id="1",
        title="X",
        company="Y",
        url=f"file://{CHECKED_AND_OPTIONAL_CHECKBOXES_FIXTURE_PATH}",
        description="",
    )
    adapter = LinkedInAdapter(playwright_page)

    submitted = adapter.fill_and_submit(
        posting,
        answer_question=lambda label: "",
        resume_path=None,
        cover_letter_text=None,
        dry_run=True,
    )

    assert submitted is False  # dry-run: reached Submit but stopped before clicking it


def test_unanswered_required_file_field_fails_fast_instead_of_submitting_incomplete(playwright_page):
    """Real bug this guards against: nothing checked required file inputs
    at all. Two file fields, neither confidently identifiable as the
    resume field (see _upload_resume_if_requested()'s "ambiguous file
    field" branch, already covered by
    test_resume_is_not_uploaded_when_a_second_file_field_is_ambiguous) -
    correctly leaving both empty rather than guessing which to fill. But
    the first one is required, and with no detection for an unfilled
    required file input, fill_and_submit() went on to find and click
    Submit anyway - confirmed live before this fix, `dry_run=True`
    returned False (reached Submit) instead of raising.
    """
    posting = JobPosting(
        job_id="1", title="X", company="Y", url=f"file://{REQUIRED_FILE_AMBIGUOUS_FIXTURE_PATH}", description=""
    )
    adapter = LinkedInAdapter(playwright_page)

    with pytest.raises(RuntimeError, match="Supporting document 1"):
        adapter.fill_and_submit(
            posting,
            answer_question=lambda label: "",
            resume_path=str(RESUME_FIXTURE_PATH),
            cover_letter_text=None,
            dry_run=True,
        )


def test_dialog_never_appearing_raises_a_diagnostic_error_not_a_bare_playwright_timeout(playwright_page):
    """Real failure from an actual run's failed_applications.log: clicking
    Easy Apply can silently open nothing (a re-authentication checkpoint,
    a "no longer accepting applications" notice, or unusually slow
    rendering instead of the dialog) - Playwright's own timeout message
    named only the selector that never became visible, giving no lead on
    what the page showed instead. Takes the real ~10s timeout to run,
    since that's the actual behavior under test.
    """
    posting = JobPosting(
        job_id="1", title="X", company="Y", url=f"file://{DIALOG_NEVER_APPEARS_FIXTURE_PATH}", description=""
    )
    adapter = LinkedInAdapter(playwright_page)

    with pytest.raises(RuntimeError, match="dialog never appeared") as exc_info:
        adapter.fill_and_submit(
            posting,
            answer_question=lambda label: "",
            resume_path=None,
            cover_letter_text=None,
            dry_run=True,
        )

    assert "page was at" in str(exc_info.value)
    assert "titled" in str(exc_info.value)


def test_answering_every_required_field_still_completes_normally(playwright_page):
    posting = JobPosting(
        job_id="1", title="X", company="Y", url=f"file://{REQUIRED_FIELD_FIXTURE_PATH}", description=""
    )
    adapter = LinkedInAdapter(playwright_page)

    submitted = adapter.fill_and_submit(
        posting,
        answer_question=lambda label: "5",
        resume_path=None,
        cover_letter_text=None,
        dry_run=True,
    )

    assert submitted is False
    assert playwright_page.locator("#years-python").input_value() == "5"
    assert playwright_page.locator("#backend-combo").input_value() == "5"


def test_submit_button_matched_by_text_when_aria_label_is_missing(playwright_page):
    """"Stuck on a step with no Next/Review/Submit button found" was, by a
    wide margin, this adapter's single most common real-world failure - a
    real audit log showed ~124 occurrences of it against 80 successful
    applications. A button whose aria-label doesn't exactly match "Submit
    application" (an A/B-tested LinkedIn rollout, a differently-generated
    form) is a plausible cause the old aria-label-only selector could never
    catch - SUBMIT_BUTTON_SELECTORS' text-based fallback covers exactly
    this: a submit button with no aria-label at all, just visible text.
    """
    posting = JobPosting(
        job_id="1", title="X", company="Y", url=f"file://{SUBMIT_BUTTON_TEXT_ONLY_FIXTURE_PATH}", description=""
    )
    adapter = LinkedInAdapter(playwright_page)

    submitted = adapter.fill_and_submit(
        posting,
        answer_question=lambda label: "5",
        resume_path=None,
        cover_letter_text=None,
        dry_run=False,
    )

    assert submitted is True


def test_stuck_error_names_the_buttons_that_were_actually_on_screen(playwright_page):
    """The generic "stuck" message used to give zero lead on why - every
    occurrence looked identical in failed_applications.log regardless of
    cause. This checks the diagnostic detail added alongside the fallback
    selectors above actually surfaces whatever button text was visible
    (here, an unrelated "Save and exit" button, no real progress control at
    all), so a future occurrence isn't another unexplained black box.
    """
    posting = JobPosting(
        job_id="1", title="X", company="Y", url=f"file://{NO_PROGRESS_BUTTON_FIXTURE_PATH}", description=""
    )
    adapter = LinkedInAdapter(playwright_page)

    with pytest.raises(RuntimeError, match="Save and exit"):
        adapter.fill_and_submit(
            posting,
            answer_question=lambda label: "5",
            resume_path=None,
            cover_letter_text=None,
            dry_run=True,
        )


def test_unanswered_required_radio_group_fails_fast_with_a_specific_message(playwright_page):
    """Real-world failure this guards against: an eligibility/sponsorship-
    style radio question the LLM's answer doesn't clearly match either
    option for is deliberately left unanswered (see _select_best_radio()'s
    "never guess" docstring) - which, before this check existed, fell all
    the way through to the generic "stuck on a step" RuntimeError with no
    indication of which question was the actual problem. Across weeks of
    real runs this was the dominant failure mode: audit.log showed ~33
    generic "stuck" errors against a single specific one, because
    LinkedIn's own high-stakes questions are overwhelmingly radio groups,
    not free text (the only case the older, text-only check could name).
    """
    posting = JobPosting(
        job_id="1", title="X", company="Y", url=f"file://{REQUIRED_RADIO_FIXTURE_PATH}", description=""
    )
    adapter = LinkedInAdapter(playwright_page)

    with pytest.raises(RuntimeError, match="relocate"):
        adapter.fill_and_submit(
            posting,
            answer_question=lambda label: "I am not sure how to answer that",
            resume_path=None,
            cover_letter_text=None,
            dry_run=True,
        )


def test_unanswered_required_select_fails_fast_with_a_specific_message(playwright_page):
    posting = JobPosting(
        job_id="1", title="X", company="Y", url=f"file://{REQUIRED_SELECT_FIXTURE_PATH}", description=""
    )
    adapter = LinkedInAdapter(playwright_page)

    with pytest.raises(RuntimeError, match="security clearance"):
        adapter.fill_and_submit(
            posting,
            answer_question=lambda label: "I am not sure how to answer that",
            resume_path=None,
            cover_letter_text=None,
            dry_run=True,
        )


def test_answered_required_radio_group_does_not_fail(playwright_page):
    """A required radio group the answer DOES clearly match must not be
    mistaken for an unanswered one - _first_unanswered_required_choice_label()
    checks is_checked(), so a real match from _select_best_radio() earlier
    in the same pass must clear this check.
    """
    posting = JobPosting(
        job_id="1", title="X", company="Y", url=f"file://{REQUIRED_RADIO_FIXTURE_PATH}", description=""
    )
    adapter = LinkedInAdapter(playwright_page)

    submitted = adapter.fill_and_submit(
        posting,
        answer_question=lambda label: "Yes",
        resume_path=None,
        cover_letter_text=None,
        dry_run=True,
    )

    assert submitted is False
    assert playwright_page.locator("#relocate-yes").is_checked()


def test_radio_covered_by_its_own_label_is_still_selected(playwright_page):
    """Real-world failure this guards against: LinkedIn commonly styles
    radio buttons as pills/cards with the native <input> visually hidden
    behind its own <label>, which is the actual clickable surface. Checking
    the input directly (the old behavior) fails Playwright's actionability
    check ("label intercepts pointer events") - observed live timing out
    after ~30s on a real application - so _select_best_radio() must click
    the label instead, exactly like a real user does.
    """
    posting = JobPosting(
        job_id="4", title="X", company="Y", url=f"file://{RADIO_COVERED_BY_LABEL_FIXTURE_PATH}", description=""
    )
    adapter = LinkedInAdapter(playwright_page)

    submitted = adapter.fill_and_submit(
        posting,
        answer_question=lambda label: "Yes",
        resume_path=None,
        cover_letter_text=None,
        dry_run=True,
    )

    assert submitted is False
    assert playwright_page.locator("#relocate-yes").is_checked()
    assert not playwright_page.locator("#relocate-no").is_checked()


def test_non_matching_answer_never_guesses_a_radio_option(playwright_page):
    """An answer that doesn't correspond to either radio option's text
    (e.g. the LLM said something not literally "Yes"/"No") must not fall
    back to picking an arbitrary option on a field this sensitive.
    """
    posting = JobPosting(job_id="2b", title="X", company="Y", url=f"file://{FIXTURE_PATH}", description="")
    adapter = LinkedInAdapter(playwright_page)

    adapter.fill_and_submit(
        posting,
        answer_question=lambda label: "I am not sure how to answer that",
        resume_path=None,
        cover_letter_text=None,
        dry_run=True,
    )

    assert not playwright_page.locator("#auth-yes").is_checked()
    assert not playwright_page.locator("#auth-no").is_checked()


def test_select_with_no_blank_placeholder_still_gets_answered(playwright_page):
    """A <select> with no blank "Select an option" placeholder has its first
    real option auto-selected by the browser with nothing chosen by anyone.
    The old heuristic (any non-empty, non-placeholder-text value means
    "already answered") would treat that as a real answer and skip the
    field entirely, silently submitting the unvetted default. The fix
    (selectedIndex == 0 means "unanswered") must still consult
    answer_question() and select the right option.
    """
    posting = JobPosting(
        job_id="2c", title="X", company="Y", url=f"file://{SELECT_NO_PLACEHOLDER_FIXTURE_PATH}", description=""
    )
    adapter = LinkedInAdapter(playwright_page)

    adapter.fill_and_submit(
        posting,
        answer_question=lambda label: "Yes",
        resume_path=None,
        cover_letter_text=None,
        dry_run=True,
    )

    assert playwright_page.locator("#work-auth").input_value() == "yes"


def test_resume_is_uploaded_when_resume_path_given(playwright_page):
    posting = JobPosting(job_id="3", title="X", company="Y", url=f"file://{FIXTURE_PATH}", description="")
    adapter = LinkedInAdapter(playwright_page)

    adapter.fill_and_submit(
        posting,
        answer_question=lambda label: "",
        resume_path=str(RESUME_FIXTURE_PATH),
        cover_letter_text=None,
        dry_run=True,
    )

    uploaded = playwright_page.evaluate("document.getElementById('resume-upload').files[0]?.name")
    assert uploaded == RESUME_FIXTURE_PATH.name


def test_resume_is_not_uploaded_to_a_differently_labeled_file_field(playwright_page):
    """A file input explicitly labeled for something else (a cover letter
    document, in this fixture) must never receive the resume file - see
    _looks_like_non_resume_file_field().
    """
    posting = JobPosting(job_id="3b", title="X", company="Y", url=f"file://{FIXTURE_PATH}", description="")
    adapter = LinkedInAdapter(playwright_page)

    adapter.fill_and_submit(
        posting,
        answer_question=lambda label: "",
        resume_path=str(RESUME_FIXTURE_PATH),
        cover_letter_text=None,
        dry_run=True,
    )

    resume_field_count = playwright_page.evaluate("document.getElementById('cover-letter-file').files.length")
    assert resume_field_count == 0
    uploaded = playwright_page.evaluate("document.getElementById('resume-upload').files[0]?.name")
    assert uploaded == RESUME_FIXTURE_PATH.name


def test_resume_is_not_reuploaded_when_linkedin_already_has_one_selected(playwright_page):
    """Real-world bug this guards against: LinkedIn's "Resume" step shows a
    card list of previously uploaded resumes with one already selected, but
    still keeps a hidden input[type="file"] on the page regardless (behind
    the "Upload resume" button). Before this check existed, that hidden
    input was blindly filled on every single application - confirmed live,
    this silently added a duplicate copy of the same resume document to the
    user's LinkedIn resume library each time (5 identical entries had
    accumulated there from repeated runs).
    """
    posting = JobPosting(
        job_id="3d", title="X", company="Y", url=f"file://{RESUME_ALREADY_SELECTED_FIXTURE_PATH}", description=""
    )
    adapter = LinkedInAdapter(playwright_page)

    adapter.fill_and_submit(
        posting,
        answer_question=lambda label: "",
        resume_path=str(RESUME_FIXTURE_PATH),
        cover_letter_text=None,
        dry_run=True,
    )

    file_count = playwright_page.evaluate("document.getElementById('resume-upload').files.length")
    assert file_count == 0


def test_resume_is_not_uploaded_when_a_second_file_field_is_ambiguous(playwright_page):
    """Two file fields on the same step, one clearly the resume ("Resume")
    and one neither positively resume-shaped nor denylisted ("Additional
    attachment") - the ambiguous one must be left alone rather than
    guessed, even though it isn't on the non-resume denylist. See
    _upload_resume_if_requested()'s multi-field branch.
    """
    posting = JobPosting(
        job_id="3c", title="X", company="Y", url=f"file://{AMBIGUOUS_FILE_FIXTURE_PATH}", description=""
    )
    adapter = LinkedInAdapter(playwright_page)

    adapter.fill_and_submit(
        posting,
        answer_question=lambda label: "",
        resume_path=str(RESUME_FIXTURE_PATH),
        cover_letter_text=None,
        dry_run=True,
    )

    uploaded = playwright_page.evaluate("document.getElementById('resume-upload').files[0]?.name")
    assert uploaded == RESUME_FIXTURE_PATH.name
    mystery_count = playwright_page.evaluate("document.getElementById('mystery-file').files.length")
    assert mystery_count == 0


def test_no_upload_attempted_when_resume_path_is_none(playwright_page):
    posting = JobPosting(job_id="4", title="X", company="Y", url=f"file://{FIXTURE_PATH}", description="")
    adapter = LinkedInAdapter(playwright_page)

    adapter.fill_and_submit(
        posting,
        answer_question=lambda label: "",
        resume_path=None,
        cover_letter_text=None,
        dry_run=True,
    )

    uploaded = playwright_page.evaluate("document.getElementById('resume-upload').files.length")
    assert uploaded == 0


def test_cover_letter_field_is_filled_from_generated_cover_letter(playwright_page):
    posting = JobPosting(job_id="5", title="X", company="Y", url=f"file://{FIXTURE_PATH}", description="")
    adapter = LinkedInAdapter(playwright_page)

    def answer_question(label: str) -> str:
        assert "cover letter" not in label.casefold(), (
            f"answer_question should not be called for the cover letter field, got label={label!r}"
        )
        return ""

    adapter.fill_and_submit(
        posting,
        answer_question=answer_question,
        resume_path=None,
        cover_letter_text="Dear Hiring Manager, I'm excited to apply.",
        dry_run=True,
    )

    assert (
        playwright_page.locator("#cover-letter").input_value() == "Dear Hiring Manager, I'm excited to apply."
    )


def test_cover_letter_field_falls_back_to_qa_when_no_cover_letter_given(playwright_page):
    posting = JobPosting(job_id="6", title="X", company="Y", url=f"file://{FIXTURE_PATH}", description="")
    adapter = LinkedInAdapter(playwright_page)

    def answer_question(label: str) -> str:
        return "fallback answer" if "cover letter" in label.casefold() else ""

    adapter.fill_and_submit(
        posting,
        answer_question=answer_question,
        resume_path=None,
        cover_letter_text=None,
        dry_run=True,
    )

    assert playwright_page.locator("#cover-letter").input_value() == "fallback answer"


def test_search_skips_already_applied_and_deduplicates_across_pages(playwright_page, monkeypatch):
    real_goto = playwright_page.goto
    monkeypatch.setattr(playwright_page, "goto", lambda url, **kw: real_goto(f"file://{SEARCH_FIXTURE_PATH}"))
    adapter = LinkedInAdapter(playwright_page)

    postings = adapter.search("python", "Remote", max_results=10)

    ids = {p.job_id for p in postings}
    assert ids == {"101", "103"}  # 102 is marked Applied and excluded
    assert all(p.title and p.company for p in postings)


def test_search_respects_max_results(playwright_page, monkeypatch):
    real_goto = playwright_page.goto
    monkeypatch.setattr(playwright_page, "goto", lambda url, **kw: real_goto(f"file://{SEARCH_FIXTURE_PATH}"))
    adapter = LinkedInAdapter(playwright_page)

    postings = adapter.search("python", "Remote", max_results=1)

    assert len(postings) == 1


def test_search_adds_the_experience_level_filter_to_the_search_url(playwright_page, monkeypatch):
    real_goto = playwright_page.goto
    requested_urls = []

    def fake_goto(url, **kw):
        requested_urls.append(url)
        return real_goto(f"file://{SEARCH_FIXTURE_PATH}")

    monkeypatch.setattr(playwright_page, "goto", fake_goto)
    adapter = LinkedInAdapter(playwright_page)

    adapter.search("python", "Remote", max_results=10, experience_levels=["mid-senior", "director"])

    assert "f_E=4,5" in requested_urls[0]


def test_search_omits_the_experience_level_filter_when_not_given(playwright_page, monkeypatch):
    real_goto = playwright_page.goto
    requested_urls = []

    def fake_goto(url, **kw):
        requested_urls.append(url)
        return real_goto(f"file://{SEARCH_FIXTURE_PATH}")

    monkeypatch.setattr(playwright_page, "goto", fake_goto)
    adapter = LinkedInAdapter(playwright_page)

    adapter.search("python", "Remote", max_results=10)

    assert "f_E=" not in requested_urls[0]


# --- _best_match_index (no browser needed - pure string matching) ---


def test_best_match_index_exact_match():
    assert LinkedInAdapter._best_match_index(["Yes", "No"], "No") == 1


def test_best_match_index_word_boundary_fallback_matches_correctly():
    options = ["Not sure", "Yes, I am authorized"]
    assert LinkedInAdapter._best_match_index(options, "yes") == 1


def test_best_match_index_does_not_match_an_unrelated_option_containing_the_answer_as_a_substring():
    """Plain (unanchored) substring containment would match answer "no"
    against option "None" (which contains "no") or "Notice period" - wrong
    option, silently selected, on a field this sensitive. Word-boundarying
    the fallback match must return None here instead of guessing.
    """
    assert LinkedInAdapter._best_match_index(["None", "Notice period"], "no") is None


def test_best_match_index_word_boundary_still_matches_within_a_longer_option():
    """The word-boundary fix must not become *too* strict - a short answer
    that's genuinely a whole word within a longer option's text should still
    match (this is the case the fallback tier exists for).
    """
    assert LinkedInAdapter._best_match_index(["I am not sure", "No, I am not"], "no") == 1


@pytest.mark.parametrize(
    ("answer", "options", "expected"),
    [
        ("5+", ["Select an option", "1-2 years", "5+ years"], 2),
        ("C++", ["Java", "C++ developer"], 1),
        ("100%", ["50% travel", "100% remote"], 1),
    ],
)
def test_best_match_index_matches_answers_ending_in_a_non_word_character(answer, options, expected):
    """An answer whose first or last character isn't a word character
    ("5+", "C++", "100%") used to match nothing at all: \\b is defined
    relative to the adjacent character on both sides, so r"\\b5\\+\\b" can
    never match "5+ years" - the position after "+" sits between two
    non-word characters. The field was then left blank, which stalls the
    Easy Apply form on a required question.
    """
    assert LinkedInAdapter._best_match_index(options, answer) == expected


def test_best_match_index_non_word_answer_still_rejects_a_partial_number_match():
    """The looser matching must not make "5+" match "15+ years" - the digit
    is butted against another digit, which is exactly what the boundary
    check exists to reject.
    """
    assert LinkedInAdapter._best_match_index(["15+ years", "Not sure"], "5+") is None


def test_best_match_index_matches_a_short_option_named_by_a_long_explanatory_answer():
    """Real bug this guards against: qa_answerer.py's own system prompt
    allows (and in practice usually produces) an explanatory answer rather
    than a bare "yes"/"no" - the original single-direction check (answer
    found within option) can then never match ANY yes/no-shaped question,
    no matter how clearly the answer states its position, since the
    answer is always longer than either option. Confirmed live in one
    real user's own data: this exact answer shape, against options
    ["Yes", "No"], returned None under the old code and was recorded as
    the same unanswerable "commuting" gap 27 times despite the model
    clearly knowing and stating the answer every time.
    """
    options = ["Yes", "No"]
    answer = "No, I am currently located in St Louis, MO and would need to relocate."
    assert LinkedInAdapter._best_match_index(options, answer) == 1


def test_best_match_index_reverse_direction_respects_word_boundaries():
    """The reverse-direction fallback must not match option "No" against
    an answer merely containing "no" as a substring of a longer word
    (e.g. "know", "normal") - the same word-boundary protection the
    forward direction already has.
    """
    options = ["Yes", "No"]
    answer = "I know the role well and am a normal full-time candidate."
    assert LinkedInAdapter._best_match_index(options, answer) is None


def test_best_match_index_reverse_direction_picks_the_first_stated_option():
    """When an answer's wording could plausibly relate to more than one
    option, the option matched first (in the order given - the order the
    real form presents them in) wins, consistent with how a direct answer
    naturally leads with its actual position.
    """
    options = ["Yes", "No", "Maybe"]
    answer = "No, though I might consider it under the right circumstances - maybe."
    assert LinkedInAdapter._best_match_index(options, answer) == 1


# --- search() URL handling and pagination ---


def test_search_resolves_relative_hrefs_to_absolute_linkedin_urls(playwright_page, monkeypatch):
    """LinkedIn serves job-card anchors with root-relative hrefs. Storing
    those verbatim makes every downstream consumer break: page.goto()
    rejects a relative URL outright (so load_description/fill_and_submit
    can't open the posting), and the dashboard renders it as a link to its
    own localhost origin.
    """
    real_goto = playwright_page.goto
    monkeypatch.setattr(
        playwright_page, "goto", lambda url, **kw: real_goto(f"file://{RELATIVE_HREF_FIXTURE_PATH}")
    )
    adapter = LinkedInAdapter(playwright_page)

    postings = adapter.search("python", "Remote", max_results=10)

    assert [p.url for p in postings] == [
        "https://www.linkedin.com/jobs/view/201/?refId=abc&trackingId=xyz",
        "https://www.linkedin.com/jobs/view/202/",
    ]


def test_search_leaves_an_already_absolute_href_untouched(playwright_page, monkeypatch):
    real_goto = playwright_page.goto
    monkeypatch.setattr(playwright_page, "goto", lambda url, **kw: real_goto(f"file://{SEARCH_FIXTURE_PATH}"))
    adapter = LinkedInAdapter(playwright_page)

    postings = adapter.search("python", "Remote", max_results=10)

    assert all(p.url.startswith("https://example.com/jobs/") for p in postings)


def test_search_pages_past_a_page_whose_results_are_all_already_applied(playwright_page, monkeypatch):
    """A first page of results the user has already applied to is common in
    an active search. It yields zero postings, but it is still real progress
    through the result set - ending the search there (as keying the
    stop-condition off "postings kept on this page" did) never reaches the
    applicable jobs on the next page.
    """
    real_goto = playwright_page.goto
    requested_starts = []

    def fake_goto(url, **kw):
        start = int(parse_qs(urlparse(url).query).get("start", ["0"])[0])
        requested_starts.append(start)
        if start == 0:
            return real_goto(f"file://{ALL_APPLIED_FIXTURE_PATH}")
        if start == RESULTS_PER_PAGE:
            return real_goto(f"file://{PAGE_TWO_FIXTURE_PATH}")
        return real_goto("about:blank")

    monkeypatch.setattr(playwright_page, "goto", fake_goto)
    adapter = LinkedInAdapter(playwright_page)

    postings = adapter.search("python", "Remote", max_results=10)

    assert [p.job_id for p in postings] == ["403"]
    assert requested_starts[:2] == [0, RESULTS_PER_PAGE]


def test_search_stops_when_a_page_repeats_only_already_seen_jobs(playwright_page, monkeypatch):
    """The stop condition that remains: LinkedIn re-serving results already
    collected from an earlier page is how paging past the last result looks,
    and the search must end rather than loop to MAX_SEARCH_PAGES - within
    each date-posted window search() tries (see the 24h/3-day widening
    test below), not just once overall.
    """
    real_goto = playwright_page.goto
    page_loads = []

    def fake_goto(url, **kw):
        page_loads.append(url)
        return real_goto(f"file://{SEARCH_FIXTURE_PATH}")

    monkeypatch.setattr(playwright_page, "goto", fake_goto)
    adapter = LinkedInAdapter(playwright_page)

    postings = adapter.search("python", "Remote", max_results=50)

    assert {p.job_id for p in postings} == {"101", "103"}
    # Only 2 unique jobs ever exist in this fixture regardless of URL, so
    # max_results=50 is never reached in the 24h window either - search()
    # widens to the 3-day window too, each stopping after 2 pages the same
    # way (page 2 repeats page 1's ids) - 4 page loads total, not 2.
    assert len(page_loads) == 4


def test_search_uses_the_past_24_hours_filter_first(playwright_page, monkeypatch):
    real_goto = playwright_page.goto
    urls: list[str] = []

    def fake_goto(url, **kw):
        urls.append(url)
        return real_goto(f"file://{SEARCH_FIXTURE_PATH}")

    monkeypatch.setattr(playwright_page, "goto", fake_goto)
    adapter = LinkedInAdapter(playwright_page)

    adapter.search("python", "Remote", max_results=2)

    assert f"f_TPR={DATE_POSTED_24H}" in urls[0]


def test_search_widens_to_3_days_when_24h_window_has_too_few_results(playwright_page, monkeypatch):
    """Real-world behavior this is for: only searching the last 24 hours
    can genuinely come up short (fewer postings than max_results), and the
    fallback must never reach further back than 3 days - never a week or a
    month - so results stay fresh even when widened.
    """
    real_goto = playwright_page.goto
    urls: list[str] = []

    def fake_goto(url, **kw):
        urls.append(url)
        return real_goto(f"file://{SEARCH_FIXTURE_PATH}")

    monkeypatch.setattr(playwright_page, "goto", fake_goto)
    adapter = LinkedInAdapter(playwright_page)

    # The fixture only ever has 2 unique jobs, so max_results=50 can't be
    # satisfied by the 24h window alone.
    postings = adapter.search("python", "Remote", max_results=50)

    assert {p.job_id for p in postings} == {"101", "103"}
    assert any(f"f_TPR={DATE_POSTED_3_DAYS}" in url for url in urls)
    # Never widens past 3 days - no week/month-scale filter value used.
    assert not any("r604800" in url or "r2592000" in url for url in urls)


def test_search_does_not_widen_when_24h_window_already_has_enough(playwright_page, monkeypatch):
    real_goto = playwright_page.goto
    urls: list[str] = []

    def fake_goto(url, **kw):
        urls.append(url)
        return real_goto(f"file://{SEARCH_FIXTURE_PATH}")

    monkeypatch.setattr(playwright_page, "goto", fake_goto)
    adapter = LinkedInAdapter(playwright_page)

    # The fixture's 2 unique jobs exactly satisfy max_results=2 - no need
    # to ever ask for the wider, less-fresh window.
    adapter.search("python", "Remote", max_results=2)

    assert not any(f"f_TPR={DATE_POSTED_3_DAYS}" in url for url in urls)


def test_search_default_marks_every_posting_easy_apply(playwright_page, monkeypatch):
    real_goto = playwright_page.goto
    monkeypatch.setattr(playwright_page, "goto", lambda url, **kw: real_goto(f"file://{SEARCH_FIXTURE_PATH}"))
    adapter = LinkedInAdapter(playwright_page)

    postings = adapter.search("python", "Remote", max_results=10)

    assert all(p.easy_apply for p in postings)


def test_search_include_external_classifies_each_posting_by_its_own_badge(playwright_page, monkeypatch):
    """f_AL=true (LinkedIn's own Easy Apply filter) is only omitted from the
    search URL when include_external=True is actually passed - otherwise
    every result is guaranteed Easy Apply server-side and trusted as such
    without needing a per-card badge (see the other test above). Only a
    mixed page needs classifying at all.
    """
    real_goto = playwright_page.goto
    requested_urls = []

    def fake_goto(url, **kw):
        requested_urls.append(url)
        return real_goto(f"file://{MIXED_APPLY_TYPES_FIXTURE_PATH}")

    monkeypatch.setattr(playwright_page, "goto", fake_goto)
    adapter = LinkedInAdapter(playwright_page)

    postings = adapter.search("python", "Remote", max_results=10, include_external=True)

    by_id = {p.job_id: p for p in postings}
    assert by_id["201"].easy_apply is True
    assert by_id["202"].easy_apply is False
    assert by_id["203"].easy_apply is True
    assert "f_AL=true" not in requested_urls[0]


def test_open_external_application_returns_the_popup_page(playwright_page, monkeypatch):
    real_goto = playwright_page.goto
    monkeypatch.setattr(
        playwright_page, "goto", lambda url, **kw: real_goto(f"file://{EXTERNAL_APPLY_POSTING_FIXTURE_PATH}")
    )
    adapter = LinkedInAdapter(playwright_page)
    posting = JobPosting(
        job_id="202",
        title="Senior Java Engineer",
        company="FusionAuth",
        url=f"file://{EXTERNAL_APPLY_POSTING_FIXTURE_PATH}",
        description="",
        easy_apply=False,
    )

    external_page = adapter.open_external_application(posting)

    try:
        assert external_page is not None
        assert "external_company_application_form.html" in external_page.url
        assert external_page.locator("#full-name").count() == 1
    finally:
        if external_page is not None:
            external_page.close()


def test_open_external_application_raises_a_diagnostic_error_when_no_popup_opens(playwright_page, monkeypatch):
    """Same "diagnostic, not a bare Playwright timeout" treatment as
    fill_and_submit()'s dialog wait: a click that fails to open a popup at
    all (the destination navigating in place instead of via window.open(),
    or a broken control) used to surface as a bare
    playwright._impl._errors.TimeoutError with no indication of what the
    page actually did instead. Takes the real ~15s timeout to run, since
    that's the actual behavior under verification.
    """
    real_goto = playwright_page.goto
    monkeypatch.setattr(
        playwright_page,
        "goto",
        lambda url, **kw: real_goto(f"file://{EXTERNAL_APPLY_NO_POPUP_FIXTURE_PATH}"),
    )
    adapter = LinkedInAdapter(playwright_page)
    posting = JobPosting(
        job_id="202",
        title="Senior Java Engineer",
        company="FusionAuth",
        url=f"file://{EXTERNAL_APPLY_NO_POPUP_FIXTURE_PATH}",
        description="",
        easy_apply=False,
    )

    with pytest.raises(RuntimeError, match="didn't open a new tab/popup") as exc_info:
        adapter.open_external_application(posting)

    assert "page was at" in str(exc_info.value)
    assert "titled" in str(exc_info.value)


def test_open_external_application_returns_none_when_theres_no_external_button(playwright_page, monkeypatch):
    """FIXTURE_PATH is the ordinary Easy Apply fixture - no
    on-company-website button at all, the way a posting that turned out to
    be Easy Apply (or stopped accepting applications) would look.
    """
    real_goto = playwright_page.goto
    monkeypatch.setattr(playwright_page, "goto", lambda url, **kw: real_goto(f"file://{FIXTURE_PATH}"))
    adapter = LinkedInAdapter(playwright_page)
    posting = JobPosting(
        job_id="1", title="X", company="Y", url=f"file://{FIXTURE_PATH}", description="", easy_apply=False
    )

    assert adapter.open_external_application(posting) is None
