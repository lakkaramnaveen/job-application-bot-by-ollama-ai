"""Exercises the LinkedIn adapter's form-filling logic against a local static
HTML fixture that mimics the accessible structure of the Easy Apply modal
(role="dialog", label/input pairs, a fieldset radio group, a select, and an
aria-labeled submit button). This never touches the real linkedin.com - it
verifies the field-detection and answer-matching logic in isolation.
"""

from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from job_bot.browser.base_adapter import JobPosting
from job_bot.browser.linkedin_adapter import LinkedInAdapter

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "easy_apply_form.html"
SEARCH_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "search_results.html"
RESUME_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_resume.txt"


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
