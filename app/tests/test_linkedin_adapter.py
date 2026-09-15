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
from job_bot.browser.linkedin_adapter import RESULTS_PER_PAGE, LinkedInAdapter

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
REQUIRED_RADIO_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "easy_apply_form_required_radio.html"
REQUIRED_SELECT_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "easy_apply_form_required_select.html"


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
    and the search must end rather than loop to MAX_SEARCH_PAGES.
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
    # Page 1 collects them, page 2 repeats them and ends the search.
    assert len(page_loads) == 2
