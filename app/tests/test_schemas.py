import pytest
from pydantic import ValidationError

from job_bot.models.schemas import (
    ApplicationAnswer,
    CoverLetter,
    EmailClassification,
    JobMatchScore,
    TailoredResume,
)


def make_job_match_score(**overrides):
    defaults = dict(
        eligibility="pass",
        technical_fit=85,
        experience_fit=85,
        culture_fit=85,
        score=85,
        reasoning="Strong overlap",
        should_apply=True,
    )
    defaults.update(overrides)
    return JobMatchScore(**defaults)


def test_job_match_score_rejects_out_of_range_score():
    with pytest.raises(ValidationError):
        make_job_match_score(score=150)


def test_job_match_score_accepts_valid_data():
    score = make_job_match_score()
    assert score.missing_qualifications == []
    assert score.eligibility_note == ""


def test_job_match_score_rejects_invalid_eligibility_value():
    with pytest.raises(ValidationError):
        make_job_match_score(eligibility="maybe")


def test_eligibility_field_description_covers_all_active_eligibility_rules():
    """This description reaches the model as part of the JSON schema sent
    alongside scorer.py's system prompt (see ollama_provider.py's `format`)
    - it used to say eligibility='fail' means only citizenship/clearance,
    which became a real, live contradiction once matching/scorer.py's
    system prompt grew seniority/years-of-experience and W2-only checks:
    the model received two authoritative-sounding but disagreeing
    explanations of the same field in the same request.
    """
    description = JobMatchScore.model_json_schema()["properties"]["eligibility"]["description"]
    assert "citizenship" in description.casefold()
    assert "seniority" in description.casefold() or "years" in description.casefold()
    assert "w2" in description.casefold()


def test_application_answer_rejects_a_negative_confidence():
    with pytest.raises(ValidationError):
        ApplicationAnswer(answer="Yes", confidence=-0.1, based_on_resume=True)


def test_application_answer_accepts_valid_data():
    answer = ApplicationAnswer(answer="5 years", confidence=0.9, based_on_resume=True)
    assert answer.confidence == 0.9


@pytest.mark.parametrize(
    "leaked_reasoning",
    [
        "I need to answer the question about years of experience with Tailwind CSS based on the resume.",
        "Let me carefully check the resume for any mention of Golang experience before answering.",
        "Let me search the FAQ data for anything relevant to this question first.",
        "I should not fabricate any information, so let me look at the resume section by section.",
    ],
)
def test_application_answer_rejects_leaked_reasoning(leaked_reasoning):
    """Real bug this guards against: qwen3:30b (via Ollama) occasionally
    emits its internal chain-of-thought directly into the `answer` field
    instead of a real answer, truncated mid-thought once generation runs
    out of room before ever reaching a conclusion - seen live, 115 of
    1,644 recorded answers in one real user's qa_history were exactly
    this. Syntactically this is a perfectly valid string (a bare `str`
    field has nothing to reject it), but it's not an answer - and it was
    getting typed into a real application form field verbatim, then reused
    via Tracker.recent_qa_pairs() as "informal reference" for future
    questions, compounding the problem.
    """
    with pytest.raises(ValidationError, match="leaked reasoning"):
        ApplicationAnswer(answer=leaked_reasoning, confidence=0.5, based_on_resume=False)


def test_application_answer_does_not_reject_a_long_but_genuine_answer():
    """The leaked-reasoning check must not become so broad it rejects a
    real, if lengthy, explanatory answer - only strings matching this
    local model's own distinctive reasoning-trace phrasing.
    """
    genuine = (
        "No, I am currently located in St Louis, MO and would need to relocate for this role, "
        "though I am open to discussing relocation assistance."
    )
    answer = ApplicationAnswer(answer=genuine, confidence=0.8, based_on_resume=True)
    assert answer.answer == genuine


def test_cover_letter_rejects_leaked_reasoning_and_a_duplicate_draft():
    """Real, serious failure this guards against: an actually-submitted
    cover letter (qwen3:30b, via Ollama) contained the finished letter
    followed by the model's own self-review ("Let me check if I've
    included only facts from the resume: ... Let me count [paragraphs]
    ... I think this is good to go ... Final version:") and then a
    second, duplicate copy of the letter - sent to a real employer under
    the user's name. This is the actual contaminated text (company/
    contact details unchanged from what was really sent).
    """
    contaminated = (
        "Dear Hiring Manager,\n\n"
        "I am writing to express my interest in the Software Engineer position at Acme Corp.\n\n"
        "Sincerely,\nJane Doe\n\n"
        "Let me check if I've included only facts from the resume:\n"
        "- 5+ years of experience: Yes, mentioned in the summary.\n\n"
        "Let me count: 1. Introduction. 2. Body. 3. Closing.\n"
        "That's three paragraphs. I think this is good to go.\n\n"
        "Final version:\n\n"
        "Dear Hiring Manager,\n\nI am writing to express my interest..."
    )
    with pytest.raises(ValidationError, match="leaked reasoning"):
        CoverLetter(body=contaminated)


def test_cover_letter_accepts_a_genuine_multi_paragraph_letter():
    genuine = (
        "Dear Hiring Manager,\n\n"
        "I am writing to express my interest in the Software Engineer position at Acme Corp. "
        "With over five years of experience building Java backend services, I have a proven "
        "track record of delivering reliable, high-performance systems.\n\n"
        "Sincerely,\nJane Doe"
    )
    letter = CoverLetter(body=genuine)
    assert letter.body == genuine


def test_tailored_resume_summary_rejects_leaked_reasoning():
    contaminated = (
        "Let me carefully check the resume for the most relevant experience before writing "
        "the summary, focusing on backend development and cloud infrastructure."
    )
    with pytest.raises(ValidationError, match="leaked reasoning"):
        TailoredResume(summary=contaminated, highlighted_skills=["Java"], bullet_points=["Shipped X"])


def test_tailored_resume_summary_accepts_a_genuine_summary():
    genuine = (
        "Software Engineer with 5+ years of experience building and maintaining Java backend "
        "services, with a focus on microservices architecture and CI/CD automation."
    )
    resume = TailoredResume(summary=genuine, highlighted_skills=["Java"], bullet_points=["Shipped X"])
    assert resume.summary == genuine


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (100, 1.0),
        (85, 0.85),
        (1.5, 0.015),
        (500, 1.0),  # clamped: 500 has no sensible fraction, so this floors at 1.0
    ],
)
def test_application_answer_normalizes_a_percent_scale_confidence(raw, expected):
    """Real failure this guards against: llama3.1:8b (via Ollama) answered
    confidence=100 for a real application question, which used to hard-fail
    pydantic's le=1.0 check and abort the whole application attempt over one
    out-of-range field - despite both the prompt and the JSON schema sent to
    the model stating the 0.0-1.0 range.
    """
    answer = ApplicationAnswer(answer="Yes", confidence=raw, based_on_resume=True)
    assert answer.confidence == pytest.approx(expected)


def test_email_classification_normalizes_a_percent_scale_confidence():
    classification = EmailClassification(is_job_related=True, category="offer", confidence=95)
    assert classification.confidence == pytest.approx(0.95)


def test_email_classification_rejects_a_negative_confidence():
    with pytest.raises(ValidationError):
        EmailClassification(is_job_related=True, category="offer", confidence=-0.1)
