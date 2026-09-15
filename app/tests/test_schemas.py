import pytest
from pydantic import ValidationError

from job_bot.models.schemas import ApplicationAnswer, EmailClassification, JobMatchScore


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


def test_application_answer_rejects_a_negative_confidence():
    with pytest.raises(ValidationError):
        ApplicationAnswer(answer="Yes", confidence=-0.1, based_on_resume=True)


def test_application_answer_accepts_valid_data():
    answer = ApplicationAnswer(answer="5 years", confidence=0.9, based_on_resume=True)
    assert answer.confidence == 0.9


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
