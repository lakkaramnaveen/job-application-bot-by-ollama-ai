import pytest
from pydantic import ValidationError

from job_bot.models.schemas import ApplicationAnswer, JobMatchScore


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


def test_application_answer_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        ApplicationAnswer(answer="Yes", confidence=1.5, based_on_resume=True)


def test_application_answer_accepts_valid_data():
    answer = ApplicationAnswer(answer="5 years", confidence=0.9, based_on_resume=True)
    assert answer.confidence == 0.9
