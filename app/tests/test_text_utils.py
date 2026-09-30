import pytest

from job_bot.text_utils import is_echoed_question, is_per_position_field, normalize_company_name


def test_normalize_company_name_strips_and_casefolds():
    assert normalize_company_name("  Acme Corp  ") == "acme corp"
    assert normalize_company_name("ACME CORP") == "acme corp"


def test_normalize_company_name_collapses_internal_whitespace():
    assert normalize_company_name("Acme   Corp") == "acme corp"
    assert normalize_company_name("Acme\tCorp\n") == "acme corp"


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("Phone country code", "Phone country code"),
        ("Year of From", "year of from"),
        ("LinkedIn", "  LinkedIn "),
        ("Willing to relocate?", "Willing to relocate"),
        ("First name*", "First  name"),
    ],
)
def test_is_echoed_question_catches_the_label_repeated_back(question, answer):
    assert is_echoed_question(question, answer) is True


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("Phone country code", "United States (+1)"),
        ("Year of From", "2019"),
        ("Willing to relocate?", "Yes"),
        ("Willing to relocate?", ""),  # no answer at all is not an echo
        ("?", "?"),  # normalizes to empty - not an echo either
    ],
)
def test_is_echoed_question_leaves_real_or_empty_answers_alone(question, answer):
    assert is_echoed_question(question, answer) is False


@pytest.mark.parametrize("label", ["Year of From", "Month of From", "year of to", " Month of To* "])
def test_is_per_position_field_matches_linkedins_work_history_date_labels(label):
    assert is_per_position_field(label) is True


@pytest.mark.parametrize(
    "label",
    ["Years of experience?", "From which city are you relocating?", "Year of graduation", "Start date", ""],
)
def test_is_per_position_field_leaves_other_questions_alone(label):
    assert is_per_position_field(label) is False
