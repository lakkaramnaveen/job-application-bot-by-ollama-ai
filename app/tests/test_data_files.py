import json

import pytest

from job_bot.data_files import CorruptDataFile, assert_safe_to_overwrite
from job_bot.resume.store import ResumeStore
from job_bot.safety.answer_gaps import AnswerGapStore
from job_bot.safety.blacklist import CompanyBlacklist

CORRUPT = b'{"Willing to relocate?": "No", "Years of Python?": "5"'  # truncated - a bad hand edit


def test_a_missing_file_is_safe_to_create(tmp_path):
    assert_safe_to_overwrite(tmp_path / "absent.json", dict)


def test_a_valid_file_of_the_expected_type_is_safe(tmp_path):
    path = tmp_path / "f.json"
    path.write_text(json.dumps({"a": "b"}), encoding="utf-8")
    assert_safe_to_overwrite(path, dict)


@pytest.mark.parametrize(
    ("content", "expected_type"),
    [
        (CORRUPT, dict),
        (b"\xff\xfe not utf-8", dict),
        (json.dumps(["a list"]).encode(), dict),
        (json.dumps({"a": "dict"}).encode(), list),
    ],
)
def test_an_unreadable_or_wrong_shaped_file_is_refused(tmp_path, content, expected_type):
    path = tmp_path / "f.json"
    path.write_bytes(content)
    with pytest.raises(CorruptDataFile, match="refusing to overwrite"):
        assert_safe_to_overwrite(path, expected_type)


def test_save_faq_answer_never_replaces_an_unreadable_faq_file(tmp_path):
    """Real data-loss bug: faq_answers() loads a corrupt file as {}, and
    save_faq_answer() used to rebuild the file from that - one cached answer
    silently wiped every entry still recoverable by hand.
    """
    faq_path = tmp_path / "faq.json"
    faq_path.write_bytes(CORRUPT)

    with pytest.raises(CorruptDataFile):
        ResumeStore(tmp_path / "resume.txt", faq_path).save_faq_answer("New question?", "Yes")

    assert faq_path.read_bytes() == CORRUPT


def test_blacklist_add_never_replaces_an_unreadable_blacklist(tmp_path):
    path = tmp_path / "blacklist.json"
    path.write_bytes(b'["Acme", "Globex"')

    with pytest.raises(CorruptDataFile):
        CompanyBlacklist(path).add("Initech")

    assert path.read_bytes() == b'["Acme", "Globex"'


def test_answer_gap_record_never_replaces_an_unreadable_gaps_file(tmp_path):
    path = tmp_path / "answer_gaps.json"
    path.write_bytes(CORRUPT)

    with pytest.raises(CorruptDataFile):
        AnswerGapStore(path).record("Phone country code", job_id="1", company="Acme", title="SWE")

    assert path.read_bytes() == CORRUPT


def test_saving_to_a_healthy_file_still_works(tmp_path):
    faq_path = tmp_path / "faq.json"
    store = ResumeStore(tmp_path / "resume.txt", faq_path)
    store.save_faq_answer("Q1?", "A1")
    store.save_faq_answer("Q2?", "A2")
    assert json.loads(faq_path.read_text()) == {"Q1?": "A1", "Q2?": "A2"}


def test_a_directory_where_the_file_should_be_is_refused_not_crashed_on(tmp_path):
    """OSError (a directory in place, permission denied) used to escape as a
    raw IsADirectoryError/PermissionError traceback instead of the clean
    refusal.
    """
    path = tmp_path / "f.json"
    path.mkdir()
    with pytest.raises(CorruptDataFile, match="refusing to overwrite"):
        assert_safe_to_overwrite(path, dict)


def test_every_store_loads_an_unreadable_path_as_empty_instead_of_crashing(tmp_path):
    """Same graceful fallback as invalid JSON (doctor reports it) - before,
    faq_answers() raised on every question `job-bot run` asked.
    """
    for name in ("faq.json", "blacklist.json", "gaps.json"):
        (tmp_path / name).mkdir()

    assert ResumeStore(tmp_path / "resume.txt", tmp_path / "faq.json").faq_answers() == {}
    assert CompanyBlacklist(tmp_path / "blacklist.json").list_companies() == []
    assert AnswerGapStore(tmp_path / "gaps.json").list_unanswered() == {}
