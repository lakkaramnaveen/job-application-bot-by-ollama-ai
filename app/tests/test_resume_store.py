import json
import os

from job_bot.resume.store import ResumeStore


def make_store(tmp_path, resume_text="Experienced Python developer.") -> ResumeStore:
    resume_path = tmp_path / "resume.txt"
    resume_path.write_text(resume_text, encoding="utf-8")
    return ResumeStore(resume_path, tmp_path / "faq_answers.json")


def test_resume_text_reads_and_caches(tmp_path):
    store = make_store(tmp_path, "Experienced Python developer.")

    assert store.resume_text() == "Experienced Python developer."
    # Second call must not need the file anymore - prove caching by removing it.
    store._resume_path.unlink()
    assert store.resume_text() == "Experienced Python developer."


def test_resume_text_picks_up_an_edited_file_without_a_new_store(tmp_path):
    """Real bug this guards against: `job-bot run --loop` constructs one
    ResumeStore and holds it for the whole multi-hour session - the plain
    "cache forever" version above meant a resume edited and re-exported
    mid-loop (e.g. correcting a typo, updating contact info) silently kept
    using the stale version already in memory until the process was
    restarted, with no error or warning.
    """
    store = make_store(tmp_path, "Original resume content")
    assert store.resume_text() == "Original resume content"

    resume_path = store._resume_path
    resume_path.write_text("Updated resume content", encoding="utf-8")
    # Set an explicit, distinct mtime rather than sleeping - some
    # filesystems have coarse (1s) mtime resolution, and a real sleep in a
    # test is slow and still not fully deterministic.
    new_mtime = (resume_path.stat().st_mtime or 0) + 5
    os.utime(resume_path, (new_mtime, new_mtime))

    assert store.resume_text() == "Updated resume content"


def test_resume_text_survives_a_transient_stat_failure_by_keeping_the_cache(tmp_path, monkeypatch):
    store = make_store(tmp_path, "Original resume content")
    assert store.resume_text() == "Original resume content"

    def raise_oserror(*args, **kwargs):
        raise OSError("transient failure")

    monkeypatch.setattr(type(store._resume_path), "stat", raise_oserror)

    assert store.resume_text() == "Original resume content"


def test_faq_answers_missing_file_returns_empty_dict(tmp_path):
    store = make_store(tmp_path)
    assert store.faq_answers() == {}


def test_faq_answers_loads_existing_file(tmp_path):
    store = make_store(tmp_path)
    store._faq_path.write_text(json.dumps({"Years of experience?": "5"}), encoding="utf-8")

    assert store.faq_answers() == {"Years of experience?": "5"}


def test_faq_answers_malformed_json_returns_empty_dict(tmp_path):
    store = make_store(tmp_path)
    store._faq_path.write_text("not valid json {{{", encoding="utf-8")

    assert store.faq_answers() == {}


def test_faq_answers_non_dict_json_returns_empty_dict(tmp_path):
    store = make_store(tmp_path)
    store._faq_path.write_text(json.dumps(["a", "list", "not", "a", "dict"]), encoding="utf-8")

    assert store.faq_answers() == {}


def test_save_faq_answer_persists_and_merges(tmp_path):
    store = make_store(tmp_path)

    store.save_faq_answer("Years of experience?", "5")
    store.save_faq_answer("Willing to relocate?", "No")

    assert store.faq_answers() == {"Years of experience?": "5", "Willing to relocate?": "No"}


def test_save_faq_answer_overwrites_same_question(tmp_path):
    store = make_store(tmp_path)

    store.save_faq_answer("Years of experience?", "5")
    store.save_faq_answer("Years of experience?", "6")

    assert store.faq_answers() == {"Years of experience?": "6"}


def test_save_faq_answer_creates_parent_directory(tmp_path):
    resume_path = tmp_path / "resume.txt"
    resume_path.write_text("resume", encoding="utf-8")
    nested_faq_path = tmp_path / "nested" / "dir" / "faq.json"
    store = ResumeStore(resume_path, nested_faq_path)

    store.save_faq_answer("Q", "A")

    assert nested_faq_path.exists()


def test_save_faq_answer_visible_to_a_fresh_store_instance(tmp_path):
    store1 = make_store(tmp_path)
    store1.save_faq_answer("Years of experience?", "5")

    store2 = ResumeStore(store1._resume_path, store1._faq_path)
    assert store2.faq_answers() == {"Years of experience?": "5"}
