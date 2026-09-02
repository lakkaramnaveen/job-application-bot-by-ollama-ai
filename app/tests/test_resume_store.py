import json

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
