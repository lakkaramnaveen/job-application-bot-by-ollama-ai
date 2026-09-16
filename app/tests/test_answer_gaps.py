from job_bot.safety.answer_gaps import AnswerGapStore


def test_record_then_list_shows_the_gap(tmp_path):
    store = AnswerGapStore(tmp_path / "answer_gaps.json")

    store.record(
        "Are you comfortable commuting to this job's location?",
        job_id="1",
        company="Acme",
        title="Backend Engineer",
    )

    gaps = store.list_unanswered()
    assert "Are you comfortable commuting to this job's location?" in gaps
    entry = gaps["Are you comfortable commuting to this job's location?"]
    assert entry["count"] == 1
    assert entry["example_company"] == "Acme"
    assert entry["example_title"] == "Backend Engineer"


def test_repeated_question_accumulates_a_count_instead_of_duplicating(tmp_path):
    """The same eligibility/sponsorship-style question is asked near-
    verbatim across many different postings - repeated occurrences must
    bump a count, not create a separate entry per posting, so review shows
    the questions actually worth answering first.
    """
    store = AnswerGapStore(tmp_path / "answer_gaps.json")
    question = "Are you comfortable commuting to this job's location?"

    store.record(question, job_id="1", company="Acme", title="Backend Engineer")
    store.record(question, job_id="2", company="Globex", title="Full Stack Engineer")
    store.record(question, job_id="3", company="Initech", title="Software Engineer")

    gaps = store.list_unanswered()
    assert len(gaps) == 1
    assert gaps[question]["count"] == 3
    # Most recent occurrence's details win.
    assert gaps[question]["example_company"] == "Initech"


def test_resolve_removes_the_gap(tmp_path):
    store = AnswerGapStore(tmp_path / "answer_gaps.json")
    question = "Are you comfortable commuting to this job's location?"
    store.record(question, job_id="1", company="Acme", title="Backend Engineer")

    store.resolve(question)

    assert store.list_unanswered() == {}


def test_resolve_of_a_question_never_recorded_is_a_no_op(tmp_path):
    store = AnswerGapStore(tmp_path / "answer_gaps.json")
    store.resolve("Never recorded")  # must not raise
    assert store.list_unanswered() == {}


def test_list_unanswered_empty_when_no_file_exists_yet(tmp_path):
    store = AnswerGapStore(tmp_path / "answer_gaps.json")
    assert store.list_unanswered() == {}


def test_survives_a_corrupted_file(tmp_path):
    path = tmp_path / "answer_gaps.json"
    path.write_text("not json at all", encoding="utf-8")
    store = AnswerGapStore(path)

    assert store.list_unanswered() == {}
