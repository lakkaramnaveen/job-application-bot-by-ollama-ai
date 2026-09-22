import sqlite3

import pytest

from job_bot.tracker.db import InvalidSort, InvalidStatus, Tracker


def make_tracker(tmp_path) -> Tracker:
    return Tracker(tmp_path / "db.sqlite3")


def test_upsert_then_has_applied_is_false_until_marked(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1", match_score=80)

    assert tracker.has_applied("1") is False

    tracker.mark_applied("1")

    assert tracker.has_applied("1") is True


def test_has_applied_stays_true_after_status_moves_on(tmp_path):
    """has_applied() must key off applied_at, not the current status - once
    a real submission happened, no later status change (a legitimate
    progression to "interviewing"/"offer", or a manual correction via
    `job-bot status` or the dashboard) may make the run loop's dedup check
    (`if tracker.has_applied(...): continue`) forget that and let a second
    real application through for the same job.
    """
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1")
    tracker.mark_applied("1")

    tracker.update_status("1", "interviewing")
    assert tracker.has_applied("1") is True

    tracker.update_status("1", "seen")
    assert tracker.has_applied("1") is True


def test_mark_applied_raises_for_unknown_job_id(tmp_path):
    tracker = make_tracker(tmp_path)

    with pytest.raises(ValueError, match="No tracked job"):
        tracker.mark_applied("does-not-exist")


def test_mark_skipped_sets_status(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1")

    tracker.mark_skipped("1")

    assert tracker.has_applied("1") is False
    assert tracker.status_counts() == {"skipped": 1}


def test_record_score_should_apply_sets_seen_status_and_score(tmp_path):
    tracker = make_tracker(tmp_path)

    tracker.record_score("1", "Engineer", "Acme", "https://example.com/1", score=85, should_apply=True)

    job = tracker.get_job("1")
    assert job["status"] == "seen"
    assert job["match_score"] == 85


def test_record_score_not_should_apply_sets_skipped_status(tmp_path):
    tracker = make_tracker(tmp_path)

    tracker.record_score("1", "Engineer", "Acme", "https://example.com/1", score=20, should_apply=False)

    job = tracker.get_job("1")
    assert job["status"] == "skipped"
    assert job["match_score"] == 20


def test_record_score_on_existing_job_updates_score_and_status(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.record_score("1", "Engineer", "Acme", "https://example.com/1", score=20, should_apply=False)

    tracker.record_score("1", "Engineer", "Acme", "https://example.com/1", score=90, should_apply=True)

    job = tracker.get_job("1")
    assert job["status"] == "seen"
    assert job["match_score"] == 90


def test_record_score_persists_the_llms_reasoning(tmp_path):
    """reasoning (JobMatchScore.reasoning, models/schemas.py) is otherwise
    computed on every single job and thrown away the moment record_score()
    returns - persisting it is what lets `job-bot status <job_id>` show
    *why* a job got the score/skip decision it did.
    """
    tracker = make_tracker(tmp_path)

    tracker.record_score(
        "1", "Engineer", "Acme", "https://example.com/1", score=85, should_apply=True, reasoning="Great fit"
    )

    assert tracker.get_job("1")["match_reasoning"] == "Great fit"


def test_record_score_defaults_reasoning_to_empty_string(tmp_path):
    tracker = make_tracker(tmp_path)

    tracker.record_score("1", "Engineer", "Acme", "https://example.com/1", score=85, should_apply=True)

    assert tracker.get_job("1")["match_reasoning"] == ""


def test_record_score_updates_reasoning_on_an_existing_job(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.record_score(
        "1", "Engineer", "Acme", "https://example.com/1", score=20, should_apply=False, reasoning="Weak fit"
    )

    tracker.record_score(
        "1", "Engineer", "Acme", "https://example.com/1", score=90, should_apply=True, reasoning="Great fit"
    )

    assert tracker.get_job("1")["match_reasoning"] == "Great fit"


def test_match_reasoning_column_is_added_to_a_database_created_before_this_feature(tmp_path):
    """Same migration shape as test_notes_column_is_added_to_a_database_
    created_before_this_feature: an existing db.sqlite3 from before
    match_reasoning existed must get the column added (not recreated) the
    next time a Tracker is constructed against it, with existing rows kept.
    """
    db_path = tmp_path / "pre_existing.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            company TEXT NOT NULL,
            url TEXT NOT NULL,
            match_score INTEGER,
            status TEXT NOT NULL DEFAULT 'seen',
            first_seen_at TEXT NOT NULL,
            applied_at TEXT,
            notes TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO jobs (job_id, title, company, url, match_score, status, first_seen_at) "
        "VALUES ('1', 'Engineer', 'Acme', 'https://example.com/1', 85, 'seen', '2024-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()

    tracker = Tracker(db_path)

    job = tracker.get_job("1")
    assert job["title"] == "Engineer"
    assert job["match_reasoning"] is None
    tracker.record_score("1", "Engineer", "Acme", "https://example.com/1", score=90, should_apply=True, reasoning="ok")
    assert tracker.get_job("1")["match_reasoning"] == "ok"


def test_record_score_persists_the_eligibility_verdict_and_note(tmp_path):
    """eligibility/eligibility_note (JobMatchScore, models/schemas.py) are
    the other half of what record_score() used to discard - status=skipped
    alone can't distinguish a categorical eligibility-gate rejection from a
    plain low score, and reasoning alone doesn't carry the specific quoted
    posting language driving an eligibility verdict.
    """
    tracker = make_tracker(tmp_path)

    tracker.record_score(
        "1",
        "Engineer",
        "Acme",
        "https://example.com/1",
        score=20,
        should_apply=False,
        eligibility="fail",
        eligibility_note="Requires active US security clearance.",
    )

    job = tracker.get_job("1")
    assert job["eligibility"] == "fail"
    assert job["eligibility_note"] == "Requires active US security clearance."


def test_record_score_defaults_eligibility_and_note_to_empty_string(tmp_path):
    tracker = make_tracker(tmp_path)

    tracker.record_score("1", "Engineer", "Acme", "https://example.com/1", score=85, should_apply=True)

    job = tracker.get_job("1")
    assert job["eligibility"] == ""
    assert job["eligibility_note"] == ""


def test_eligibility_columns_are_added_to_a_database_created_before_this_feature(tmp_path):
    """Same migration shape as test_match_reasoning_column_is_added_to_a_
    database_created_before_this_feature above, for the two eligibility
    columns added alongside it.
    """
    db_path = tmp_path / "pre_existing.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            company TEXT NOT NULL,
            url TEXT NOT NULL,
            match_score INTEGER,
            status TEXT NOT NULL DEFAULT 'seen',
            first_seen_at TEXT NOT NULL,
            applied_at TEXT,
            notes TEXT,
            match_reasoning TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO jobs (job_id, title, company, url, match_score, status, first_seen_at) "
        "VALUES ('1', 'Engineer', 'Acme', 'https://example.com/1', 85, 'seen', '2024-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()

    tracker = Tracker(db_path)

    job = tracker.get_job("1")
    assert job["title"] == "Engineer"
    assert job["eligibility"] is None
    assert job["eligibility_note"] is None
    tracker.record_score(
        "1", "Engineer", "Acme", "https://example.com/1", score=90, should_apply=True, eligibility="pass"
    )
    assert tracker.get_job("1")["eligibility"] == "pass"


def test_update_status_accepts_valid_outcome_status(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1")
    tracker.mark_applied("1")

    tracker.update_status("1", "interviewing")

    assert tracker.status_counts() == {"interviewing": 1}


def test_update_status_to_applied_stamps_applied_at(tmp_path):
    """A job moved to "applied" via update_status() (gmail_sync's
    application_confirmation match, or a manual `job-bot status <id>
    applied`) never went through mark_applied(), which is the only other
    place applied_at gets set. Leaving it null here would make
    has_applied() return False for a job the tracker itself says is
    applied - and the run loop's dedup check, and report --stale-days's
    follow-up nudge, both key off applied_at rather than status.
    """
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1")

    tracker.update_status("1", "applied")

    job = tracker.get_job("1")
    assert job["status"] == "applied"
    assert job["applied_at"] is not None
    assert tracker.has_applied("1") is True


def test_update_status_to_applied_does_not_overwrite_an_existing_applied_at(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1")
    tracker.mark_applied("1")
    original_applied_at = tracker.get_job("1")["applied_at"]

    tracker.update_status("1", "applied")

    assert tracker.get_job("1")["applied_at"] == original_applied_at


def test_update_status_rejects_unknown_status(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1")

    with pytest.raises(InvalidStatus, match="Unknown status"):
        tracker.update_status("1", "ghosted")


def test_update_status_rejects_unknown_job_id(tmp_path):
    tracker = make_tracker(tmp_path)

    with pytest.raises(ValueError, match="No tracked job"):
        tracker.update_status("does-not-exist", "offer")


def test_set_note_persists_and_is_returned_by_get_job(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1")

    tracker.set_note("1", "Recruiter mentioned a $150k base.")

    assert tracker.get_job("1")["notes"] == "Recruiter mentioned a $150k base."


def test_set_note_on_a_job_with_no_note_yet_defaults_to_none(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1")

    assert tracker.get_job("1")["notes"] is None


def test_set_note_overwrites_an_existing_note(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1")
    tracker.set_note("1", "First note.")

    tracker.set_note("1", "Updated note.")

    assert tracker.get_job("1")["notes"] == "Updated note."


def test_set_note_to_empty_string_clears_it(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1")
    tracker.set_note("1", "A note.")

    tracker.set_note("1", "")

    assert tracker.get_job("1")["notes"] == ""


def test_set_note_rejects_unknown_job_id(tmp_path):
    tracker = make_tracker(tmp_path)

    with pytest.raises(ValueError, match="No tracked job"):
        tracker.set_note("does-not-exist", "A note.")


def test_notes_column_is_added_to_a_database_created_before_this_feature(tmp_path):
    """Real migration this guards against: a db.sqlite3 created by a
    version of this project before `notes` existed has a `jobs` table with
    no such column - opening it with a Tracker that expects one must add
    the column (via ALTER TABLE, in _init_db()) rather than silently fail
    or lose the pre-existing table's rows.
    """
    db_path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            company TEXT NOT NULL,
            url TEXT NOT NULL,
            match_score INTEGER,
            status TEXT NOT NULL DEFAULT 'seen',
            first_seen_at TEXT NOT NULL,
            applied_at TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO jobs (job_id, title, company, url, first_seen_at) VALUES (?, ?, ?, ?, ?)",
        ("1", "Engineer", "Acme", "https://example.com/1", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    tracker = Tracker(db_path)

    job = tracker.get_job("1")
    assert job["title"] == "Engineer"  # the pre-existing row survived
    assert job["notes"] is None
    tracker.set_note("1", "Works now.")
    assert tracker.get_job("1")["notes"] == "Works now."


def test_status_counts_reflects_multiple_jobs(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "A", "Acme", "https://example.com/1")
    tracker.upsert_job("2", "B", "Acme", "https://example.com/2")
    tracker.upsert_job("3", "C", "Acme", "https://example.com/3")
    tracker.mark_applied("1")
    tracker.mark_applied("2")
    tracker.mark_skipped("3")

    assert tracker.status_counts() == {"applied": 2, "skipped": 1}


def test_status_counts_can_be_scoped_to_a_search_term(tmp_path):
    """The dashboard's stat pills call this with the current search box
    text so the counts they show match what's actually visible in the
    table - not whole-database totals unrelated to what the user is
    looking at.
    """
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Backend Engineer", "Acme", "https://example.com/1")
    tracker.upsert_job("2", "Frontend Engineer", "Acme", "https://example.com/2")
    tracker.upsert_job("3", "Data Scientist", "Acme", "https://example.com/3")
    tracker.mark_applied("1")
    tracker.mark_applied("2")

    assert tracker.status_counts(search="engineer") == {"applied": 2}
    assert tracker.status_counts(search="scientist") == {"seen": 1}
    assert tracker.status_counts(search="nonexistent") == {}


def test_record_qa_and_upsert_job_do_not_conflict(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1")

    tracker.record_qa("1", "Years of experience?", "5")
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1", match_score=90)

    assert tracker.status_counts() == {"seen": 1}


def test_list_qa_returns_chronological_transcript(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1")

    tracker.record_qa("1", "Years of experience?", "5")
    tracker.record_qa("1", "Willing to relocate?", "No")

    qa = tracker.list_qa("1")

    assert [entry["question"] for entry in qa] == ["Years of experience?", "Willing to relocate?"]
    assert qa[0]["answer"] == "5"


def test_list_qa_empty_for_job_with_no_questions(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1")

    assert tracker.list_qa("1") == []


def test_recent_qa_pairs_returns_most_recent_first(tmp_path):
    tracker = make_tracker(tmp_path)

    tracker.record_qa("1", "Years of experience?", "5")
    tracker.record_qa("2", "Willing to relocate?", "No")

    pairs = tracker.recent_qa_pairs()

    assert pairs[0] == {"question": "Willing to relocate?", "answer": "No"}
    assert pairs[1] == {"question": "Years of experience?", "answer": "5"}


def test_recent_qa_pairs_dedupes_by_question_keeping_the_latest_answer(tmp_path):
    """The same question is commonly asked across many different postings -
    without dedup, one frequently-recurring question would crowd out every
    other question's answer from the (size-limited) reference list.
    """
    tracker = make_tracker(tmp_path)

    tracker.record_qa("1", "Years of experience?", "4")
    tracker.record_qa("2", "Willing to relocate?", "No")
    tracker.record_qa("3", "Years of experience?", "5")  # more recent, different answer

    pairs = tracker.recent_qa_pairs()

    assert len(pairs) == 2
    years_answer = next(p["answer"] for p in pairs if p["question"] == "Years of experience?")
    assert years_answer == "5"


def test_recent_qa_pairs_respects_the_limit(tmp_path):
    tracker = make_tracker(tmp_path)
    for i in range(5):
        tracker.record_qa(str(i), f"Question {i}?", f"Answer {i}")

    pairs = tracker.recent_qa_pairs(limit=2)

    assert len(pairs) == 2
    assert pairs[0]["question"] == "Question 4?"
    assert pairs[1]["question"] == "Question 3?"


def test_recent_qa_pairs_empty_when_nothing_recorded(tmp_path):
    tracker = make_tracker(tmp_path)
    assert tracker.recent_qa_pairs() == []


def test_list_jobs_filters_by_status(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1")
    tracker.upsert_job("2", "Designer", "Acme", "https://example.com/2")
    tracker.mark_applied("1")

    applied = tracker.list_jobs(status="applied")

    assert [j["job_id"] for j in applied] == ["1"]


def test_list_jobs_search_matches_title_or_company_case_insensitively(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Backend Engineer", "Acme", "https://example.com/1")
    tracker.upsert_job("2", "Designer", "Widgets Inc", "https://example.com/2")

    by_title = tracker.list_jobs(search="engineer")
    by_company = tracker.list_jobs(search="widgets")

    assert [j["job_id"] for j in by_title] == ["1"]
    assert [j["job_id"] for j in by_company] == ["2"]


def test_list_jobs_search_escapes_like_wildcards(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "50% Time Role", "Acme", "https://example.com/1")
    tracker.upsert_job("2", "Full Time Role", "Acme", "https://example.com/2")

    results = tracker.list_jobs(search="50%")

    assert [j["job_id"] for j in results] == ["1"]


def test_list_jobs_search_also_matches_notes(tmp_path):
    """Real gap this guards against: a note is exactly the kind of
    free-text context ("Referred by Jane") someone would later search for
    without remembering which job it was attached to, but _where_clause()
    only ever checked title/company.
    """
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Backend Engineer", "Acme", "https://example.com/1")
    tracker.upsert_job("2", "Designer", "Beta", "https://example.com/2")
    tracker.set_note("1", "Referred by Jane, mentioned $150k base.")

    results = tracker.list_jobs(search="Jane")

    assert [j["job_id"] for j in results] == ["1"]


def test_count_jobs_search_also_matches_notes(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Backend Engineer", "Acme", "https://example.com/1")
    tracker.set_note("1", "Referred by Jane.")

    assert tracker.count_jobs(search="Jane") == 1
    assert tracker.count_jobs(search="Nobody") == 0


def test_list_jobs_search_also_matches_match_reasoning(tmp_path):
    """Real gap this guards against: match_reasoning (Tracker.record_score())
    is exactly the kind of free-text context someone would later search
    for without remembering which job it was attached to - e.g. a
    specific technology mentioned in the LLM's own scoring explanation.
    """
    tracker = make_tracker(tmp_path)
    tracker.record_score(
        "1", "Backend Engineer", "Acme", "https://example.com/1", score=90, should_apply=True,
        reasoning="Strong Python and AWS overlap with the posting.",
    )
    tracker.record_score(
        "2", "Designer", "Beta", "https://example.com/2", score=40, should_apply=False,
        reasoning="Weak fit, mostly design-focused role.",
    )

    results = tracker.list_jobs(search="AWS")

    assert [j["job_id"] for j in results] == ["1"]


def test_list_jobs_search_also_matches_eligibility_note(tmp_path):
    """Real gap this guards against: eligibility_note quotes the specific
    posting wording driving an eligibility verdict (e.g. "Requires US
    citizenship") - exactly the kind of thing someone would search for to
    find every job the eligibility gate flagged over the same rule.
    """
    tracker = make_tracker(tmp_path)
    tracker.record_score(
        "1", "Backend Engineer", "Acme", "https://example.com/1", score=20, should_apply=False,
        eligibility="fail", eligibility_note="Requires active US security clearance.",
    )
    tracker.record_score(
        "2", "Designer", "Beta", "https://example.com/2", score=85, should_apply=True, eligibility="pass",
    )

    results = tracker.list_jobs(search="clearance")

    assert [j["job_id"] for j in results] == ["1"]


def test_count_jobs_search_also_matches_match_reasoning_and_eligibility_note(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.record_score(
        "1", "Backend Engineer", "Acme", "https://example.com/1", score=90, should_apply=True,
        reasoning="Strong Python overlap.", eligibility="fail", eligibility_note="Requires US citizenship.",
    )

    assert tracker.count_jobs(search="Python") == 1
    assert tracker.count_jobs(search="citizenship") == 1
    assert tracker.count_jobs(search="Nobody") == 0


def test_list_jobs_sorts_by_match_score(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "A", "Acme", "https://example.com/1", match_score=40)
    tracker.upsert_job("2", "B", "Acme", "https://example.com/2", match_score=90)

    ranked = tracker.list_jobs(sort="match_score", direction="desc")

    assert [j["job_id"] for j in ranked] == ["2", "1"]


def test_list_jobs_paginates_with_limit_and_offset(tmp_path):
    tracker = make_tracker(tmp_path)
    for i in range(5):
        tracker.upsert_job(str(i), f"Job {i}", "Acme", f"https://example.com/{i}")

    page1 = tracker.list_jobs(sort="title", direction="asc", limit=2, offset=0)
    page2 = tracker.list_jobs(sort="title", direction="asc", limit=2, offset=2)

    assert [j["job_id"] for j in page1] == ["0", "1"]
    assert [j["job_id"] for j in page2] == ["2", "3"]


def test_list_jobs_rejects_unknown_sort_column(tmp_path):
    tracker = make_tracker(tmp_path)

    with pytest.raises(InvalidSort):
        tracker.list_jobs(sort="job_id; DROP TABLE jobs;--")


def test_list_jobs_rejects_unknown_direction(tmp_path):
    tracker = make_tracker(tmp_path)

    with pytest.raises(InvalidSort):
        tracker.list_jobs(direction="sideways")


def test_count_jobs_reflects_filters(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1")
    tracker.upsert_job("2", "Designer", "Acme", "https://example.com/2")
    tracker.mark_applied("1")

    assert tracker.count_jobs() == 2
    assert tracker.count_jobs(status="applied") == 1
    assert tracker.count_jobs(search="designer") == 1


def test_record_resume_generation_round_trips_through_best_resume_examples(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.record_resume_generation(
        "1", "Engineer", "Acme", "A tailored summary.", ["Python", "AWS"], ["Did a thing."]
    )

    examples = tracker.best_resume_examples(limit=5)

    assert len(examples) == 1
    assert examples[0]["job_id"] == "1"
    assert examples[0]["summary"] == "A tailored summary."
    assert examples[0]["skills"] == ["Python", "AWS"]
    assert examples[0]["bullets"] == ["Did a thing."]


def test_best_resume_examples_prioritizes_jobs_with_a_positive_outcome(tmp_path):
    """A generation tied to a job later marked "interviewing"/"offer" - a
    real, human-confirmed sign that resume helped - should be preferred
    over a more recent generation with no such signal, even though
    best_resume_examples() otherwise orders by recency.
    """
    tracker = make_tracker(tmp_path)
    tracker.upsert_job("1", "Engineer", "Acme", "https://example.com/1")
    tracker.upsert_job("2", "Designer", "Beta", "https://example.com/2")
    tracker.record_resume_generation("1", "Engineer", "Acme", "Older, but landed an interview.", [], [])
    tracker.record_resume_generation("2", "Designer", "Beta", "Newer, no outcome yet.", [], [])
    tracker.update_status("1", "interviewing")

    examples = tracker.best_resume_examples(limit=1)

    assert examples[0]["job_id"] == "1"


def test_best_resume_examples_falls_back_to_recency_with_no_outcomes(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.record_resume_generation("1", "Engineer", "Acme", "Older.", [], [])
    tracker.record_resume_generation("2", "Designer", "Beta", "Newer.", [], [])

    examples = tracker.best_resume_examples(limit=1)

    assert examples[0]["job_id"] == "2"


def test_best_resume_examples_empty_when_nothing_recorded(tmp_path):
    tracker = make_tracker(tmp_path)
    assert tracker.best_resume_examples() == []


def test_get_resume_generation_returns_the_recorded_generation(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.record_resume_generation(
        "1", "Engineer", "Acme", "A tailored summary.", ["Python", "AWS"], ["Did a thing."]
    )

    generation = tracker.get_resume_generation("1")

    assert generation["job_id"] == "1"
    assert generation["summary"] == "A tailored summary."
    assert generation["skills"] == ["Python", "AWS"]
    assert generation["bullets"] == ["Did a thing."]


def test_get_resume_generation_returns_none_when_never_generated(tmp_path):
    tracker = make_tracker(tmp_path)
    assert tracker.get_resume_generation("never-generated") is None


def test_get_resume_generation_returns_the_most_recent_of_several(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.record_resume_generation("1", "Engineer", "Acme", "First generation.", [], [])
    tracker.record_resume_generation("1", "Engineer", "Acme", "Second, more recent generation.", [], [])

    generation = tracker.get_resume_generation("1")

    assert generation["summary"] == "Second, more recent generation."


def test_get_resume_generation_only_returns_this_jobs_own_generation(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.record_resume_generation("1", "Engineer", "Acme", "For job 1.", [], [])
    tracker.record_resume_generation("2", "Designer", "Beta", "For job 2.", [], [])

    assert tracker.get_resume_generation("1")["summary"] == "For job 1."
    assert tracker.get_resume_generation("2")["summary"] == "For job 2."
