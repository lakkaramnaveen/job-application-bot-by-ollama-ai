"""AnswerService: the answer lifecycle for one form question - cache
lookup, model call, what may be cached, and learning from LinkedIn's
rejections. See job_bot/pipeline/answers.py."""

import json

import pytest

from job_bot.browser.base_adapter import JobPosting
from job_bot.browser.linkedin_adapter import FieldsRejected
from job_bot.models.schemas import ApplicationAnswer
from job_bot.pipeline.answers import AnswerService
from job_bot.resume.store import ResumeStore
from job_bot.safety.answer_gaps import AnswerGapStore
from job_bot.tracker.db import Tracker

POSTING = JobPosting(job_id="j1", title="Engineer", company="Acme", url="https://example.com/j1", description="")


@pytest.fixture
def env(tmp_path):
    resume = tmp_path / "resume.txt"
    resume.write_text("Backend engineer, 5 years of Python.", encoding="utf-8")
    faq = tmp_path / "faq.json"
    tracker = Tracker(tmp_path / "t.sqlite3")
    tracker.upsert_job("j1", "Engineer", "Acme", "https://example.com/j1")
    calls, notes = [], []

    def make(answer="5", confidence=0.9, based_on_resume=True, faq_contents=None):
        if faq_contents is not None:
            faq.write_text(json.dumps(faq_contents), encoding="utf-8")

        def fake_model(provider, resume_text, faq_answers, question, recent_answers=None):
            calls.append(question)
            return ApplicationAnswer(answer=answer, confidence=confidence, based_on_resume=based_on_resume)

        service = AnswerService(
            provider=None,
            resume_text=resume.read_text(),
            resume_store=ResumeStore(resume, faq),
            tracker=tracker,
            answer_gaps=AnswerGapStore(tmp_path / "gaps.json"),
            save_confidence=0.7,
            answer_fn=fake_model,
            notify=notes.append,
        )
        return service

    def faq_now():
        return json.loads(faq.read_text()) if faq.exists() else {}

    return make, calls, notes, faq_now, tmp_path


def test_an_exact_faq_match_is_used_without_calling_the_model(env):
    make, calls, _, _, _ = env
    service = make(faq_contents={"Are you authorized to work in the US?": "Yes"})
    assert service.answer("Are you authorized to work in the US?", "j1") == "Yes"
    assert calls == []


def test_a_new_question_asks_the_model_and_caches_a_confident_answer_in_shape(env):
    make, calls, _, faq_now, _ = env
    service = make(answer="5+ years")
    question = "How many years of work experience do you have with Python?"
    assert service.answer(question, "j1") == "5+ years"  # the fill step shapes it for the field
    assert calls == [question]
    assert faq_now() == {question: "5"}  # cached in the shape the question asks for


def test_a_low_confidence_or_ungrounded_answer_is_used_but_not_cached(env):
    make, _, _, faq_now, _ = env
    assert make(answer="Maybe", confidence=0.3).answer("Willing to travel?", "j1") == "Maybe"
    assert make(answer="Maybe", based_on_resume=False).answer("Willing to relocate?", "j1") == "Maybe"
    assert faq_now() == {}


def test_an_echoed_answer_is_no_answer(env):
    make, _, _, faq_now, _ = env
    question = "Years of experience with Kotlin"
    assert make(answer=question).answer(question, "j1") == ""
    assert faq_now() == {}


def test_a_per_position_field_is_never_answered(env):
    """Work-history date fields ("Year of From") differ per position - one
    cached answer would be wrong for every other entry."""
    make, calls, _, faq_now, _ = env
    assert make().answer("Year of From", "j1") == ""
    assert calls == [] and faq_now() == {}


def test_a_rejected_answer_is_dropped_from_the_cache_and_queued_for_review(env):
    make, _, notes, faq_now, tmp_path = env
    question = "How many years of work experience do you have with Java?"
    service = make(faq_contents={question: "5+ years", "Are you authorized to work in the US?": "Yes"})

    service.learn_from_rejection(FieldsRejected("j1", [(question, "Invalid input")], "stuck"), POSTING)

    assert faq_now() == {"Are you authorized to work in the US?": "Yes"}
    assert question in AnswerGapStore(tmp_path / "gaps.json").list_unanswered()
    assert notes == [
        f"  LinkedIn rejected the answer to {question!r} (Invalid input). Dropped its cached answer. "
        "Answer it once with `job-bot review-answers` and it'll be reused."
    ]


def test_an_unreadable_answer_gaps_file_costs_only_the_learning(env):
    make, _, notes, _, tmp_path = env
    service = make()
    (tmp_path / "gaps.json").write_bytes(b'{"broken": ')
    service.learn_from_rejection(FieldsRejected("j1", [("Q?", "Invalid input")], "stuck"), POSTING)
    assert notes and notes[0].startswith("Warning: couldn't record the rejected answer")
    assert (tmp_path / "gaps.json").read_bytes() == b'{"broken": '  # left as it was
