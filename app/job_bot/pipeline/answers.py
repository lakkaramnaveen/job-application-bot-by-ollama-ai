"""Answering Easy Apply form questions, and learning from LinkedIn's
verdict on the answers.

Before this module, this was a closure inside cli.py's _run_apply_cycle()
plus two module-level helpers, testable only by running a whole
`job-bot run` with fakes. It's also where this week's "the bot isn't
learning" problems lived: a contact block cached as the phone answer and
replayed into every application, and refused answers replayed on every
posting that asked the same question. AnswerService keeps the whole
answer lifecycle in one place - the cache lookup, the model call, what's
allowed back into the cache, and dropping what LinkedIn rejected.

Every collaborator is injected (the model call, the stores, the
`notify` sink for user-facing lines), so the policy is testable without
a browser, a model, or a run. See docs/architecture.md.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from job_bot.browser.base_adapter import JobPosting
from job_bot.browser.linkedin_adapter import FieldsRejected, LinkedInAdapter
from job_bot.data_files import CorruptDataFile
from job_bot.generation.qa_answerer import answer_question
from job_bot.llm.base import LLMProvider
from job_bot.models.schemas import ApplicationAnswer
from job_bot.resume.store import ResumeStore
from job_bot.safety.answer_gaps import AnswerGapStore
from job_bot.text_utils import is_echoed_question, is_per_position_field
from job_bot.tracker.db import Tracker

AnswerFn = Callable[..., ApplicationAnswer]


def cacheable_answer(question: str, answer: str) -> str | None:
    """What may be saved to FAQ_PATH for `question`: the answer reduced to
    the shape the question asks for, or None to not cache it at all.

    Real case (2026-10-01/02): the cached answer to "Mobile phone number*"
    was a whole contact block - name, phone, email, city - saved once at
    high confidence and replayed into every later application. The fill
    step reduces a value to its field's type (LinkedInAdapter's
    _phone_value() and friends), but a cached answer outlives the field it
    was produced for, so it's checked here too:
    - a phone question caches just the number, an email question just the
      address, and a "how many"/years question just the number;
    - an answer with no such value isn't cached (it's still used for this
      application, just not reused).
    Anything else is cached as written.
    """
    q = question.casefold()
    if "phone" in q and "email" not in q:
        return LinkedInAdapter._phone_value(answer)
    if "email" in q and "phone" not in q:
        return LinkedInAdapter._email_value(answer)
    if LinkedInAdapter._asks_for_a_number(question):
        return LinkedInAdapter._numeric_value(answer)
    return answer


class AnswerService:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        resume_text: str,
        resume_store: ResumeStore,
        tracker: Tracker,
        answer_gaps: AnswerGapStore,
        save_confidence: float,
        answer_fn: AnswerFn = answer_question,
        notify: Callable[[str], Any] = print,
    ):
        self._provider = provider
        self._resume_text = resume_text
        self._resume_store = resume_store
        self._tracker = tracker
        self._answer_gaps = answer_gaps
        self._save_confidence = save_confidence
        self._answer_fn = answer_fn
        self._notify = notify

    def answer(self, question: str, job_id: str) -> str:
        """The answer to one form question for posting `job_id` - "" leaves
        the field unanswered, which the adapter reports as an unanswered
        required question (queued for `job-bot review-answers`)."""
        if is_per_position_field(question):
            # Nothing recorded or cached - see is_per_position_field().
            return ""
        faq_answers = self._resume_store.faq_answers()
        # An exact-text FAQ match is a curated, confident answer - asking
        # the model to re-derive it is a guaranteed-redundant call, costly
        # on a local model. A near-miss phrasing still goes to the model.
        # An echoed entry cached before is_echoed_question() existed is
        # ignored rather than replayed.
        cached = faq_answers.get(question)
        if cached is not None and not is_echoed_question(question, cached):
            self._tracker.record_qa(job_id, question, cached)
            return cached
        result = self._answer_fn(
            self._provider,
            self._resume_text,
            faq_answers,
            question,
            recent_answers=self._tracker.recent_qa_pairs(),
        )
        if is_echoed_question(question, result.answer):
            # No answer, not a bad one: never recorded or cached, so it
            # can't be replayed from FAQ_PATH or reused as a few-shot example.
            return ""
        self._tracker.record_qa(job_id, question, result.answer)
        cacheable = cacheable_answer(question, result.answer)
        if cacheable is not None and result.based_on_resume and result.confidence >= self._save_confidence:
            try:
                self._resume_store.save_faq_answer(question, cacheable)
            except CorruptDataFile as e:
                # Caching is an optimization - the answer itself is still
                # good, so the application goes ahead uncached.
                self._notify(f"Warning: answer not cached - {e}")
        return result.answer

    def learn_from_rejection(self, e: FieldsRejected, posting: JobPosting) -> None:
        """LinkedIn refused these answers - don't replay them. Each rejected
        question loses its cached FAQ answer (so the next posting asking it
        gets a fresh answer instead of the same refused one) and is queued
        in answer_gaps for `job-bot review-answers`, where answering it once
        saves the right answer for every future posting.

        Like UnansweredRequiredQuestion, this doesn't count toward
        MAX_APPLY_ATTEMPTS (the caller's rule): its fix is the answer.
        Best effort - a data-file problem here only loses the learning,
        never the run.
        """
        for question, error in e.rejected:
            if not question:
                continue
            try:
                removed = self._resume_store.remove_faq_answer(question)
                self._answer_gaps.record(
                    question, job_id=posting.job_id, company=posting.company, title=posting.title
                )
            except CorruptDataFile as data_error:
                self._notify(f"Warning: couldn't record the rejected answer - {data_error}")
                continue
            dropped = " Dropped its cached answer." if removed else ""
            self._notify(
                f"  LinkedIn rejected the answer to {question!r} ({error}).{dropped} "
                "Answer it once with `job-bot review-answers` and it'll be reused."
            )
