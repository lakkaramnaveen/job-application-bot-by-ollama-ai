"""Tracks required Easy Apply questions the LLM couldn't answer confidently
(a required text/radio/select field left deliberately unanswered rather than
guessed - see browser/linkedin_adapter.py's UnansweredRequiredQuestion and
its "never guess" reasoning), so they can be reviewed and answered once via
`job-bot review-answers` instead of the same question silently failing the
same way on every future posting that happens to ask it.

This is the practical shape of "the bot should learn from its mistakes" for
a local model whose weights this project never touches: it can't retrain
itself, but it can remember precisely what it couldn't answer, and an
answer given here is reused as context for every future application via
qa_answerer.py's FAQ_PATH plumbing - the same mechanism that already lets a
good answer get reused, just closing the loop for the failures too.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from job_bot.data_files import assert_safe_to_overwrite


class AnswerGapStore:
    def __init__(self, path: Path):
        self._path = path

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # UnicodeDecodeError alongside JSONDecodeError - an
            # answer_gaps.json saved with a non-UTF-8 encoding is
            # corruption exactly the same way invalid JSON already is, and
            # gets the same graceful fallback here rather than crashing
            # every command that touches unanswered-question tracking.
            # cli.py's own _answer_gaps_check() doctor check is what
            # actually surfaces this to the user, instead of it silently
            # looking like "nothing unanswered".
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, gaps: dict[str, dict[str, Any]]) -> None:
        assert_safe_to_overwrite(self._path, dict)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(gaps, indent=2), encoding="utf-8")

    def record(self, question: str, *, job_id: str, company: str, title: str) -> None:
        """Log one occurrence of a question that couldn't be answered.
        Repeated occurrences of the same question (very common - the same
        eligibility/sponsorship-style question is asked near-verbatim
        across many different postings) accumulate a count rather than
        creating duplicate entries, so review shows the questions actually
        worth answering first.
        """
        gaps = self._load()
        now = datetime.now(UTC).isoformat()
        entry = gaps.get(question, {"count": 0, "first_seen_at": now})
        entry["count"] = int(entry.get("count", 0)) + 1
        entry["last_seen_at"] = now
        entry["example_job_id"] = job_id
        entry["example_company"] = company
        entry["example_title"] = title
        gaps[question] = entry
        self._save(gaps)

    def list_unanswered(self) -> dict[str, dict[str, Any]]:
        return self._load()

    def resolve(self, question: str) -> bool:
        """Remove a question, whether because it's been answered (see
        cmd_review_answers()'s interactive loop) or permanently dismissed
        without an answer (`job-bot review-answers --dismiss`) - either
        way it isn't a gap anymore. Returns True if it was present (and is
        now removed), False if it wasn't there to begin with - the same
        convention CompanyBlacklist.remove() already uses, so a caller can
        report which requested dismissal(s) didn't actually match anything.
        """
        gaps = self._load()
        if question not in gaps:
            return False
        del gaps[question]
        self._save(gaps)
        return True
