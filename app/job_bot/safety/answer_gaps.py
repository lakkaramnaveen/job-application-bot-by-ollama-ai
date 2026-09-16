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


class AnswerGapStore:
    def __init__(self, path: Path):
        self._path = path

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, gaps: dict[str, dict[str, Any]]) -> None:
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

    def resolve(self, question: str) -> None:
        """Remove a question once it's been answered (see
        cmd_review_answers()) - it isn't a gap anymore.
        """
        gaps = self._load()
        if question in gaps:
            del gaps[question]
            self._save(gaps)
