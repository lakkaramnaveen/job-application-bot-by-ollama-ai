import json
from pathlib import Path

from job_bot.data_files import assert_safe_to_overwrite
from job_bot.models.schemas import looks_like_leaked_reasoning
from job_bot.resume.parser import parse_resume
from job_bot.text_utils import is_echoed_question


def unusable_faq_reason(question: str, answer: str) -> str | None:
    """Why a cached FAQ answer can't be used, or None if it's fine - the
    same two rules cli.py's answer() and Tracker.recent_qa_pairs() enforce
    at runtime (docs/qwen_notes.md §1 and §6). Shared by `job-bot faq clean`
    and the dashboard's FAQ panel, so both flag exactly what a run skips.
    """
    if is_echoed_question(question, answer):
        return "echoes the question"
    if looks_like_leaked_reasoning(answer):
        return "leaked reasoning"
    return None


class ResumeStore:
    """Loads the user's resume text and previously-answered FAQ questions."""

    def __init__(self, resume_path: Path, faq_path: Path):
        self._resume_path = resume_path
        self._faq_path = faq_path
        self._resume_text: str | None = None
        self._resume_mtime: float | None = None

    def resume_text(self) -> str:
        """Caches by the file's last-modified time, not just for this
        object's lifetime: `job-bot run --loop` can hold one ResumeStore
        for many hours across many search cycles, and re-parsing the same
        PDF on every scoring/tailoring call within one cycle would be
        wasteful - but a resume edited and re-saved mid-loop (exactly what
        happens when you export a corrected PDF to RESUME_PATH) must still
        take effect on the next cycle, not silently keep using the version
        that was current when the loop started until the process restarts.
        """
        if self._resume_text is None:
            self._resume_text = parse_resume(self._resume_path)
            self._resume_mtime = self._resume_path.stat().st_mtime
            return self._resume_text

        try:
            mtime = self._resume_path.stat().st_mtime
        except OSError:
            # Can't check freshness right now (e.g. a transient permission
            # issue, or the file momentarily missing mid-save) - keep
            # serving the last known-good copy rather than failing a
            # long-running loop over it.
            return self._resume_text

        if mtime != self._resume_mtime:
            self._resume_text = parse_resume(self._resume_path)
            self._resume_mtime = mtime
        return self._resume_text

    def faq_answers(self) -> dict[str, str]:
        """Question -> previously given answer, used as few-shot context."""
        if not self._faq_path.exists():
            return {}
        try:
            data = json.loads(self._faq_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # UnicodeDecodeError alongside JSONDecodeError - a faq_cache.json
            # saved with a non-UTF-8 encoding (a hand edit, a bad
            # restore/backup, ...) is corruption exactly the same way
            # invalid JSON already is, and gets the same graceful fallback
            # here rather than crashing every command that touches FAQ
            # answers. cli.py's own _faq_check() doctor check is what
            # actually surfaces this corruption to the user, instead of it
            # silently looking like "no FAQ cache yet".
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items()}

    def save_faq_answer(self, question: str, answer: str) -> None:
        assert_safe_to_overwrite(self._faq_path, dict)
        answers = self.faq_answers()
        answers[question] = answer
        self._faq_path.parent.mkdir(parents=True, exist_ok=True)
        self._faq_path.write_text(json.dumps(answers, indent=2), encoding="utf-8")

    def remove_faq_answer(self, question: str) -> bool:
        """Removes one cached FAQ answer - e.g. to force a wrong or stale
        one (a low-confidence guess that slipped past FAQ_SAVE_CONFIDENCE,
        or a typo made during `job-bot review-answers`) to be re-asked and
        re-reviewed instead of kept forever, exactly as `save_faq_answer()`
        commits it. Returns True if the question was cached (and is now
        removed), False if it wasn't there to begin with.
        """
        answers = self.faq_answers()
        if question not in answers:
            return False
        del answers[question]
        self._faq_path.write_text(json.dumps(answers, indent=2), encoding="utf-8")
        return True
