import json
from pathlib import Path

from job_bot.resume.parser import parse_resume


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
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items()}

    def save_faq_answer(self, question: str, answer: str) -> None:
        answers = self.faq_answers()
        answers[question] = answer
        self._faq_path.parent.mkdir(parents=True, exist_ok=True)
        self._faq_path.write_text(json.dumps(answers, indent=2), encoding="utf-8")
