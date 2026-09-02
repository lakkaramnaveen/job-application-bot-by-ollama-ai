import json
from pathlib import Path

from job_bot.resume.parser import parse_resume


class ResumeStore:
    """Loads the user's resume text and previously-answered FAQ questions."""

    def __init__(self, resume_path: Path, faq_path: Path):
        self._resume_path = resume_path
        self._faq_path = faq_path
        self._resume_text: str | None = None

    def resume_text(self) -> str:
        if self._resume_text is None:
            self._resume_text = parse_resume(self._resume_path)
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
