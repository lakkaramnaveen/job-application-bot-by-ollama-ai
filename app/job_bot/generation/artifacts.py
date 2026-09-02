"""Persists generated application material to disk.

The LinkedIn adapter always uploads the user's own verified resume file
(settings.resume_path) - never a freshly LLM-generated document the user
hasn't reviewed, since a factual error in a document actually submitted to
an employer is a real, hard-to-undo risk. The *tailored* resume (summary,
highlighted skills, reordered bullets) is instead written here as a plain
text file per job, for the user to read, copy from, or reuse in interview
prep - closing the loop on generation without auto-submitting unreviewed
content.
"""

from pathlib import Path

from job_bot.models.schemas import CoverLetter, TailoredResume


def _job_dir(base_dir: Path, job_id: str) -> Path:
    out_dir = base_dir / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_tailored_resume(base_dir: Path, job_id: str, tailored: TailoredResume) -> Path:
    lines = [
        "SUMMARY",
        tailored.summary,
        "",
        "HIGHLIGHTED SKILLS (most relevant first)",
        *(f"- {skill}" for skill in tailored.highlighted_skills),
        "",
        "TAILORED BULLET POINTS",
        *(f"- {bullet}" for bullet in tailored.bullet_points),
        "",
    ]
    path = _job_dir(base_dir, job_id) / "tailored_resume.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_cover_letter(base_dir: Path, job_id: str, cover_letter: CoverLetter) -> Path:
    path = _job_dir(base_dir, job_id) / "cover_letter.txt"
    path.write_text(cover_letter.body, encoding="utf-8")
    return path
