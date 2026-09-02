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

import re
from pathlib import Path

from job_bot.models.schemas import CoverLetter, TailoredResume

# job_id ultimately comes from a scraped LinkedIn data-job-id DOM attribute -
# untrusted data (see linkedin_adapter.py's search()) - and is used below as
# a filesystem path component. Restrict it to a safe charset (no "/", "\",
# or other separators) and reject the two reserved components that would
# otherwise resolve to a different directory even within that charset, so a
# malicious or malformed job_id can never write outside base_dir.
_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_RESERVED_COMPONENTS = {".", ".."}


class UnsafeJobId(ValueError):
    pass


def _job_dir(base_dir: Path, job_id: str) -> Path:
    if not _SAFE_JOB_ID.fullmatch(job_id) or job_id in _RESERVED_COMPONENTS:
        raise UnsafeJobId(
            f"Refusing to use job_id {job_id!r} as a filesystem path component - "
            "job IDs are scraped, untrusted data, and this value doesn't look "
            "like a real LinkedIn job ID."
        )
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
