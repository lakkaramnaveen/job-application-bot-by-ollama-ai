"""Persists generated application material to disk.

write_tailored_resume() below is always written, as a plain-text reference
copy of the summary/skills/bullets for the user to read, copy from, or reuse
in interview prep. write_tailored_resume_docx() additionally builds the
actual per-job .docx LinkedInAdapter.fill_and_submit() uploads in place of
the user's static resume file, at the user's request to have the submitted
resume vary by job description - see generation/resume_document.py's module
docstring for why it only ever tailors the summary/skills, never the real
work-history section, and for why it can decline (returning None here) if
that resume's sections can't be confidently located; cli.py falls back to
the user's own unmodified resume_path whenever that happens.

Materials land under base_dir (settings.applications_dir) as
<today's date>/<job folder>/ - one dated folder per day's worth of
applications, so browsing base_dir on disk (e.g. pointed at a folder on the
Desktop) reads as a day-by-day job-search log rather than one flat pile of
job-id-named folders.
"""

import re
from datetime import date
from pathlib import Path

from job_bot.generation.resume_document import build_tailored_resume_docx
from job_bot.models.schemas import CoverLetter, TailoredResume

# job_id ultimately comes from a scraped LinkedIn data-job-id DOM attribute -
# untrusted data (see linkedin_adapter.py's search()) - and is used below as
# a filesystem path component. Restrict it to a safe charset (no "/", "\",
# or other separators) and reject the two reserved components that would
# otherwise resolve to a different directory even within that charset, so a
# malicious or malformed job_id can never write outside base_dir.
_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_RESERVED_COMPONENTS = {".", ".."}

# company/title are also scraped, untrusted LinkedIn text, but unlike job_id
# they're only ever used as a cosmetic label appended to the (already-safe)
# job_id, never as the sole or leading path component - so a much more
# permissive whitelist (letters, digits, spaces, and a few punctuation
# marks) is fine here: it only needs to keep the result a single, sane path
# segment, not double as job_id's uniqueness/identity guarantee.
_UNSAFE_LABEL_CHARS = re.compile(r"[^A-Za-z0-9 ._-]")


class UnsafeJobId(ValueError):
    pass


def _safe_label(text: str, *, max_length: int = 60) -> str:
    cleaned = _UNSAFE_LABEL_CHARS.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned[:max_length]


def _job_dir(base_dir: Path, job_id: str, *, company: str = "", title: str = "") -> Path:
    if not _SAFE_JOB_ID.fullmatch(job_id) or job_id in _RESERVED_COMPONENTS:
        raise UnsafeJobId(
            f"Refusing to use job_id {job_id!r} as a filesystem path component - "
            "job IDs are scraped, untrusted data, and this value doesn't look "
            "like a real LinkedIn job ID."
        )
    label = " - ".join(part for part in (_safe_label(company), _safe_label(title)) if part)
    folder_name = f"{job_id} - {label}" if label else job_id
    out_dir = base_dir / date.today().isoformat() / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_tailored_resume(
    base_dir: Path, job_id: str, tailored: TailoredResume, *, company: str = "", title: str = ""
) -> Path:
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
    path = _job_dir(base_dir, job_id, company=company, title=title) / "tailored_resume.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_tailored_resume_docx(
    base_dir: Path, job_id: str, resume_text: str, tailored: TailoredResume, *, company: str = "", title: str = ""
) -> Path | None:
    """Returns the written .docx's path, or None if build_tailored_resume_docx()
    declined (resume_text's sections couldn't be confidently located) -
    callers must fall back to the user's own resume_path in that case, never
    treat None as "retry" or "error".
    """
    path = _job_dir(base_dir, job_id, company=company, title=title) / "tailored_resume.docx"
    return path if build_tailored_resume_docx(resume_text, tailored, path) else None


def write_cover_letter(
    base_dir: Path, job_id: str, cover_letter: CoverLetter, *, company: str = "", title: str = ""
) -> Path:
    path = _job_dir(base_dir, job_id, company=company, title=title) / "cover_letter.txt"
    path.write_text(cover_letter.body, encoding="utf-8")
    return path
