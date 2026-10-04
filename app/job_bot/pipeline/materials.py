"""Tailoring the application materials for one posting: the resume and
cover letter the form gets, saved as reference copies and recorded.

Extracted from the generate_materials() closure in pipeline/cycle.py's
run_cycle() (docs/architecture.md, "split run_cycle()'s per-posting
body"). It's the slowest stage of a posting - two model calls - and now
has a name, explicit inputs, and its own tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from job_bot.browser.base_adapter import JobPosting
from job_bot.generation.artifacts import write_cover_letter, write_tailored_resume, write_tailored_resume_docx
from job_bot.generation.cover_letter import generate_cover_letter
from job_bot.generation.resume_tailor import tailor_resume
from job_bot.llm.base import LLMProvider
from job_bot.models.schemas import CoverLetter, TailoredResume
from job_bot.safety.audit_log import AuditLogger
from job_bot.tracker.db import Tracker


@dataclass(frozen=True)
class Materials:
    cover_letter: CoverLetter  # its body is what fills the form's cover-letter field
    resume_path: str  # what gets uploaded as the resume


def prepare_materials(
    posting: JobPosting,
    description: str,
    *,
    provider: LLMProvider,
    resume_text: str,
    resume_path: Path,
    applications_dir: Path,
    tracker: Tracker,
    audit: AuditLogger,
) -> Materials:
    """Tailors the resume (using past generations that led to a real
    interview/offer as few-shot examples - see
    Tracker.best_resume_examples()) and a cover letter, writes both to disk
    as reference material, and records the generation. The resume uploaded
    is a freshly tailored .docx when write_tailored_resume_docx() could
    confidently build one (see generation/resume_document.py's module
    docstring for exactly what it will and won't change), else the user's
    own unmodified `resume_path`.
    """
    examples = [
        TailoredResume(summary=r["summary"], highlighted_skills=r["skills"], bullet_points=r["bullets"])
        for r in tracker.best_resume_examples(limit=3)
    ]
    tailored = tailor_resume(provider, resume_text, description, examples=examples)
    tracker.record_resume_generation(
        posting.job_id,
        posting.title,
        posting.company,
        tailored.summary,
        tailored.highlighted_skills,
        tailored.bullet_points,
    )
    cover_letter = generate_cover_letter(provider, resume_text, description, posting.company)
    write_tailored_resume(applications_dir, posting.job_id, tailored, company=posting.company, title=posting.title)
    tailored_resume_path = write_tailored_resume_docx(
        applications_dir, posting.job_id, resume_text, tailored, company=posting.company, title=posting.title
    )
    write_cover_letter(applications_dir, posting.job_id, cover_letter, company=posting.company, title=posting.title)
    audit.log("generated_materials", job_id=posting.job_id)
    return Materials(
        cover_letter=cover_letter,
        resume_path=str(tailored_resume_path) if tailored_resume_path is not None else str(resume_path),
    )
