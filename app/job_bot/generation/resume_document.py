"""Builds an actual per-job resume FILE (.docx) from a TailoredResume, for
upload to the application form itself - unlike artifacts.py's
write_tailored_resume() (a plain-text reference copy for the user to read),
this is the document LinkedInAdapter.fill_and_submit() attaches in place of
the user's static resume_path, at the user's explicit request to have the
submitted resume actually vary by job description.

Deliberately conservative about what it's willing to change: only the
professional summary and the skills line are LLM-generated content, since
those are what most affect ATS keyword matching and a recruiter's first
impression, and a paraphrased sentence being imperfect is a low-stakes
mistake. Job titles, companies, dates, and every bullet point under them are
copied verbatim from the user's own real resume_text, byte-for-byte - never
the model's own rewritten bullet_points, and never reordered or reworded -
because a wrong company name, date range, or fabricated-sounding
accomplishment in a document actually submitted to a real employer is a
much more serious, harder-to-undo mistake than a slightly-off summary
sentence. build_tailored_resume_docx() returns False rather than guess
whenever it can't confidently find the SUMMARY/SKILLS/EXPERIENCE section
boundaries in resume_text - the caller then falls back to uploading the
user's own unmodified resume file, exactly like before this feature existed.
"""

import re
from pathlib import Path

from job_bot.models.schemas import TailoredResume

# A real section header is a short line naming the section, not a paragraph
# that happens to mention the word - caps how long a line can be and still
# be treated as one.
_MAX_HEADER_LINE_LENGTH = 40

# Tried in order against whatever follows the SKILLS section - covers the
# common headings a resume's work-history section is filed under. Everything
# from whichever of these is found onward (through education, certifications,
# projects, and anything else after it) is kept verbatim, so this doesn't
# need to separately recognize any section that comes later.
_EXPERIENCE_HEADER_KEYWORDS = ("EXPERIENCE", "EMPLOYMENTHISTORY", "WORKHISTORY")

# "PROFILE" alone is too permissive to match as a free substring: unlike
# SUMMARY/OBJECTIVE/EXPERIENCE, it's also an ordinary word in a resume's own
# contact block ("LinkedIn Profile", "GitHub Profile", "Portfolio"), which
# a plain substring match would mistake for the header itself, truncating
# the real header_lines and potentially misplacing summary_idx (confirmed:
# "LinkedIn Profile" as a lone contact-info line collapses to
# "LINKEDINPROFILE", which contains "PROFILE"). Matched only when the whole
# collapsed line is exactly "PROFILE" or one of a small allowlist of
# legitimate resume-header modifiers plus "PROFILE" - never as a substring
# of an arbitrary surrounding word.
_AMBIGUOUS_KEYWORDS = frozenset({"PROFILE"})
_HEADER_PREFIX_WORDS = ("PROFESSIONAL", "PERSONAL", "CAREER", "EXECUTIVE", "CANDIDATE")


def _matches_keyword(collapsed: str, keyword: str) -> bool:
    if keyword not in _AMBIGUOUS_KEYWORDS:
        return keyword in collapsed
    return collapsed == keyword or any(collapsed == prefix + keyword for prefix in _HEADER_PREFIX_WORDS)


def _collapse(line: str) -> str:
    """Removes all whitespace and uppercases a line before keyword matching.

    PDF text extraction can split a header's own letters across stray
    spaces from kerning/ligature artifacts (confirmed against a real
    resume.pdf parsed by this project: "PROFESSIONAL SUMMAR Y", "EDUCA
    TION") - collapsing whitespace first still matches "SUMMARY"/
    "EDUCATION" cleanly despite that.
    """
    return re.sub(r"\s+", "", line).upper()


def _find_header_line(lines: list[str], keywords: tuple[str, ...], start: int) -> int | None:
    for i in range(start, len(lines)):
        collapsed = _collapse(lines[i])
        if len(collapsed) > _MAX_HEADER_LINE_LENGTH:
            continue
        if any(_matches_keyword(collapsed, keyword) for keyword in keywords):
            return i
    return None


def build_tailored_resume_docx(resume_text: str, tailored: TailoredResume, output_path: Path) -> bool:
    """Writes a tailored resume .docx to output_path and returns True, or
    returns False (writing nothing) if resume_text's SUMMARY/SKILLS/
    EXPERIENCE-or-equivalent section headers can't be confidently located -
    see this module's docstring for why guessing wrong here is worse than
    just not tailoring this one upload.
    """
    lines = resume_text.splitlines()
    summary_idx = _find_header_line(lines, ("SUMMARY", "OBJECTIVE", "PROFILE"), 0)
    if summary_idx is None or summary_idx == 0:
        return False
    skills_idx = _find_header_line(lines, ("SKILLS", "TECHNICALSKILLS", "CORECOMPETENCIES"), summary_idx + 1)
    if skills_idx is None:
        return False
    experience_idx = _find_header_line(lines, _EXPERIENCE_HEADER_KEYWORDS, skills_idx + 1)
    if experience_idx is None:
        return False

    header_lines = [line.strip() for line in lines[:summary_idx] if line.strip()]
    verbatim_rest = lines[experience_idx:]

    import docx  # lazy import - see resume/parser.py's _parse_docx() for the same pattern

    document = docx.Document()
    for line in header_lines:
        document.add_paragraph(line)

    document.add_heading("Professional Summary", level=2)
    document.add_paragraph(tailored.summary)

    document.add_heading("Skills", level=2)
    document.add_paragraph(", ".join(tailored.highlighted_skills))

    for line in verbatim_rest:
        document.add_paragraph(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(output_path))
    return True
