"""Extracts plain text from the user's resume file, in whatever format they
kept it in (PDF, DOCX, or plain text). Every downstream consumer (scoring,
tailoring, cover letters, Q&A) works from this text, not the original file -
see resume/store.py, which caches the result of parse_resume() for the
lifetime of one run.
"""

from pathlib import Path


class ResumeParseError(RuntimeError):
    """Raised for a missing file, an unsupported extension, or a PDF/DOCX
    that yields no extractable text (e.g. a scanned image with no OCR text
    layer) - caught in cli.py's EXPECTED_ERRORS and reported as a clean
    message rather than a traceback.
    """


def parse_resume(path: Path) -> str:
    """Extract plain text from a resume file. Supports PDF and DOCX."""
    if not path.exists():
        raise ResumeParseError(f"Resume file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path)
    if suffix == ".docx":
        return _parse_docx(path)
    if suffix == ".txt":
        return _parse_txt(path)
    raise ResumeParseError(f"Unsupported resume format: {suffix} (use .pdf, .docx, or .txt)")


def _parse_txt(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        # Same empty-content guard as _parse_pdf/_parse_docx below - an
        # empty or whitespace-only resume.txt (truncated download, wrong
        # RESUME_PATH, ...) used to silently return "" here instead of
        # failing clearly like the other two formats already did, letting
        # every downstream consumer (scoring, tailoring, Q&A) run against a
        # blank resume with no error until something further along produced
        # a confusing, hard-to-trace failure.
        raise ResumeParseError(f"No extractable text found in resume file: {path}")
    return text


def _parse_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise ResumeParseError(f"No extractable text found in PDF: {path}")
    return text


def _parse_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    text = "\n".join(p.text for p in document.paragraphs)
    if not text.strip():
        raise ResumeParseError(f"No extractable text found in DOCX: {path}")
    return text
