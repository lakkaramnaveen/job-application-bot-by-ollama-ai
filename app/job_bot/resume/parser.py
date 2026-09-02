from pathlib import Path


class ResumeParseError(RuntimeError):
    pass


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
        return path.read_text(encoding="utf-8")
    raise ResumeParseError(f"Unsupported resume format: {suffix} (use .pdf, .docx, or .txt)")


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
