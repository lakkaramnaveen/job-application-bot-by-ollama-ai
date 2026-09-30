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
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        # A resume.txt saved with a non-UTF-8 encoding (common for a plain
        # text file: exported from Word on Windows, or hand-saved by an
        # editor defaulting to the system locale's encoding rather than
        # UTF-8) previously crashed parse_resume() with a raw
        # UnicodeDecodeError - the one exception type this module doesn't
        # define and cli.py's EXPECTED_ERRORS doesn't catch, unlike every
        # other resume-parsing failure here, which already raises the
        # clean ResumeParseError EXPECTED_ERRORS turns into a one-line
        # message.
        raise ResumeParseError(
            f"Could not read {path} as UTF-8 text: {e}. Re-save it with UTF-8 encoding, or use a "
            ".pdf/.docx resume instead."
        ) from e
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
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except PyPdfError as e:
        # A truncated download or a non-PDF file with a .pdf extension
        # (confirmed live: pypdf raises PdfStreamError, a PyPdfError
        # subclass, on garbage bytes) previously propagated straight out
        # as a raw pypdf exception instead of this module's own
        # ResumeParseError - the one exception type callers (cli.py's
        # EXPECTED_ERRORS, config.py's validate_ready()) actually handle
        # cleanly.
        raise ResumeParseError(f"Could not read PDF file {path}: {e}") from e
    if not text.strip():
        raise ResumeParseError(f"No extractable text found in PDF: {path}")
    return text


def _parse_docx(path: Path) -> str:
    import docx
    from docx.opc.exceptions import OpcError

    try:
        document = docx.Document(str(path))
        text = "\n".join(p.text for p in document.paragraphs)
    except OpcError as e:
        # Same reasoning as _parse_pdf's PyPdfError handling above -
        # confirmed live: python-docx raises PackageNotFoundError (an
        # OpcError subclass) for a non-DOCX file with a .docx extension.
        raise ResumeParseError(f"Could not read DOCX file {path}: {e}") from e
    if not text.strip():
        raise ResumeParseError(f"No extractable text found in DOCX: {path}")
    return text


def find_moved_resume(path: Path, *, max_depth: int = 2, max_entries: int = 5000) -> Path | None:
    """A file with `path`'s exact name near where `path` used to be, or None
    - for suggesting a fix when RESUME_PATH points at a resume that was
    moved rather than deleted (the real case this exists for: a resume on
    the Desktop moved one folder down along with the project).

    Searches up to `max_depth` directory levels below the nearest existing
    ancestor of `path`'s folder, skipping hidden directories, and gives up
    after `max_entries` directory entries so a large home directory can't
    make `job-bot doctor`/`run` slow. Only ever suggests - nothing is
    changed. Matches are sorted so the suggestion is deterministic.
    """
    start = path.parent
    while not start.exists() and start != start.parent:
        start = start.parent
    matches: list[Path] = []
    seen = 0
    frontier = [(start, 0)]
    while frontier:
        directory, depth = frontier.pop(0)
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            seen += 1
            if seen > max_entries:
                return min(matches) if matches else None
            if entry.name == path.name and entry.is_file():
                matches.append(entry)
            elif depth < max_depth and not entry.name.startswith(".") and entry.is_dir() and not entry.is_symlink():
                frontier.append((entry, depth + 1))
    return min(matches) if matches else None
