"""parse_resume() had no test coverage at all before this file - a
regression here (e.g. a version bump changing how pypdf/python-docx report
empty text) would only surface when a real user's resume silently failed
to parse. The PDF/DOCX cases mock PdfReader/Document rather than generating
real binary fixtures, since the behavior under test is job_bot's own
join-pages-and-check-for-empty-text logic, not pypdf's or python-docx's own
extraction correctness.
"""

from types import SimpleNamespace

import pytest

from job_bot.resume.parser import ResumeParseError, parse_resume


def test_parse_resume_raises_for_missing_file(tmp_path):
    missing = tmp_path / "resume.pdf"
    with pytest.raises(ResumeParseError, match="not found"):
        parse_resume(missing)


def test_parse_resume_raises_for_unsupported_extension(tmp_path):
    path = tmp_path / "resume.rtf"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(ResumeParseError, match="Unsupported resume format"):
        parse_resume(path)


def test_parse_resume_reads_txt_file(tmp_path):
    path = tmp_path / "resume.txt"
    path.write_text("Jane Doe\nSoftware Engineer", encoding="utf-8")
    assert parse_resume(path) == "Jane Doe\nSoftware Engineer"


def test_parse_resume_extension_check_is_case_insensitive(tmp_path):
    """A resume downloaded on Windows can easily carry an uppercase
    extension - the format dispatch must not silently misroute it to the
    "unsupported format" error.
    """
    path = tmp_path / "resume.TXT"
    path.write_text("Case-insensitive extension", encoding="utf-8")
    assert parse_resume(path) == "Case-insensitive extension"


def test_parse_resume_extracts_text_from_docx(tmp_path, monkeypatch):
    path = tmp_path / "resume.docx"
    path.write_bytes(b"stub - docx.Document() is mocked below")
    fake_document = SimpleNamespace(
        paragraphs=[SimpleNamespace(text="Jane Doe"), SimpleNamespace(text="Software Engineer")]
    )
    monkeypatch.setattr("docx.Document", lambda p: fake_document)

    assert parse_resume(path) == "Jane Doe\nSoftware Engineer"


def test_parse_resume_raises_for_docx_with_no_extractable_text(tmp_path, monkeypatch):
    path = tmp_path / "resume.docx"
    path.write_bytes(b"stub")
    fake_document = SimpleNamespace(paragraphs=[SimpleNamespace(text=""), SimpleNamespace(text="   ")])
    monkeypatch.setattr("docx.Document", lambda p: fake_document)

    with pytest.raises(ResumeParseError, match="No extractable text found in DOCX"):
        parse_resume(path)


def test_parse_resume_extracts_text_from_pdf(tmp_path, monkeypatch):
    path = tmp_path / "resume.pdf"
    path.write_bytes(b"stub - pypdf.PdfReader() is mocked below")
    fake_pages = [
        SimpleNamespace(extract_text=lambda: "Jane Doe"),
        SimpleNamespace(extract_text=lambda: "Software Engineer"),
    ]
    monkeypatch.setattr("pypdf.PdfReader", lambda p: SimpleNamespace(pages=fake_pages))

    assert parse_resume(path) == "Jane Doe\nSoftware Engineer"


def test_parse_resume_raises_for_pdf_with_no_extractable_text(tmp_path, monkeypatch):
    """Covers a scanned-image PDF with no OCR text layer - extract_text()
    returns "" (or None, which pypdf itself can also return) for every page.
    """
    path = tmp_path / "resume.pdf"
    path.write_bytes(b"stub")
    fake_pages = [SimpleNamespace(extract_text=lambda: None), SimpleNamespace(extract_text=lambda: "")]
    monkeypatch.setattr("pypdf.PdfReader", lambda p: SimpleNamespace(pages=fake_pages))

    with pytest.raises(ResumeParseError, match="No extractable text found in PDF"):
        parse_resume(path)
