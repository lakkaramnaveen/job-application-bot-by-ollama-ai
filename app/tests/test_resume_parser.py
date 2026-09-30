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

from job_bot.resume.parser import ResumeParseError, find_moved_resume, parse_resume


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


def test_parse_resume_raises_for_empty_txt_file(tmp_path):
    """Same empty-content guard as the PDF/DOCX cases below - a truncated
    download or a wrong RESUME_PATH pointing at an empty file must fail
    clearly here, not silently hand every downstream consumer (scoring,
    tailoring, Q&A) a blank resume with no error until much later.
    """
    path = tmp_path / "resume.txt"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ResumeParseError, match="No extractable text found"):
        parse_resume(path)


def test_parse_resume_raises_for_whitespace_only_txt_file(tmp_path):
    path = tmp_path / "resume.txt"
    path.write_text("   \n\n\t  ", encoding="utf-8")
    with pytest.raises(ResumeParseError, match="No extractable text found"):
        parse_resume(path)


def test_parse_resume_raises_a_clean_error_for_a_non_utf8_txt_file(tmp_path):
    """Real failure this guards against: a resume.txt saved with a non-UTF-8
    encoding (exported from Word on Windows, or hand-saved by an editor
    defaulting to the system locale rather than UTF-8) previously crashed
    parse_resume() with a raw UnicodeDecodeError instead of this module's
    own ResumeParseError, the one exception type EXPECTED_ERRORS catches.
    U+2019 (a right single quotation mark - a common "smart quote" in a
    Word-authored resume) encodes to cp1252 as the single byte 0x92, which
    is not valid UTF-8 on its own.
    """
    path = tmp_path / "resume.txt"
    path.write_bytes("Jane Doe’s resume".encode("cp1252"))
    with pytest.raises(ResumeParseError, match="Could not read .* as UTF-8 text"):
        parse_resume(path)


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


def test_parse_resume_raises_cleanly_for_a_corrupted_pdf(tmp_path):
    """Unlike the other PDF/DOCX cases in this file, this one does NOT mock
    PdfReader - it exercises pypdf's own real behavior against a genuinely
    invalid file, since the bug being guarded against is specifically pypdf
    raising its own PdfStreamError instead of this module's ResumeParseError.
    A truncated download or a non-PDF file with a .pdf extension previously
    crashed with a raw pypdf exception - not caught by cli.py's
    EXPECTED_ERRORS or config.py's validate_ready() - instead of the clean
    message every other unparseable-resume case already produced.
    """
    path = tmp_path / "resume.pdf"
    path.write_bytes(b"this is not a real pdf file")

    with pytest.raises(ResumeParseError, match="Could not read PDF file"):
        parse_resume(path)


def test_parse_resume_raises_cleanly_for_a_corrupted_docx(tmp_path):
    """Same real-library-integration reasoning as the corrupted-PDF case
    above, for python-docx: a non-DOCX file with a .docx extension raises
    PackageNotFoundError, which must surface as ResumeParseError too.
    """
    path = tmp_path / "resume.docx"
    path.write_bytes(b"this is not a real docx file")

    with pytest.raises(ResumeParseError, match="Could not read DOCX file"):
        parse_resume(path)


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


def test_find_moved_resume_finds_the_same_name_one_folder_down(tmp_path):
    """The real case: RESUME_PATH still points at ~/Desktop/<name>, but the
    file moved into ~/Desktop/DESK/ along with the project."""
    (tmp_path / "DESK").mkdir()
    moved = tmp_path / "DESK" / "My resume.docx"
    moved.write_bytes(b"x")
    assert find_moved_resume(tmp_path / "My resume.docx") == moved


def test_find_moved_resume_returns_none_when_nothing_matches(tmp_path):
    (tmp_path / "DESK").mkdir()
    (tmp_path / "DESK" / "Other resume.docx").write_bytes(b"x")
    assert find_moved_resume(tmp_path / "My resume.docx") is None


def test_find_moved_resume_starts_from_the_nearest_folder_that_still_exists(tmp_path):
    """If the old folder itself is gone (renamed), search from its parent."""
    (tmp_path / "renamed").mkdir()
    moved = tmp_path / "renamed" / "cv.pdf"
    moved.write_bytes(b"x")
    assert find_moved_resume(tmp_path / "old-folder" / "cv.pdf") == moved


def test_find_moved_resume_is_bounded_in_depth_and_skips_hidden_folders(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "cv.pdf").write_bytes(b"x")  # three levels down - beyond max_depth=2
    hidden = tmp_path / ".cache"
    hidden.mkdir()
    (hidden / "cv.pdf").write_bytes(b"x")
    assert find_moved_resume(tmp_path / "cv.pdf") is None


def test_find_moved_resume_gives_up_after_max_entries(tmp_path):
    (tmp_path / "zz").mkdir()
    (tmp_path / "zz" / "cv.pdf").write_bytes(b"x")
    for i in range(20):
        (tmp_path / f"filler-{i:02d}.txt").write_text("x")
    assert find_moved_resume(tmp_path / "cv.pdf", max_entries=10) is None
    assert find_moved_resume(tmp_path / "cv.pdf") == tmp_path / "zz" / "cv.pdf"
