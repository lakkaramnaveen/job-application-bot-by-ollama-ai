import docx

from job_bot.generation.resume_document import build_tailored_resume_docx
from job_bot.models.schemas import TailoredResume

STRUCTURED_RESUME = (
    "Jane Doe\n"
    "jane@example.com • 555-0100\n\n"
    "SUMMARY\n"
    "Backend engineer with 5 years of Python experience.\n\n"
    "SKILLS\n"
    "Python, Django, PostgreSQL, AWS\n\n"
    "EXPERIENCE\n"
    "Software Engineer, Acme Corp, 2020-Present\n"
    "- Built and maintained REST APIs handling 100,000+ requests daily.\n"
    "- Reduced query latency by 40% through indexing and caching.\n\n"
    "EDUCATION\n"
    "B.S. Computer Science, State University, 2016-2020\n"
)

TAILORED = TailoredResume(
    summary="Backend engineer specializing in Python and Django, with a track record of scaling APIs.",
    highlighted_skills=["Python", "Django", "PostgreSQL"],
    bullet_points=["Rewrote bullet the document must NOT use verbatim."],
)


def _paragraph_texts(path):
    return [p.text for p in docx.Document(str(path)).paragraphs]


def test_build_tailored_resume_docx_uses_the_tailored_summary_and_skills(tmp_path):
    output_path = tmp_path / "tailored.docx"

    built = build_tailored_resume_docx(STRUCTURED_RESUME, TAILORED, output_path)

    assert built is True
    assert output_path.exists()
    texts = _paragraph_texts(output_path)
    assert TAILORED.summary in texts
    assert ", ".join(TAILORED.highlighted_skills) in texts


def test_build_tailored_resume_docx_never_uses_the_models_rewritten_bullets(tmp_path):
    """Real risk this guards against: TailoredResume.bullet_points are the
    model's own rewritten text, not grounded to a specific employer/date -
    inserting them into the actual uploaded resume risks misattributing
    fabricated-sounding claims to a real employer. Only the verbatim
    original EXPERIENCE section (with its real bullets) may appear.
    """
    output_path = tmp_path / "tailored.docx"

    build_tailored_resume_docx(STRUCTURED_RESUME, TAILORED, output_path)

    texts = _paragraph_texts(output_path)
    assert TAILORED.bullet_points[0] not in texts


def test_build_tailored_resume_docx_preserves_the_real_work_history_verbatim(tmp_path):
    output_path = tmp_path / "tailored.docx"

    build_tailored_resume_docx(STRUCTURED_RESUME, TAILORED, output_path)

    texts = _paragraph_texts(output_path)
    assert "Software Engineer, Acme Corp, 2020-Present" in texts
    assert "- Built and maintained REST APIs handling 100,000+ requests daily." in texts
    assert "B.S. Computer Science, State University, 2016-2020" in texts


def test_build_tailored_resume_docx_declines_when_no_summary_header_is_found(tmp_path):
    """Real bug this guards against: guessing section boundaries in an
    unstructured resume risks silently dropping real content (work
    history, education) from the uploaded document - declining outright
    is safer than a wrong guess. Caller (write_tailored_resume_docx()) must
    fall back to the user's own unmodified resume file in this case.
    """
    unstructured = "Experienced backend engineer skilled in Python."
    output_path = tmp_path / "tailored.docx"

    built = build_tailored_resume_docx(unstructured, TAILORED, output_path)

    assert built is False
    assert not output_path.exists()


def test_build_tailored_resume_docx_declines_when_no_experience_header_is_found(tmp_path):
    no_experience_section = "Jane Doe\n\nSUMMARY\nA summary.\n\nSKILLS\nPython\n\nEnd of resume."
    output_path = tmp_path / "tailored.docx"

    built = build_tailored_resume_docx(no_experience_section, TAILORED, output_path)

    assert built is False
    assert not output_path.exists()


def test_build_tailored_resume_docx_tolerates_pdf_ligature_split_headers(tmp_path):
    """Confirmed against a real resume.pdf parsed by this project: PDF text
    extraction can split a header's own letters across a stray space
    (kerning/ligature artifacts), e.g. "SUMMAR Y" and "EDUCA TION" instead
    of "SUMMARY"/"EDUCATION" - section detection must still work.
    """
    split_headers = (
        "Jane Doe\n\n"
        "PROFESSIONAL  SUMMAR Y\n"
        "A summary.\n\n"
        "SKILLS\n"
        "Python\n\n"
        "WORK EXPERIENCE\n"
        "Software Engineer, Acme Corp\n"
        "- Did real work.\n"
    )
    output_path = tmp_path / "tailored.docx"

    built = build_tailored_resume_docx(split_headers, TAILORED, output_path)

    assert built is True
    texts = _paragraph_texts(output_path)
    assert "- Did real work." in texts
