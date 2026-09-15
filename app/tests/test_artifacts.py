from datetime import date

import pytest

from job_bot.generation.artifacts import UnsafeJobId, write_cover_letter, write_tailored_resume
from job_bot.models.schemas import CoverLetter, TailoredResume

TODAY = date.today().isoformat()


def test_write_tailored_resume_creates_readable_file(tmp_path):
    tailored = TailoredResume(
        summary="Backend engineer with 5 years of Python experience.",
        highlighted_skills=["Python", "Django", "PostgreSQL"],
        bullet_points=["Built a payments service handling 1M req/day"],
    )

    path = write_tailored_resume(tmp_path, "job123", tailored)

    assert path == tmp_path / TODAY / "job123" / "tailored_resume.txt"
    content = path.read_text(encoding="utf-8")
    assert "Backend engineer with 5 years" in content
    assert "- Python" in content
    assert "- Built a payments service handling 1M req/day" in content


def test_write_cover_letter_creates_readable_file(tmp_path):
    letter = CoverLetter(body="Dear Hiring Manager,\n\nI'm excited to apply...")

    path = write_cover_letter(tmp_path, "job123", letter)

    assert path == tmp_path / TODAY / "job123" / "cover_letter.txt"
    assert path.read_text(encoding="utf-8") == letter.body


def test_both_artifacts_share_the_same_job_directory(tmp_path):
    write_tailored_resume(
        tmp_path, "job123", TailoredResume(summary="s", highlighted_skills=[], bullet_points=[])
    )
    write_cover_letter(tmp_path, "job123", CoverLetter(body="b"))

    job_dir = tmp_path / TODAY / "job123"
    assert sorted(p.name for p in job_dir.iterdir()) == ["cover_letter.txt", "tailored_resume.txt"]


def test_writes_land_under_a_dated_folder_for_todays_date(tmp_path):
    """The point of the date folder: browsing base_dir on disk (e.g. a
    folder on the Desktop) reads as one folder per day's worth of
    applications, not a flat pile of job-id-named folders.
    """
    write_cover_letter(tmp_path, "job123", CoverLetter(body="b"))

    assert [p.name for p in tmp_path.iterdir()] == [TODAY]


def test_job_folder_is_labeled_with_company_and_title_when_given(tmp_path):
    path = write_cover_letter(
        tmp_path, "job123", CoverLetter(body="b"), company="Acme Corp", title="Backend Engineer"
    )

    assert path == tmp_path / TODAY / "job123 - Acme Corp - Backend Engineer" / "cover_letter.txt"


def test_job_folder_has_no_label_when_company_and_title_are_omitted(tmp_path):
    path = write_cover_letter(tmp_path, "job123", CoverLetter(body="b"))

    assert path == tmp_path / TODAY / "job123" / "cover_letter.txt"


def test_job_folder_label_strips_unsafe_characters_from_scraped_text(tmp_path):
    """company/title are scraped, untrusted LinkedIn text too - unlike
    job_id, they're only ever a cosmetic label appended to the
    already-validated job_id (never the sole or leading path component),
    but they still must never let a crafted value break out of base_dir or
    otherwise corrupt the path.
    """
    path = write_cover_letter(
        tmp_path,
        "job123",
        CoverLetter(body="b"),
        company="../../etc",
        title="Role/With\\Slashes",
    )

    assert path.parent.parent == tmp_path / TODAY
    assert ".." not in path.parts
    assert "/" not in path.parent.name
    assert "\\" not in path.parent.name


def test_job_folder_label_is_truncated_for_an_absurdly_long_title(tmp_path):
    path = write_cover_letter(tmp_path, "job123", CoverLetter(body="b"), title="X" * 500)

    assert len(path.parent.name) < 120


@pytest.mark.parametrize(
    "malicious_job_id",
    [
        "../../etc/passwd",
        "..",
        ".",
        "../sibling",
        "job/../../escape",
        "job/with/slash",
        "job\\with\\backslash",
        "",
    ],
)
def test_write_tailored_resume_rejects_path_traversal_job_id(tmp_path, malicious_job_id):
    """job_id comes from a scraped LinkedIn data-job-id attribute - untrusted
    data - and must never be usable to write outside base_dir.
    """
    before = set(tmp_path.rglob("*"))

    with pytest.raises(UnsafeJobId):
        write_tailored_resume(
            tmp_path,
            malicious_job_id,
            TailoredResume(summary="s", highlighted_skills=[], bullet_points=[]),
        )

    assert set(tmp_path.rglob("*")) == before  # nothing was written or created


def test_write_cover_letter_rejects_path_traversal_job_id(tmp_path):
    with pytest.raises(UnsafeJobId):
        write_cover_letter(tmp_path, "../escape", CoverLetter(body="b"))


def test_realistic_linkedin_job_id_is_accepted(tmp_path):
    # Real LinkedIn job IDs are purely numeric, e.g. "3812345678".
    path = write_cover_letter(tmp_path, "3812345678", CoverLetter(body="b"))
    assert path == tmp_path / TODAY / "3812345678" / "cover_letter.txt"
