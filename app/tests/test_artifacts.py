import pytest

from job_bot.generation.artifacts import UnsafeJobId, write_cover_letter, write_tailored_resume
from job_bot.models.schemas import CoverLetter, TailoredResume


def test_write_tailored_resume_creates_readable_file(tmp_path):
    tailored = TailoredResume(
        summary="Backend engineer with 5 years of Python experience.",
        highlighted_skills=["Python", "Django", "PostgreSQL"],
        bullet_points=["Built a payments service handling 1M req/day"],
    )

    path = write_tailored_resume(tmp_path, "job123", tailored)

    assert path == tmp_path / "job123" / "tailored_resume.txt"
    content = path.read_text(encoding="utf-8")
    assert "Backend engineer with 5 years" in content
    assert "- Python" in content
    assert "- Built a payments service handling 1M req/day" in content


def test_write_cover_letter_creates_readable_file(tmp_path):
    letter = CoverLetter(body="Dear Hiring Manager,\n\nI'm excited to apply...")

    path = write_cover_letter(tmp_path, "job123", letter)

    assert path == tmp_path / "job123" / "cover_letter.txt"
    assert path.read_text(encoding="utf-8") == letter.body


def test_both_artifacts_share_the_same_job_directory(tmp_path):
    write_tailored_resume(
        tmp_path, "job123", TailoredResume(summary="s", highlighted_skills=[], bullet_points=[])
    )
    write_cover_letter(tmp_path, "job123", CoverLetter(body="b"))

    job_dir = tmp_path / "job123"
    assert sorted(p.name for p in job_dir.iterdir()) == ["cover_letter.txt", "tailored_resume.txt"]


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
    assert path == tmp_path / "3812345678" / "cover_letter.txt"
