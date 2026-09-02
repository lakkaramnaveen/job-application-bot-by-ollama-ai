from job_bot.generation.artifacts import write_cover_letter, write_tailored_resume
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
