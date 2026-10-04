"""prepare_materials(): the tailored resume and cover letter for one
posting - see job_bot/pipeline/materials.py."""

import json

from job_bot.browser.base_adapter import JobPosting
from job_bot.models.schemas import CoverLetter, TailoredResume
from job_bot.pipeline.materials import Materials, prepare_materials
from job_bot.safety.audit_log import AuditLogger
from job_bot.tracker.db import Tracker

POSTING = JobPosting(job_id="j1", title="Backend Engineer", company="Acme", url="https://example.com/j1", description="")


class FakeModel:
    def __init__(self):
        self.schemas = []

    def generate_structured(self, *, system, prompt, schema):
        self.schemas.append(schema)
        if schema is TailoredResume:
            return TailoredResume(summary="Backend engineer.", highlighted_skills=["Python"], bullet_points=["Built APIs."])
        return CoverLetter(body="Dear hiring team, ...")


def test_writes_records_and_returns_the_materials(tmp_path):
    resume = tmp_path / "resume.txt"
    resume.write_text("Plain text resume, 5 years of Python.", encoding="utf-8")
    tracker = Tracker(tmp_path / "t.sqlite3")
    audit_path = tmp_path / "audit.log"
    model = FakeModel()

    materials = prepare_materials(
        POSTING,
        "We need a Python backend engineer.",
        provider=model,
        resume_text=resume.read_text(),
        resume_path=resume,
        applications_dir=tmp_path / "applications",
        tracker=tracker,
        audit=AuditLogger(audit_path),
    )

    assert isinstance(materials, Materials)
    assert materials.cover_letter.body == "Dear hiring team, ..."
    # A plain-text resume can't be rebuilt as a tailored .docx, so the user's own file is uploaded.
    assert materials.resume_path == str(resume)
    assert model.schemas == [TailoredResume, CoverLetter]
    assert tracker.get_resume_generation("j1")["summary"] == "Backend engineer."
    assert any((tmp_path / "applications").rglob("*"))  # reference copies written
    assert [json.loads(line)["action"] for line in audit_path.read_text().splitlines()] == ["generated_materials"]
