import re
from pathlib import Path

import pytest

from job_bot.config import HARD_DAILY_APPLICATION_CEILING, Settings, SettingsError

ENV_EXAMPLE_PATH = Path(__file__).resolve().parent.parent / ".env.example"


def make_settings(tmp_path, **overrides):
    defaults = dict(
        _env_file=None,
        llm_provider="claude",
        anthropic_api_key="sk-ant-fake",
        resume_path=tmp_path / "resume.txt",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def test_validate_ready_passes_with_resume_and_key(tmp_path):
    resume = tmp_path / "resume.txt"
    resume.write_text("Jane Doe\nSoftware Engineer with 5 years of experience.")
    settings = make_settings(tmp_path, resume_path=resume)

    assert settings.validate_ready() == []


def test_validate_ready_fails_when_resume_missing(tmp_path):
    settings = make_settings(tmp_path, resume_path=tmp_path / "missing.txt")

    with pytest.raises(SettingsError, match="Resume file not found"):
        settings.validate_ready()


def test_validate_ready_fails_for_a_present_but_unparseable_resume(tmp_path):
    """Real bug this guards against: a resume file that exists (passing the
    old exists()-only check) but can't actually be parsed - here an empty
    resume.txt - used to only surface once cmd_run's resume_store.resume_text()
    call ran inside the already-open browser_session(), so a real browser
    window opened for a run that was always going to fail on its very
    first cycle.
    """
    resume = tmp_path / "resume.txt"
    resume.write_text("")
    settings = make_settings(tmp_path, resume_path=resume)

    with pytest.raises(SettingsError, match="No extractable text found"):
        settings.validate_ready()


def test_validate_ready_fails_when_claude_key_missing(tmp_path):
    resume = tmp_path / "resume.txt"
    resume.write_text("Jane Doe\nSoftware Engineer with 5 years of experience.")
    settings = make_settings(tmp_path, resume_path=resume, anthropic_api_key=None)

    with pytest.raises(SettingsError, match="ANTHROPIC_API_KEY"):
        settings.validate_ready()


def test_validate_ready_ok_for_ollama_without_api_key(tmp_path):
    resume = tmp_path / "resume.txt"
    resume.write_text("Jane Doe\nSoftware Engineer with 5 years of experience.")
    settings = make_settings(tmp_path, resume_path=resume, llm_provider="ollama", anthropic_api_key=None)

    assert settings.validate_ready() == []


def test_validate_ready_warns_when_cap_exceeds_ceiling(tmp_path):
    resume = tmp_path / "resume.txt"
    resume.write_text("Jane Doe\nSoftware Engineer with 5 years of experience.")
    settings = make_settings(
        tmp_path, resume_path=resume, daily_application_cap=HARD_DAILY_APPLICATION_CEILING + 10
    )

    warnings = settings.validate_ready()
    assert any("exceeds the hard ceiling" in w for w in warnings)


def test_max_years_experience_and_require_w2_default_to_off(tmp_path):
    """Consistent with the rest of the "Role quality" section
    (min_match_score, exclude_title_keywords, default_experience_levels all
    default to "no filter") - a fresh clone with no .env customization
    shouldn't unexpectedly reject postings on either of these.
    """
    settings = make_settings(tmp_path)

    assert settings.max_years_experience is None
    assert settings.require_w2 is False


def test_every_settings_field_is_documented_in_env_example():
    """Real gap this guards against: stale_after_days had no line in
    .env.example at all - not even commented out - so there was no
    discoverable way for a user to know STALE_AFTER_DAYS existed short of
    reading config.py's source directly. Every Settings field must appear
    in .env.example as "FIELD_NAME=" (commented lines count, since several
    optional/off-by-default settings are deliberately shown commented out
    as an example rather than active), so a newly added setting can't
    silently go undocumented the same way again.
    """
    env_text = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
    documented = set(re.findall(r"^#?\s*([A-Z_][A-Z0-9_]*)=", env_text, re.MULTILINE))

    undocumented = sorted(field for field in Settings.model_fields if field.upper() not in documented)
    assert undocumented == []


def test_security_md_states_the_correct_hard_daily_application_ceiling():
    """Real gap this guards against: SECURITY.md's stated hard ceiling
    ("HARD_DAILY_APPLICATION_CEILING (50)") had drifted from the actual
    code value (100) - a wrong number in a doc specifically about this
    project's safety guarantees, exactly the kind of claim a
    security-conscious reader would take at face value without checking
    config.py themselves. Asserts the current constant's value appears in
    the doc's own sentence about it, so a future change to the constant
    without updating the doc fails a test instead of silently shipping a
    stale, misleading number.
    """
    security_text = (Path(__file__).resolve().parent.parent / "SECURITY.md").read_text(encoding="utf-8")
    assert f"HARD_DAILY_APPLICATION_CEILING` ({HARD_DAILY_APPLICATION_CEILING})" in security_text
