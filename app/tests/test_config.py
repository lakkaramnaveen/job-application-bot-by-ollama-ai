import pytest

from job_bot.config import HARD_DAILY_APPLICATION_CEILING, Settings, SettingsError


def make_settings(tmp_path, **overrides):
    defaults = dict(
        _env_file=None,
        llm_provider="claude",
        anthropic_api_key="sk-ant-fake",
        resume_path=tmp_path / "resume.pdf",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def test_validate_ready_passes_with_resume_and_key(tmp_path):
    resume = tmp_path / "resume.pdf"
    resume.write_text("fake pdf")
    settings = make_settings(tmp_path, resume_path=resume)

    assert settings.validate_ready() == []


def test_validate_ready_fails_when_resume_missing(tmp_path):
    settings = make_settings(tmp_path, resume_path=tmp_path / "missing.pdf")

    with pytest.raises(SettingsError, match="Resume file not found"):
        settings.validate_ready()


def test_validate_ready_fails_when_claude_key_missing(tmp_path):
    resume = tmp_path / "resume.pdf"
    resume.write_text("fake pdf")
    settings = make_settings(tmp_path, resume_path=resume, anthropic_api_key=None)

    with pytest.raises(SettingsError, match="ANTHROPIC_API_KEY"):
        settings.validate_ready()


def test_validate_ready_ok_for_ollama_without_api_key(tmp_path):
    resume = tmp_path / "resume.pdf"
    resume.write_text("fake pdf")
    settings = make_settings(tmp_path, resume_path=resume, llm_provider="ollama", anthropic_api_key=None)

    assert settings.validate_ready() == []


def test_validate_ready_warns_when_cap_exceeds_ceiling(tmp_path):
    resume = tmp_path / "resume.pdf"
    resume.write_text("fake pdf")
    settings = make_settings(
        tmp_path, resume_path=resume, daily_application_cap=HARD_DAILY_APPLICATION_CEILING + 10
    )

    warnings = settings.validate_ready()
    assert any("exceeds the hard ceiling" in w for w in warnings)
