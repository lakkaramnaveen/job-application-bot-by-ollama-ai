import pytest

from job_bot.config import HARD_DAILY_APPLICATION_CEILING, Settings, SettingsError


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
