from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_DIR = Path(__file__).resolve().parent.parent

# Hard ceiling on daily applications, enforced in code regardless of what a user
# sets in .env. This is a safety backstop against a misconfigured or runaway run.
HARD_DAILY_APPLICATION_CEILING = 50


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(APP_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = Field(default="claude", pattern="^(claude|ollama)$")

    anthropic_api_key: str | None = None
    claude_model: str = "claude-opus-5"

    ollama_model: str = "deepseek-r1:8b"
    ollama_base_url: str = "http://localhost:11434"

    resume_path: Path = APP_DIR / "data" / "resume.pdf"
    faq_path: Path = APP_DIR / "data" / "faq_answers.json"
    blacklist_path: Path = APP_DIR / "data" / "company_blacklist.json"

    daily_application_cap: int = 20
    require_confirm_before_submit: bool = True
    # `job-bot report`'s default for how long an application can sit in
    # "applied" with no reply before it's worth a manual follow-up nudge.
    stale_after_days: int = 14
    # Only answers at/above this confidence AND grounded in the resume/FAQ
    # (not a guess) get cached to faq_path for reuse on future applications -
    # a low-confidence answer getting cached would otherwise compound into
    # future prompts as if it were a verified previous answer.
    faq_save_confidence: float = 0.7

    db_path: Path = APP_DIR / "data" / "job_bot.sqlite3"
    browser_profile_dir: Path = APP_DIR / "data" / "browser_profile"
    audit_log_path: Path = APP_DIR / "data" / "audit.log"
    applications_dir: Path = APP_DIR / "data" / "applications"

    # --- Gmail sync (optional) ---
    gmail_credentials_path: Path = APP_DIR / "data" / "gmail_credentials.json"
    gmail_token_path: Path = APP_DIR / "data" / "gmail_token.json"
    gmail_sync_days: int = 14
    gmail_match_confidence: float = 0.6

    # --- Dashboard ---
    dashboard_port: int = 8765

    def effective_daily_cap(self) -> int:
        """The daily cap actually enforced: never above the hard ceiling."""
        return min(self.daily_application_cap, HARD_DAILY_APPLICATION_CEILING)

    def validate_ready(self) -> list[str]:
        """Fail fast with a clear message before opening a browser window,
        rather than partway through a run. Returns non-blocking warnings;
        raises SettingsError on anything that would prevent the run from
        working at all.
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not self.resume_path.exists():
            errors.append(f"Resume file not found at {self.resume_path} (set RESUME_PATH in .env).")
        if self.llm_provider == "claude" and not self.anthropic_api_key:
            errors.append("LLM_PROVIDER=claude but ANTHROPIC_API_KEY is not set in .env.")
        if self.daily_application_cap > HARD_DAILY_APPLICATION_CEILING:
            warnings.append(
                f"DAILY_APPLICATION_CAP={self.daily_application_cap} exceeds the hard "
                f"ceiling of {HARD_DAILY_APPLICATION_CEILING}; the ceiling will be used instead."
            )

        if errors:
            raise SettingsError("\n".join(f"- {e}" for e in errors))
        return warnings


class SettingsError(RuntimeError):
    pass


def get_settings() -> Settings:
    return Settings()
