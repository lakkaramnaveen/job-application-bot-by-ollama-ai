"""Application configuration, loaded from app/.env (see .env.example) with
pydantic-settings' usual mapping: each field below is set by the
case-insensitive environment variable of the same name (e.g. `llm_provider`
<- `LLM_PROVIDER`). Every field has a default, so job_bot runs out of the box
with no .env at all except for whatever validate_ready() below flags as
actually required (a resume file, and an API key if using Claude).
"""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from job_bot.resume.parser import ResumeParseError, find_moved_resume, parse_resume

APP_DIR = Path(__file__).resolve().parent.parent

# Hard ceiling on daily applications, enforced in code regardless of what a user
# sets in .env. This is a safety backstop against a misconfigured or runaway
# run (e.g. a scoring bug marking every posting should_apply) - not a normal
# day's operating limit, which DAILY_APPLICATION_CAP in .env controls instead.
HARD_DAILY_APPLICATION_CEILING = 100


class Settings(BaseSettings):
    """All configuration for one job_bot invocation. Constructed once, at CLI
    startup (see cli.py's get_settings() call in main()), then threaded
    explicitly through every command rather than read from a global - keeps
    every function's dependencies visible in its signature and makes tests
    trivial to isolate (see e.g. tests/test_cli_run.py's make_settings()).
    """

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
    # Once `job-bot run` hits today's application cap (or the loop stops for
    # any other reason - browser closed, Ctrl+C), best-effort quit the local
    # Ollama server/app so it stops holding the model in memory for the rest
    # of the day. No effect when llm_provider is "claude", and no effect on
    # a concurrent run/other process still using Ollama for something else -
    # this is meant for the common single-user, single-purpose setup this
    # project assumes. False (default) leaves Ollama running, matching the
    # rest of this section's off-by-default convention.
    quit_ollama_when_done: bool = False

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

    # What `job-bot run` searches LinkedIn for when --keywords/--location
    # aren't given - set once here instead of typing them on every run.
    # The flags still override these for one invocation.
    search_keywords: str = "software engineer"
    search_location: str = "United States"

    # Extra floor `job-bot run` enforces on top of the LLM's own should_apply
    # verdict - a posting only gets applied to if should_apply is True AND
    # match_score >= this. The scorer's own prompt already tells the model
    # to set should_apply=False below 60, but a model can be more generous
    # than that in practice; this is a way to tighten the bar without
    # touching the prompt. 0 means no additional floor beyond the model's
    # own judgment (the prior behavior).
    min_match_score: int = 0
    # Comma-separated, case-insensitive substrings - a posting whose title
    # contains any of these is skipped before it's ever scored (saving an
    # LLM call, not just filtered after the fact). Empty means no filter.
    exclude_title_keywords: str = ""
    # Comma-separated LinkedIn seniority levels (internship/entry/associate/
    # mid-senior/director/executive) to restrict search results to at the
    # source - see linkedin_adapter.py's EXPERIENCE_LEVEL_CODES for the
    # exact values. Empty means no restriction (every level). Note LinkedIn
    # has no separate "mid" bucket - genuine mid-level postings are commonly
    # tagged "mid-senior" alongside actual senior ones, so excluding that
    # bucket entirely tends to filter out real mid-level roles too;
    # max_years_experience below is the finer-grained backstop for that.
    default_experience_levels: str = ""
    # A posting explicitly requiring more years of professional experience
    # than this, or explicitly titled/described as Senior/Staff/Principal/
    # Lead/Director-or-higher, is treated as ineligible regardless of match
    # score - see matching/scorer.py's eligibility check. None (default)
    # disables this check entirely, matching the rest of this section's
    # off-by-default convention.
    max_years_experience: int | None = None
    # A posting explicitly stated as Corp-to-Corp (C2C), 1099, or otherwise
    # not offered as direct W2 employment is treated as ineligible too - see
    # matching/scorer.py's eligibility check. A posting silent on employment
    # type is not excluded by this. False (default) disables the check.
    require_w2: bool = False

    # EXPERIMENTAL, off by default - see README.md's "Applying on company
    # websites (experimental)" section and browser/external_apply_adapter.py's
    # module docstring before turning this on. When True, `job-bot run` also
    # considers postings LinkedIn itself doesn't offer Easy Apply for,
    # following their "Apply on company website" link and attempting a
    # best-effort, heuristic fill on whatever form is actually there -
    # unlike linkedin_adapter.py, there is no single site's DOM this has
    # been tuned against, so it will get real forms wrong sometimes.
    # Confirmation before submitting is always required on this path
    # regardless of REQUIRE_CONFIRM_BEFORE_SUBMIT, since it's far less
    # tested than the LinkedIn flow.
    enable_external_apply: bool = False

    db_path: Path = APP_DIR / "data" / "job_bot.sqlite3"
    browser_profile_dir: Path = APP_DIR / "data" / "browser_profile"
    # Unset (default) means browser_session() launches its own isolated
    # Chromium profile - see that function's docstring and README.md's
    # "Browser profile" section. Setting this attaches to an already-running
    # Chrome (e.g. "http://localhost:9222") instead - not the user's regular
    # browser, since Chrome 136+ refuses to open a debugging port on the
    # default profile, so this only makes sense against a separate,
    # permanently-running debug Chrome the user already maintains. A real
    # security tradeoff regardless, since an open debugging port gives any
    # local process full control over whatever browser it's attached to.
    browser_cdp_url: str | None = None
    audit_log_path: Path = APP_DIR / "data" / "audit.log"
    # A focused, append-only log of just the postings a run couldn't finish
    # (scoring/tailoring failed, or the Easy Apply form couldn't be
    # completed) and why - the same events land in audit_log_path too,
    # interleaved with every other action a run takes, but this file is
    # meant to be grepped/read on its own to see what actually needs fixing
    # after a run, without wading through search/scored/applied noise.
    failed_applications_log_path: Path = APP_DIR / "data" / "failed_applications.log"
    # Required questions Easy Apply couldn't answer confidently and left
    # deliberately unanswered (see browser/linkedin_adapter.py's
    # UnansweredRequiredQuestion) - reviewed and answered once via
    # `job-bot review-answers`, which saves the answer to faq_path so every
    # future posting that asks the same question gets it automatically.
    answer_gaps_path: Path = APP_DIR / "data" / "answer_gaps.json"
    applications_dir: Path = APP_DIR / "data" / "applications"

    # --- Gmail sync (optional) ---
    gmail_credentials_path: Path = APP_DIR / "data" / "gmail_credentials.json"
    gmail_token_path: Path = APP_DIR / "data" / "gmail_token.json"
    gmail_sync_days: int = 14
    gmail_match_confidence: float = 0.6

    # --- Dashboard ---
    dashboard_port: int = 8765

    @field_validator(
        "resume_path",
        "faq_path",
        "blacklist_path",
        "db_path",
        "browser_profile_dir",
        "audit_log_path",
        "failed_applications_log_path",
        "answer_gaps_path",
        "applications_dir",
        "gmail_credentials_path",
        "gmail_token_path",
    )
    @classmethod
    def _anchor_relative_paths_to_app_dir(cls, value: Path) -> Path:
        """A relative path from .env (e.g. .env.example's own
        `DB_PATH=./data/job_bot.sqlite3`) means relative to the app folder,
        where that .env lives and where every default above points - not to
        whatever directory `job-bot` happens to be started from. Before this,
        Path() resolved it against the current directory, so running from
        anywhere else silently used a fresh, empty data folder: no tracker
        history (re-applying to jobs already applied to), no blacklist, and a
        never-logged-in browser profile. Confirmed on a real install - a
        stray ~/data/ with its own empty tracker and unauthenticated profile,
        whose searches all hit LinkedIn's sign-in wall. `~` is expanded too;
        absolute paths are left as-is.
        """
        value = value.expanduser()
        return value if value.is_absolute() else APP_DIR / value

    def effective_daily_cap(self) -> int:
        """The daily cap actually enforced: never above the hard ceiling."""
        return min(self.daily_application_cap, HARD_DAILY_APPLICATION_CEILING)

    def validate_ready(self) -> list[str]:
        """Fail fast with a clear message before opening a browser window,
        rather than partway through a run. Returns non-blocking warnings;
        raises SettingsError on anything that would prevent the run from
        working at all.

        Actually parses the resume, not just checks it exists - a
        present-but-corrupted PDF, an empty resume.txt, or an unsupported
        extension (resume/parser.py's ResumeParseError cases) previously
        passed this check and only surfaced once cmd_run's
        resume_store.resume_text() call ran inside the already-open
        browser_session() - a real browser window opened for a run that
        was always going to fail on its very first cycle.
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not self.resume_path.exists():
            moved = find_moved_resume(self.resume_path)
            hint = (
                f"a file with the same name is at {moved} - set RESUME_PATH to that in .env"
                if moved is not None
                else "set RESUME_PATH in .env"
            )
            errors.append(f"Resume file not found at {self.resume_path} ({hint}).")
        else:
            try:
                parse_resume(self.resume_path)
            except ResumeParseError as e:
                errors.append(str(e))
        if self.llm_provider == "claude" and not self.anthropic_api_key:
            errors.append("LLM_PROVIDER=claude but ANTHROPIC_API_KEY is not set in .env.")
        if self.daily_application_cap > HARD_DAILY_APPLICATION_CEILING:
            warnings.append(
                f"DAILY_APPLICATION_CAP={self.daily_application_cap} exceeds the hard "
                f"ceiling of {HARD_DAILY_APPLICATION_CEILING}; the ceiling will be used instead."
            )
        elif self.daily_application_cap <= 0:
            # RateLimiter.remaining_today() clamps a non-positive cap to 0
            # remaining, and record_application() then raises DailyCapReached
            # on the very first job - not a crash, but a `job-bot run` that
            # opens a real browser window, searches, scores, and then applies
            # to nothing at all, for a reason this function's own docstring
            # says it exists to catch before that browser window ever opens:
            # a misconfigured/mistyped DAILY_APPLICATION_CAP (0, or a
            # negative value) rather than one deliberately set too high.
            warnings.append(
                f"DAILY_APPLICATION_CAP={self.daily_application_cap} means job-bot run will "
                "never apply to anything today - set it above 0 in .env if that's not intended."
            )

        if errors:
            raise SettingsError("\n".join(f"- {e}" for e in errors))
        return warnings


class SettingsError(RuntimeError):
    pass


def get_settings() -> Settings:
    return Settings()
