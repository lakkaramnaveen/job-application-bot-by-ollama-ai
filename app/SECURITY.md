# Security

## Reporting a vulnerability

Open a GitHub issue describing the class of problem without a working exploit, or contact the maintainer privately if the repo has private vulnerability reporting enabled.

## Threat model, honestly stated

This is an agentic workflow: an LLM reads untrusted web content (job postings) alongside your personal data (resume, FAQ answers, application history) and can act on both. That combination is the main risk surface. What this project does about it:

- **Untrusted-input rule**: `job_bot/matching/scorer.py`, `job_bot/generation/*.py`, and `job_bot/browser/linkedin_adapter.py` all treat scraped job-posting text and application-question text as *data*, never as instructions - every system prompt says so explicitly, and postings/questions are always placed in the user turn, never concatenated into the system prompt. The browser adapter also never follows a link found inside posting text; `load_description()` reads only the description element of the page the user's own search already navigated to.
- **Categorical eligibility gate is enforced in code, not just prompted**: `job_bot/matching/scorer.py` forces `should_apply = False` whenever the model's own `eligibility` verdict is `"fail"` (e.g. a posting explicitly requires citizenship or an existing clearance the resume gives no sign of), regardless of what the model set `should_apply` to. A prompt injection or a model mistake in the surrounding fields can't undo this specific check.
- **No stored platform credentials**: `job_bot/browser/session.py` uses a Playwright persistent profile - you log into LinkedIn once, manually, in a real browser window. The bot never sees or stores the password.
- **Personal data stays local**: resume, tracker database, browser session, blacklist, and audit log all live under `app/data/`, which is gitignored (see `.gitignore` at the repo root). The only network calls this project makes are to the configured LLM provider (Claude API or your own local Ollama server) and to LinkedIn itself via the browser.
- **Secret redaction in logs**: `job_bot/safety/audit_log.py` regex-redacts Anthropic API key and Bearer-token shapes from every logged value before it's written, in case a caller accidentally passes one through.
- **Hard-coded safety ceiling**: `job_bot/config.py`'s `HARD_DAILY_APPLICATION_CEILING` (50) is enforced in `RateLimiter` independent of whatever `DAILY_APPLICATION_CAP` a user configures - a misconfigured or runaway `.env` can't remove the daily cap entirely.

Instruction-level defenses raise the bar; they are not a sandbox. If you point this at a job board or company career page you don't trust at all, use `--dry-run` first and review what got filled in before you let it actually submit.

## Scope notes

- LinkedIn's Terms of Service restrict automated use of the site. Running this tool is a decision you're making about your own account, not a claim this project makes on your behalf that it's compliant - see the "Before you use this" section of `README.md`.
- This project makes no attempt to evade LinkedIn's bot detection. It doesn't fingerprint-spoof, rotate proxies, or otherwise disguise its traffic as something other than an automated browser session - see `job_bot/browser/linkedin_adapter.py`'s module docstring.
