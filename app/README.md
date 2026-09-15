# job_bot

[![app CI](https://github.com/lakkaramnaveen/job-application-bot-by-ollama-ai/actions/workflows/app-ci.yml/badge.svg)](https://github.com/lakkaramnaveen/job-application-bot-by-ollama-ai/actions/workflows/app-ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A local, open-source job-application assistant: it scores job postings against
your resume, tailors your resume and cover letter per job, answers unfamiliar
application questions, and fills out (and optionally submits) LinkedIn Easy
Apply forms in a real Chrome window on your own machine.

It can run on the Claude API or on a free local model through
[Ollama](https://ollama.com) (DeepSeek, Llama, GLM, Qwen, or anything else you
pull) - pick one per run with `--provider`.

## Before you use this

- **This automates your own, already-authenticated browser session.** You log
  into LinkedIn once, manually, in a window this tool opens. It never sees or
  stores your password.
- **It does not try to evade LinkedIn's bot detection.** It does not do
  anything to disguise itself as a different kind of traffic. Automating job
  applications may violate the Terms of Service of LinkedIn or other job
  boards - that's a real risk you're taking on by running this, independent of
  anything this tool does or doesn't do to reduce detectability.
- **Submission is confirmed by default.** Every application pauses for a
  yes/no prompt before the final Submit click. Running with no terminal
  attached to answer that prompt from (cron, a pipe, CI) is treated as "no" -
  it never silently assumes yes - so pass `--yes-i-understand-the-risk` for
  any unattended run. There's also a hard daily cap (`DAILY_APPLICATION_CAP`
  in `.env`, capped in code at 50 no matter what you set) so a bug or a bad
  match-score threshold can't spam applications.
- **Selectors may need tuning.** LinkedIn's page structure isn't public and
  changes over time. If a run stops finding a button/field it used to find,
  check `job_bot/browser/linkedin_adapter.py`'s `SELECTORS` dict first, and use
  `--dry-run` while you fix it.
- **Categorical eligibility exclusions are a hard stop, not a fit question.**
  Before scoring a job's fit, the LLM checks the posting for an explicit
  citizenship/permanent-residency/clearance requirement your resume gives no
  sign you hold. If it finds one, `job_bot/matching/scorer.py` forces the job
  to be skipped in code - a high fit score elsewhere can't override it. See
  `SECURITY.md` for the full threat model.

## Setup on macOS

```bash
cd app
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"    # installs job_bot + the job-bot CLI entry point
playwright install chromium
```

`pip install -e ".[dev]"` also gives you `ruff` (lint), `mypy` (types), and
`pytest`. If you only want to run the bot (not develop it), `pip install -e .`
is enough.

Optional, for the free local-model path:

```bash
brew install ollama
ollama serve &                 # leave running in a terminal, or run as a background service
ollama pull deepseek-r1:8b     # or llama3.1:8b, glm4:9b, qwen2.5:7b, ...
```

Configure:

```bash
cp .env.example .env
```

Edit `.env`:
- Set `LLM_PROVIDER` to `claude` or `ollama`.
- If using Claude, set `ANTHROPIC_API_KEY` (get one at console.anthropic.com).
- If using Ollama, set `OLLAMA_MODEL` to whatever you pulled above.
- Put your resume at the path in `RESUME_PATH` (PDF, DOCX, or TXT).

## Running it

`pip install -e .` puts a `job-bot` command on your PATH (inside the venv);
`python -m job_bot.cli` works identically if you prefer that form.

1. **Log into LinkedIn once** (opens a real Chrome window; log in there manually):
   ```bash
   job-bot login
   ```
2. **Check your local setup for common problems** - resume file present, an
   API key/Ollama URL configured, a LinkedIn session saved, the daily cap
   sane. File/config checks only, no network call, so it's fast and safe to
   run any time something seems off:
   ```bash
   job-bot doctor
   ```
3. **Sanity-check your LLM provider** with one real API call (what `doctor`
   above deliberately doesn't do):
   ```bash
   job-bot test-provider
   ```
4. **Dry run** - does everything (search, score, tailor, fill the form,
   attach your resume) except the actual submit click, so you can verify it's
   making sensible decisions:
   ```bash
   job-bot run --keywords "backend engineer" --location "Austin, TX" --max-apps 3 --dry-run
   ```
5. **For real**, once you trust it:
   ```bash
   job-bot run --keywords "backend engineer" --location "Austin, TX" --max-apps 5
   ```
   Each application still pauses for your confirmation unless you pass
   `--yes-i-understand-the-risk` (the daily cap still applies either way).

Other useful flags on `run`:
- `--search-pool N` - how many Easy-Apply postings to fetch/score before
  filtering down to `--max-apps` (default 25; pages through LinkedIn's search
  results and skips postings already marked "Applied").
- `--headless` - run without a visible browser window, for unattended runs
  after you've verified the flow with `--dry-run`.

**Getting better-quality matches** - three flags (each also settable as a
persistent default in `.env` - see `.env.example`), applied in this order:
1. `--experience-level mid-senior,director` - restricts the LinkedIn search
   itself to these seniority levels (`internship`/`entry`/`associate`/
   `mid-senior`/`director`/`executive`), so junior/entry postings never even
   enter the pool - cheaper and more reliable than scoring everything and
   hoping the model rejects the wrong level.
2. `--exclude-title-keywords "forward deployed,sales engineer"` - skips a
   posting whose title contains any of these (case-insensitive) before it's
   scored at all, for titles that keyword search turns up but aren't
   actually the role you do.
3. `--min-score 75` - an extra floor on top of the model's own `should_apply`
   verdict; a posting only gets applied to if the model said yes *and* its
   score clears this. Use this if the model's own bar (it's told to say no
   below 60) feels too generous in practice. Raising this takes effect
   immediately even for postings a previous run already scored and marked
   worth applying to - it doesn't just apply going forward - so tightening
   it after the fact won't leave weak matches from before sitting in the
   queue to be applied to on the next run.

```bash
job-bot run --keywords "Full Stack Engineer" --location "United States" \
  --experience-level mid-senior,director --min-score 75 \
  --exclude-title-keywords "forward deployed,sales engineer" --dry-run
```

Switch providers per run without editing `.env`:
```bash
job-bot run --provider ollama --model deepseek-r1:8b --dry-run
job-bot run --provider claude --model claude-opus-5 --dry-run
```

For every job that passes the fit/eligibility gate (even on a `--dry-run`),
`job-bot run` writes a tailored resume and cover letter to
`<APPLICATIONS_DIR>/<today's date>/<job id - company - title>/` - one dated
folder per day's worth of applications, for you to read, copy from, or reuse
in interview prep. `APPLICATIONS_DIR` defaults to `data/applications` inside
the repo; point it at a folder outside the repo (e.g. on your Desktop) in
`.env` if you want to browse it directly day by day. The file actually
uploaded to the LinkedIn form is still always your own `RESUME_PATH`
document, unedited - the generated tailored resume is a reference artifact,
not something auto-substituted into a real submission without your review.
See `job_bot/generation/artifacts.py`.

Unlike the resume, the generated cover letter *is* used directly in the
submission: if the Easy Apply form has a "Cover letter" text field, it's
filled with the generated text (a text field is per-application content
you can always edit or clear before submitting, unlike a formal resume
document) - see `_looks_like_cover_letter_field()` in `linkedin_adapter.py`.

## Browser profile

By default, `job-bot login`/`job-bot run` launch a Chromium instance with
its own isolated profile (`BROWSER_PROFILE_DIR`, default
`data/browser_profile`) - not your everyday Chrome. You log into LinkedIn
there once and every future run reuses that saved session. This is
deliberate: that profile only ever holds whatever cookies LinkedIn (or
another job board) sets, never your other logged-in sessions (email,
banking, ...) - so if a selector ever misfires or something goes wrong, the
blast radius is limited to LinkedIn. See `SECURITY.md`.

`BROWSER_CDP_URL` exists to attach to an already-running Chrome over the
Chrome DevTools Protocol instead - **but as of Chrome 136 (2025), Chrome
itself refuses to open a remote-debugging port on your default profile at
all**, specifically to prevent another process from draining a real
session's cookies through it. `--remote-debugging-port` only takes effect
alongside a non-default `--user-data-dir`, which means whatever it attaches
to is a fresh, empty profile - not your regular, already-logged-in Chrome.
There's no way around this on current Chrome; it isn't a job-bot
limitation. In practice this setting is only useful if you already keep a
separate, permanently-running debug Chrome instance for other tooling and
want job-bot to share its (already job-board-only) session - not as a way
to reuse your everyday browser. For everyone else, the isolated profile
above is the only real option, and it only costs you one extra login.

If you do have such an instance, point `BROWSER_CDP_URL` at it in `.env`
(e.g. `http://localhost:9222`). **Understand the tradeoff before enabling
this**: an open debugging port gives *any* local process on your machine
full control over that browser window and read access to every cookie in
whatever profile it's debugging - a real reduction in isolation for that
profile, not just a convenience setting. `job-bot login`/`job-bot run` will
reuse whatever context is already open rather than closing it when they
finish, since it's a browser they attached to, not one they launched.

## Tracking outcomes

`job-bot run` only ever writes `seen`, `applied`, or `skipped` - it has no way
to observe what happens after you submit. Record what you hear back by hand:

```bash
job-bot report                          # counts of tracked jobs by status
job-bot status <job_id> interviewing    # or: offer, rejected, withdrawn, no_response
job-bot export                          # all tracked jobs as CSV, to stdout
job-bot export --status applied --out applied.csv
```

`<job_id>` is the LinkedIn job id, printed by `job-bot run` and visible in
`data/audit.log`. Valid statuses are listed in
`job_bot.tracker.db.TRACKER_STATUSES`.

Answers `job-bot run` is confident in (grounded in your resume/FAQ, above
`FAQ_SAVE_CONFIDENCE` in `.env`) are automatically cached to `FAQ_PATH` for
reuse on future applications - the bot gets faster and more consistent the
more you use it, without ever caching a low-confidence guess.

## Company blacklist

```bash
job-bot blacklist add "Company Name"      # job-bot run will always skip it
job-bot blacklist remove "Company Name"
job-bot blacklist list
```

Backed by `BLACKLIST_PATH` (default `data/company_blacklist.json`); matching
is case-insensitive.

## Dashboard

A live, one-page view of every tracked application:

```bash
job-bot dashboard              # opens http://127.0.0.1:8765 in your browser
job-bot dashboard --port 9000 --no-open
```

It's a local HTTP server (bound to `127.0.0.1` only, never your network) that
queries `data/job_bot.sqlite3` directly and polls itself every 5 seconds - no
build step, no separate frontend, nothing to deploy. Shows title, company, fit
score, status (color-coded), and applied date for every job `job-bot run` has
seen, and lets you:

- **Search** by title or company, and **filter** by status.
- **Sort** by newest/oldest, recently applied, fit score, company, or title.
- **Update a job's status inline** from the row - no need to drop to
  `job-bot status <job_id> <status>` for a quick correction.
- **View a job's Q&A history** (every application-question answer the bot
  gave, and what it was based on) in a modal, without querying the DB by hand.

The dashboard has no login (it's a local tool over your own data), so the one
state-changing endpoint (the inline status update) only accepts same-origin
requests - see `job_bot/dashboard/server.py`'s module docstring.

## Gmail sync

Recruiters reply by email, not through LinkedIn - `job-bot gmail-sync` reads
your recent Gmail, classifies each message with the LLM (interview invite /
rejection / offer / application confirmation / not job-related), matches it
to a tracked application by company name, and updates its status. It **never**
updates on an ambiguous match (multiple or zero tracked jobs match the
guessed company), a low-confidence classification, or a job already in a
terminal status (`offer`/`rejected`/`withdrawn`/`no_response`) - see
`job_bot/integrations/gmail_sync.py`'s module docstring for the exact rules.
A job moved to `applied` this way - or by hand, via `job-bot status <id>
applied` - gets the same applied-date bookkeeping a real submission through
`job-bot run` gets, so it's correctly picked up by `job-bot report`'s
follow-up nudges (see Tracking outcomes above).

**One-time setup** (Google Cloud Console):
1. Create or pick a project at [console.cloud.google.com](https://console.cloud.google.com).
2. Enable the **Gmail API** (APIs & Services -> Library -> search "Gmail API" -> Enable).
3. Configure the OAuth consent screen (External is fine for personal use;
   add your own Gmail address as a test user).
4. Create credentials -> OAuth client ID -> Application type **Desktop app**.
5. Download the JSON and save it to the path in `GMAIL_CREDENTIALS_PATH`
   (default `data/gmail_credentials.json`).

**Usage:**

```bash
job-bot gmail-sync --dry-run     # see what would change, writes nothing
job-bot gmail-sync               # first run opens a browser for the Google OAuth consent screen
job-bot gmail-sync --days 30 --max-emails 100
```

The first run opens a browser tab for you to grant **read-only** access
(`gmail.readonly` - this tool cannot send, delete, or modify mail) and saves
a refresh token to `GMAIL_TOKEN_PATH` (`data/gmail_token.json`) so you won't
be prompted again. Both files are gitignored; see `SECURITY.md`.

## Data storage

Everything stays local, under `app/data/` (gitignored):
- `data/job_bot.sqlite3` - job tracker, Q&A history, daily application counter
- `data/browser_profile/` - your persisted Chrome login session
- `data/audit.log` - a redacted, append-only log of every action taken
- `data/failed_applications.log` - just the postings a run couldn't finish
  (with the reason) - the same events are also in `data/audit.log`, but this
  file skips the search/scored/applied noise so it's readable on its own
  after a run to see what actually needs fixing
- `data/company_blacklist.json` - companies to always skip
- `data/faq_answers.json` - previously given answers, reused as context
- `data/gmail_credentials.json` / `data/gmail_token.json` - your Gmail OAuth
  client and refresh token, if you've set up Gmail sync
- `data/applications/<today's date>/<job id - company - title>/` - the
  tailored resume and cover letter generated for each job you passed the fit
  gate on, one dated folder per day (path configurable via
  `APPLICATIONS_DIR` - see "Getting better-quality matches" above)

Your `.env` (API keys) and everything in `data/` never leave your machine
except for the LLM API calls you configure.

## Development

```bash
source .venv/bin/activate
pytest                                          # tests
pytest --cov=job_bot --cov-report=term-missing  # tests + coverage
ruff check .                                    # lint
ruff format .                                   # formatting
mypy job_bot                                    # type check
```

Optionally, `pre-commit install` (config in `.pre-commit-config.yaml`) runs
ruff automatically on every commit. CI (`.github/workflows/app-ci.yml`, at the
repo root) runs lint + mypy + the full pytest suite (with Playwright's browser
installed, and a coverage floor of 70% enforced via `[tool.coverage.report]`
in `pyproject.toml`) on Python 3.11-3.13 on every push/PR touching `app/`,
plus a dependency-review check on PRs.

Tests are fully offline: the Claude provider is tested against a mocked SDK
client, the Ollama provider against a mocked HTTP server (`respx`), the
LinkedIn search/form-filling logic against local static HTML fixtures
(`tests/fixtures/`), the Gmail client against a fake `googleapiclient`
Resource (no OAuth flow, no network), and the dashboard against a real
`ThreadingHTTPServer` bound to an ephemeral localhost port - no test ever
calls a real API, touches linkedin.com, or hits Google's servers.

## Extending to other job boards

Implement `job_bot.browser.base_adapter.JobBoardAdapter` (`search()` and
`fill_and_submit()`) the way `linkedin_adapter.py` does, then wire it up in
`cli.py`. The LLM-facing code (scoring, tailoring, Q&A) is board-agnostic and
needs no changes.

## Security

See `SECURITY.md` for the full threat model (untrusted-input handling,
credential storage, the eligibility gate, data boundaries).

## Acknowledgements

This project's eligibility-gate concept, prompt-injection posture, and
CI/security-guard shape were informed by
[MadsLorentzen/ai-job-search](https://github.com/MadsLorentzen/ai-job-search),
a Claude-Code-based job-application framework with a different architecture
(human-reviewed LaTeX CV/cover-letter generation, no auto-submit) but several
directly portable ideas. Not affiliated with that project.
