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

## Quickstart

The fastest path from a fresh clone to a real (but safe) run - each step
links to the section with the full detail:

1. **Install** - Python venv, `pip install -e .`, `playwright install
   chromium`. See "Setup on macOS" below.
2. **Pick a model provider** - Claude (an API key, costs money per call) or
   Ollama (free, runs on your machine). See "Local Ollama vs Claude
   (cloud)" below if you're not sure which.
3. **Configure** - `cp .env.example .env`, then set `LLM_PROVIDER` and
   either `ANTHROPIC_API_KEY` or `OLLAMA_MODEL` to match.
4. **Add your resume** - put the file at the path `RESUME_PATH` in `.env`
   points to (default `./data/resume.pdf`, relative to `app/`) - PDF, DOCX,
   or TXT. This is the file that's actually uploaded to every real
   application; nothing here ever generates a fake one to submit in its
   place.
5. **Log in once** - `job-bot login` opens a real Chrome window for you to
   sign into LinkedIn by hand; the session is saved for every future run.
6. **Sanity-check your setup** - `job-bot doctor` (local file/config
   checks) and `job-bot test-provider` (one real API/Ollama call).
7. **Dry run** - `job-bot run --keywords "..." --location "..." --dry-run`
   does everything except the final Submit click, so you can see what it
   would have done.
8. **For real** - drop `--dry-run`, add `--yes-i-understand-the-risk` to
   skip the per-application confirmation prompt, add `--loop` to keep
   applying all day instead of stopping after one batch. See "Running it"
   below for the full flag reference.

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
  in `.env`, capped in code at `HARD_DAILY_APPLICATION_CEILING` - currently
  100 - no matter what you set) so a bug or a bad match-score threshold
  can't spam applications.
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

### Local Ollama vs Claude (cloud)

Every LLM call in this project (scoring, tailoring, answering questions) goes
through one interchangeable interface (`job_bot/llm/base.py`), so switching
providers is just an `.env` value or a `--provider`/`--model` flag - nothing
else about how the bot behaves changes.

| | Ollama (local) | Claude (cloud) |
|---|---|---|
| Cost | Free - runs on your own machine | Pay per API call (small, but it adds up over hundreds of postings) |
| Privacy | Your resume and every job posting text never leave your machine | Sent to Anthropic's API per their terms |
| Setup | `brew install ollama` + `ollama pull <model>`, no account | An Anthropic API key (console.anthropic.com) |
| Quality/reliability | Good models handle this fine, but a local model is more likely to return malformed output occasionally (retried automatically, see `ollama_provider.py`) or need `--min-score` tightened if it scores too generously | Generally more consistent structured-output and judgment quality out of the box |
| Speed | Depends entirely on your hardware - a small model (`deepseek-r1:8b`, `llama3.1:8b`) is fast on any recent Mac; a larger model needs real RAM to stay fast | Consistent, not dependent on your machine |

If you have the RAM for it (16GB+ unified memory), a mid-size model like
`qwen3:30b` (an MoE model - only ~3B parameters active per token despite
30B total, so it stays fast) is noticeably better at follow-through on
multi-step reasoning (eligibility checks, matching a radio option to an
answer) than the smaller defaults above, at zero cost. If it's a *thinking*
model (qwen3, deepseek-r1, ...), `ollama_provider.py` already passes
`think: false` on every call - only the final structured answer, not the
model's internal reasoning trace, ever mattered here, and skipping it cuts
latency roughly 10x on a thinking model with no quality loss to the answer
itself. See `docs/qwen_notes.md` for the specific failure patterns found
and fixed while running `qwen3:30b` against real applications (reasoning
leaking into an answer field, truncated JSON padding, etc.) and how each
one was addressed.

Switch per run without touching `.env`:
```bash
job-bot run --provider ollama --model qwen3:30b --dry-run
job-bot run --provider claude --model claude-opus-5 --dry-run
```

Configure:

```bash
cp .env.example .env
```

Edit `.env`:
- Set `LLM_PROVIDER` to `claude` or `ollama`.
- If using Claude, set `ANTHROPIC_API_KEY` (get one at console.anthropic.com).
- If using Ollama, set `OLLAMA_MODEL` to whatever you pulled above.
- Optionally set `QUIT_OLLAMA_WHEN_DONE=true` to have `job-bot run` quit the
  local Ollama server/app for you once today's application cap is reached
  (or the loop otherwise stops), so it stops holding the model in memory
  for the rest of the day. Off by default, and has no effect with
  `LLM_PROVIDER=claude`.
- **Put your resume at the path `RESUME_PATH` points to** - default
  `./data/resume.pdf` relative to `app/` (i.e. `app/data/resume.pdf`), or
  point `RESUME_PATH` at a file anywhere else on disk. PDF, DOCX, or TXT.
  When your resume has clearly-labeled SUMMARY/SKILLS/EXPERIENCE-or-similar
  sections, this is what a freshly per-job tailored `.docx` (built from it)
  is uploaded instead of - see "Running it" below for exactly what does and
  doesn't get changed. When those sections can't be confidently found, this
  exact file is uploaded unmodified instead, same as before that feature
  existed.

## Running it

`pip install -e .` puts a `job-bot` command on your PATH (inside the venv);
`python -m job_bot.cli` works identically if you prefer that form.

1. **Log into LinkedIn once** (opens a real Chrome window; log in there manually):
   ```bash
   job-bot login
   ```
2. **Check your local setup for common problems** - the resume file actually
   parses, an API key/Ollama URL configured, a LinkedIn session saved, the
   blacklist/FAQ/answer-gaps files are valid JSON (not silently treated as empty)
   and their directories are actually writable, the applications directory,
   audit log, and failed-applications log are all actually writable, the
   tracker database itself isn't corrupted, Gmail credentials
   present and the one-time OAuth consent actually completed (if you use
   gmail-sync), the daily cap sane and how much of it is already used today.
   File/config checks only, no network
   call, so it's fast and safe to run any time something seems off:
   ```bash
   job-bot doctor
   job-bot doctor --format json   # same checks as one JSON object, for a setup script or health-check cron job
   ```
3. **Sanity-check your LLM provider** with one real API call (what `doctor`
   above deliberately doesn't do):
   ```bash
   job-bot test-provider
   job-bot test-provider --format json   # same result as one JSON object, for a monitoring script
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
- Postings are always searched **freshest first**: `search()` looks at only
  the last 24 hours to start, and only widens to the last 3 days if that
  isn't enough to fill `--search-pool` - never further back than that, so
  you're never spending an application on a posting that's already been up
  (and collecting applicants) for a week or more. Not a flag - this is
  always on, since there's no good reason to prefer a stale posting over a
  fresh one when LinkedIn itself supports filtering for it.
- `--headless` - run without a visible browser window, for unattended runs
  after you've verified the flow with `--dry-run`.
- `--loop` - a plain run stops after one search batch (`--search-pool`
  postings) or `--max-apps` successful applications, whichever comes first -
  fine for a quick check, but it means the run ends long before the day's
  application cap does, and never sees a job posted later in the day.
  `--loop` instead keeps re-searching and applying in cycles (every
  `--loop-interval-minutes`, default 20) until today's `DAILY_APPLICATION_CAP`
  is reached or you stop it with Ctrl+C - `--max-apps` then caps applications
  *per cycle*, not for the whole day. Reuses the same browser session across
  cycles rather than reopening Chromium each time.
  ```bash
  job-bot run --keywords "Full Stack Engineer" --location "United States" \
    --loop --loop-interval-minutes 20 --max-apps 5 --yes-i-understand-the-risk
  ```

**Getting better-quality matches** - five flags (each also settable as a
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
3. `--max-years-experience 6` - rejects a posting outright (regardless of
   score) if the model reads it as explicitly titled/described as Senior/
   Staff/Principal/Lead/Director-or-higher, or explicitly requiring more
   years of experience than this. Useful alongside `--experience-level`
   rather than instead of it: LinkedIn's own seniority facet has no separate
   "mid" bucket - it bundles real mid-level postings in with senior ones
   under "mid-senior" - so targeting entry-to-mid without losing genuine
   mid-level roles means including `mid-senior` in `--experience-level` and
   letting this flag reject the truly senior ones within it.
4. `--require-w2` - rejects a posting outright if the model reads it as
   explicitly Corp-to-Corp (C2C), 1099, or otherwise not offered as direct
   W2 employment. A posting silent on employment type is not affected.
5. `--min-score 75` - an extra floor on top of the model's own `should_apply`
   verdict; a posting only gets applied to if the model said yes *and* its
   score clears this. Use this if the model's own bar (it's told to say no
   below 60) feels too generous in practice. Raising this takes effect
   immediately even for postings a previous run already scored and marked
   worth applying to - it doesn't just apply going forward - so tightening
   it after the fact won't leave weak matches from before sitting in the
   queue to be applied to on the next run.

`--max-years-experience`/`--require-w2` are assessed by the model reading
the actual posting text (the same way the always-on citizenship/clearance
check already works), not by a keyword search on the description - the
language here is too varied and context-dependent ("5+ years" in a "nice to
have" bullet vs. a hard requirement, or "no C2C" being a *good* signal
despite containing "C2C") for a substring match to get right without
rejecting postings it shouldn't.

```bash
job-bot run --keywords "Full Stack Engineer" --location "United States" \
  --experience-level entry,associate,mid-senior --max-years-experience 6 \
  --require-w2 --min-score 75 \
  --exclude-title-keywords "forward deployed,sales engineer" --dry-run
```

Switch providers per run without editing `.env`:
```bash
job-bot run --provider ollama --model deepseek-r1:8b --dry-run
job-bot run --provider claude --model claude-opus-5 --dry-run
```

For every job that passes the fit/eligibility gate (even on a `--dry-run`),
`job-bot run` writes a tailored resume (as both a plain-text reference copy
and a `.docx`) and cover letter to
`<APPLICATIONS_DIR>/<today's date>/<job id - company - title>/` - one dated
folder per day's worth of applications, for you to read, copy from, or reuse
in interview prep. `APPLICATIONS_DIR` defaults to `data/applications` inside
the repo; point it at a folder outside the repo (e.g. on your Desktop) in
`.env` if you want to browse it directly day by day.

That per-job `.docx` is also what actually gets uploaded to the LinkedIn
form, in place of your static `RESUME_PATH` file - but only the professional
summary and skills line are the LLM-generated content; your real job
titles, companies, dates, and every bullet under them are copied verbatim
from your own resume, never reworded, reordered, or replaced by the model's
own rewritten bullets, since a wrong company name or fabricated-sounding
claim in a document actually submitted to a real employer is a much more
serious mistake than an imperfectly-phrased summary sentence. If your
resume's sections can't be confidently located (see `resume_document.py`),
your unmodified `RESUME_PATH` file is uploaded instead, exactly like before
this feature existed. See `job_bot/generation/resume_document.py` and
`artifacts.py`.

Tailoring is written to be ATS-friendly: plain text with a single leading
`-` per bullet (no tables, columns, icons, or special unicode a parser can
choke on), every quantified metric from the original bullet preserved, and
the job posting's own wording echoed where the candidate genuinely has that
skill - see `resume_tailor.py`'s `SYSTEM_PROMPT`. It's also guarded against
a failure mode local models are prone to: fabricating a skill straight from
the job posting's wording that the resume never actually mentions.
`_grounded_skills()` drops any highlighted skill that doesn't trace back to
a real word in your resume, as a code-level backstop the prompt alone can't
guarantee.

There's no practical way to fine-tune Ollama's weights on every run, so
"learning from previous responses" here means something more modest but
genuinely useful: each tailored resume is logged to the tracker DB, and the
next one is generated with up to 3 of your past ones as few-shot style
reference - preferring ones for jobs you've since marked `interviewing` or
`offer` (see "Tracking outcomes" below) over merely recent ones. See
`Tracker.best_resume_examples()` and `tailor_resume()`'s `examples` param.

Every generation - not just the 3 used as prompt reference - can be
reviewed directly, with each job's current status for outcome context:

```bash
job-bot resume-history                   # every tailored-resume generation, most recent first, with outcome status
job-bot resume-history --search python   # only generations whose summary/company/title mentions "python"
job-bot resume-history --company "Acme Corp"  # exact company match (case/spacing-insensitive), not a substring like --search
job-bot resume-history --format json     # same generations as one JSON array instead
```

Unlike the resume, the generated cover letter *is* used directly in the
submission: if the Easy Apply form has a "Cover letter" text field, it's
filled with the generated text (a text field is per-application content
you can always edit or clear before submitting, unlike a formal resume
document) - see `_looks_like_cover_letter_field()` in `linkedin_adapter.py`.

## Applying on company websites (experimental)

Not every LinkedIn posting offers Easy Apply - many show "Apply" instead,
which hands you off to the employer's own career site (Greenhouse, Workday,
Lever, a fully custom form, ...). `--include-external-apply` (or
`ENABLE_EXTERNAL_APPLY=true` in `.env`) makes `job-bot run` also follow
those and attempt a best-effort, heuristic fill there, instead of skipping
them the way it does by default.

**This is genuinely experimental, and "bug-free" isn't a realistic bar for
it.** Easy Apply is one site with a knowable structure this project can
inspect and adjust to (`linkedin_adapter.py`'s `SELECTORS`). A generic
filler for arbitrary employer sites has no equivalent - every one is
structured differently, so it will get real forms wrong. The design goal is
to fail *safely* when that happens: leave a specific, readable reason in
`data/failed_applications.log` rather than guess a field, submit something
incomplete, or hang. See `job_bot/browser/external_apply_adapter.py`'s
module docstring for exactly how (word-boundary matching, never-guess
radio/select handling, the same philosophy as the LinkedIn adapter - just
looser selectors, since there's no single DOM to have learned).

**What it will never do, regardless of what a form asks for or what the LLM
might be willing to answer:**
- Solve or bypass a CAPTCHA (`recaptcha`/`hcaptcha`/Cloudflare Turnstile
  markers are detected and stop the application, not worked around).
- Create an account or enter a password (any `input[type="password"]`
  anywhere on the page stops the application, even a password the user
  would have chosen themselves).
- Fill a field asking for a Social Security number, passport number,
  driver's license number, or a bank/credit card/routing number - see
  `SENSITIVE_FIELD_MARKERS` (`browser/base_adapter.py`, shared with the
  standard Easy Apply path too - employers attach their own custom
  screening questions to Easy Apply, not just to external forms). A
  required field matching this list is treated as unanswerable, the same
  as one the LLM genuinely couldn't answer.
- Check a consent/agreement checkbox or select a radio option on the user's
  behalf at all - both are left untouched; a *required* one left unanswered
  stops the application rather than guessing or silently submitting without it.

**Confirmation before submitting is always required on this path**,
regardless of `--yes-i-understand-the-risk` or
`REQUIRE_CONFIRM_BEFORE_SUBMIT=false` - it's far less tested than the
LinkedIn flow, so every external submission pauses for your `[y/N]`
regardless of how you've configured the rest of a run.

```bash
job-bot run --keywords "Full Stack Engineer" --location "United States" \
  --include-external-apply --dry-run
```

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
job-bot report --stale-days 7           # also flag applications with no reply after 7 days (default: STALE_AFTER_DAYS)
job-bot report --by-score --format json # same data as one JSON object, for a script or cron job
job-bot report --by-eligibility         # counts by eligibility-gate verdict (pass/flag/fail/not scored)
job-bot report --by-company             # counts by company, most-applied first
job-bot report --by-missing-qualifications # which missing qualifications the LLM scorer flags most often across postings
job-bot report --by-missing-qualifications --missing-qualifications-limit 10 # only the 10 most common (most distinct phrases only occur once)
job-bot status <job_id> interviewing    # or: offer, rejected, withdrawn, no_response
job-bot status <job_id>                 # no status - print the job's record, its match reasoning, eligibility verdict, and missing qualifications (if scored), tailored resume (if any), and Q&A history instead
job-bot status <job_id> --format json   # same view as one JSON object, for a script watching one specific application
job-bot status <job_id> --note "Recruiter mentioned $150k base."   # attach a free-text note (independent of status)
job-bot export                          # all tracked jobs as CSV, to stdout
job-bot export --status applied --out applied.csv
job-bot export --format json --out applied.json    # same rows/columns, as JSON
job-bot export --search "Acme"                      # title/company/notes/reasoning match, like the dashboard's search box
job-bot export --company "Acme Corp"                # exact company match (case/spacing-insensitive), not a substring like --search
job-bot export --eligibility fail                   # every job the eligibility gate categorically disqualified
job-bot export --stale-days 14                       # applied jobs with no reply after 14 days, same rule `report --stale-days` uses
```

`<job_id>` is the LinkedIn job id, printed by `job-bot run` and visible in
`data/audit.log`. Valid statuses are listed in
`job_bot.tracker.db.TRACKER_STATUSES`.

Marking a job `interviewing` or `offer` here does more than record the
outcome: it's also the signal `best_resume_examples()` looks for to prefer
that job's tailored resume as a few-shot example for future ones (see
"Running it" above) - so keeping this up to date is what makes the resume
tailoring loop actually improve over time, not just log history.

Answers `job-bot run` is confident in (grounded in your resume/FAQ, above
`FAQ_SAVE_CONFIDENCE` in `.env`) are automatically cached to `FAQ_PATH` for
reuse on future applications - the bot gets faster and more consistent the
more you use it, without ever caching a low-confidence guess.

```bash
job-bot faq list                          # every cached question/answer pair
job-bot faq list --search python          # only pairs whose question or answer mentions "python"
job-bot faq list --format json            # same (optionally --search-filtered) pairs as one JSON object
job-bot faq remove "Years of Python experience?"   # e.g. to fix a wrong one
job-bot faq remove "Question A" "Question B"        # remove several at once, each quoted separately
job-bot faq import faq_backup.json        # merge in a backup, or another install's FAQ_PATH
job-bot faq export --out faq_backup.json  # write the cache out as JSON; omit --out to print to stdout
```

## Learning from questions it couldn't answer

The flip side of the FAQ cache above: a required text/radio/select question
the LLM genuinely can't answer confidently is deliberately left unanswered
rather than guessed (see `linkedin_adapter.py`'s "never guess" reasoning),
which fails that one application - but by itself, that's a dead end, since
the exact same question (eligibility, sponsorship, on-site requirements,
...) tends to be asked near-verbatim across many different postings, and
would otherwise fail the same way every single time.

Every one of these gets logged to `ANSWER_GAPS_PATH`
(`data/answer_gaps.json`), deduplicated by question text with a count of
how often it's come up. Review and answer them with:

```bash
job-bot review-answers
job-bot review-answers --search sponsor  # only review/print gaps whose question mentions "sponsor"
job-bot review-answers --format json   # list gaps as JSON instead of prompting - for a monitoring script
job-bot review-answers --dismiss "Some garbled or duplicate question"  # discard without answering
```

Leaving the interactive prompt blank just skips a gap for that run - it'll
keep resurfacing every time you run this. `--dismiss` (accepts one or more
questions, exact text as shown by this command's own output) permanently
discards a gap without answering it - for noise, duplicates, or a question
you've decided isn't worth caching an FAQ answer for. It doesn't touch
`FAQ_PATH`.

Each answer you give is saved straight to `FAQ_PATH`, so it's reused as
context on every future posting that asks the same question - this is
what closes the loop for "the bot learns from what it couldn't do" without
literally retraining the model: it can't update its own weights, but it
can remember precisely what it failed on, and an answer given once here
teaches every future application, not just the one that failed.

This happens automatically too, without any manual review step: every
question `job-bot run` ever answers - not just the curated subset that
made it into `FAQ_PATH` - is logged to the tracker DB
(`Tracker.recent_qa_pairs()`), and the most recent ~20 unique ones are
included as informal reference context on every future question it's
asked (`qa_answerer.py`). A lower-confidence or since-corrected past
answer is never treated as verified fact the way FAQ is - the model is
told explicitly to use it only for consistency of phrasing/style, still
checking the resume itself before answering - but it means the model's
answers get steadily more consistent with its own history the more you
use it, purely from normal usage, with no separate step required.

Every one of those logged pairs - not just the ~20 most recent used as
context, and not just the curated subset promoted to `FAQ_PATH` - can be
searched directly:

```bash
job-bot qa-history                     # every recorded Q&A pair, most recent first, with company/title context
job-bot qa-history --search python     # only pairs whose question or answer mentions "python"
job-bot qa-history --company "Acme Corp"  # exact company match (case/spacing-insensitive), not a substring like --search
job-bot qa-history --format json       # same pairs as one JSON array instead
```

## Company blacklist

```bash
job-bot blacklist add "Company Name"      # job-bot run will always skip it
job-bot blacklist add "Company A" "Company B" "Company C"   # add several at once
job-bot blacklist add "Company Name" --reason "no H1B sponsorship"  # remembered, shown by `list`
job-bot blacklist remove "Company Name"
job-bot blacklist check "Company Name"    # is it blacklisted, and why (case/spacing-insensitive)
job-bot blacklist list
job-bot blacklist list --search sponsorship   # only companies whose name or reason contains this text
job-bot blacklist list --format json          # same entries ({"name", "reason"}) as one JSON array instead
job-bot blacklist import past_employers.txt   # one company per line; blank/'#'-comment lines skipped
job-bot blacklist export --out backup.txt     # write it back out the same way; omit --out to print to stdout
```

`blacklist add` warns (without blocking the add) if the company still has a
tracked application `applied`, `interviewing`, or `offer` at - blacklisting
only stops future applications, it never touches anything already tracked,
so this is a safety net for a typo or a name confused with a similar one.
`--reason` applies to every company in that `add` call - blacklisting
several companies for the same reason at once is the common case; a
different reason per company just means a separate call each. Reasons
don't round-trip through `import`/`export` (that format stays plain
company names, one per line).

Backed by `BLACKLIST_PATH` (default `data/company_blacklist.json`); matching
is case-insensitive.

## Audit log

Every action `job-bot run`/`gmail-sync` takes - a search, a score, an apply,
a skip and why, a blacklist-driven status update - is logged, redacted of
secrets, to `AUDIT_LOG_PATH` (default `data/audit.log`; `job-bot doctor`'s
"Audit log writable" check confirms it can actually be written to). It's
readable directly:

```bash
job-bot audit-log                        # every logged action, most recent first
job-bot audit-log --search "Acme Corp"   # only entries whose action or details mention "Acme Corp"
job-bot audit-log --action applied       # only entries with this exact action (e.g. applied, skip_blacklisted, scored)
job-bot audit-log --format json          # same entries as one JSON array instead
job-bot audit-log --failures             # reads FAILED_APPLICATIONS_LOG_PATH instead - the postings a run couldn't finish and why
```

`--failures` points every flag above at `data/failed_applications.log`
instead - the same file a `job-bot run` that couldn't finish some
postings tells you to check by hand ("N posting(s) could not be
completed..."), now readable the same way as the main audit log rather
than only by opening the raw JSONL file.

See `job_bot/safety/audit_log.py`'s module docstring for exactly what is
and isn't recorded (metadata only - never full resume text or raw LLM
prompts).

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

- **Click a status pill** (e.g. "Interviewing 3") above the table to filter to
  it instantly - the pill counts update live as statuses change, and only
  scope to whichever statuses currently have a match (plus whatever's
  selected, so you can always click back off it).
- **Search** by title, company, note text, match reasoning, the
  eligibility-gate note, or missing qualifications, and **filter** by
  status (the pills and the status dropdown stay in sync either way) or
  by eligibility verdict (pass/flag/fail).
- **Sort** by newest/oldest, recently applied, fit score, company, or title.
- **Hover a fit score** to see the LLM's own reasoning for it (the same
  `match_reasoning` `job-bot status <job_id>` prints), when the job was
  scored rather than just tracked. A ⚠️ before the score means the
  eligibility gate flagged or categorically disqualified this job -
  hover for the specific posting wording that drove the verdict.
- **Spot stale applications at a glance** - a ⏰ next to the applied date
  means no reply after `STALE_AFTER_DAYS` (default 14), the same
  threshold `job-bot report --stale-days` uses, now visible without
  running a separate command.
- **Update a job's status inline** from the row - no need to drop to
  `job-bot status <job_id> <status>` for a quick correction.
- **Edit a job's note** in a modal - the same note `job-bot status <job_id>
  --note "..."` sets from the command line (the Note button is highlighted
  when a note is already set).
- **Blacklist a job's company** with one click - the same effect
  `job-bot blacklist add "<company>"` has from the command line, including
  the same warning if that company still has a tracked application
  `applied`/`interviewing`/`offer` at (blacklisting only stops future
  applications, it never touches anything already tracked).
- **Manage the blacklist** (view, with each company's reason if one was set
  via `job-bot blacklist add --reason`, and remove entries) from a modal,
  without dropping to `job-bot blacklist list`/`remove`. The one-click
  blacklist button above doesn't prompt for a reason itself (it's meant to
  stay a single click) - set one from the CLI and it shows up here.
- **View a job's Q&A history** (every application-question answer the bot
  gave, and what it was based on) in a modal, without querying the DB by hand.
- **View a job's tailored resume** (the summary/skills/bullets actually
  generated and submitted for it, if any) in a modal - the same view
  `job-bot status <job_id>` already prints, one click away instead.
- **Export CSV or JSON** for whatever's currently filtered/searched, not just
  the visible page - the same output `job-bot export` (`--format csv`/`json`)
  produces on the command line.
- **View the most common missing qualifications** across every tracked
  posting in a modal - the dashboard counterpart to `job-bot report
  --by-missing-qualifications`, showing which specific gaps the LLM scorer
  keeps flagging.
- **Browse the audit log** in a modal, with a live search box and a
  "Failed applications only" checkbox - the dashboard counterpart to
  `job-bot audit-log`/`job-bot audit-log --failures`, capped to the 50
  most recent matching entries (the CLI command has no such cap, for
  anyone who needs the full history).

The dashboard has no login (it's a local tool over your own data), so every
state-changing endpoint (the inline status update, blacklisting a company,
removing one from the blacklist, and setting a note) only accepts
same-origin requests - see `job_bot/dashboard/server.py`'s module
docstring. It follows
your OS/browser's light or dark theme automatically, and the table scrolls
independently of the page on a narrow window so the status/Q&A controls on
the right stay reachable.

## Gmail sync

Recruiters reply by email, not through LinkedIn - `job-bot gmail-sync` reads
your recent Gmail, classifies each message with the LLM (interview invite /
rejection / offer / application confirmation / not job-related), matches it
to a tracked application by company name, and updates its status. It **never**
updates on an ambiguous match (multiple or zero tracked jobs match the
guessed company), a low-confidence classification, or a job already in a
terminal status (`offer`/`rejected`/`withdrawn`/`no_response`) - see
`job_bot/integrations/gmail_sync.py`'s module docstring for the exact rules.
An email the LLM provider fails to classify (a transient error, or a
retry-exhausted response) is skipped, not fatal to the rest of the run -
the sync keeps going and reports how many at the end (see `job-bot
audit-log --action gmail_sync_classify_error` for which ones and why).
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
job-bot gmail-sync --dry-run --format json   # same result as one JSON object, e.g. for a monitoring script
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
- `data/answer_gaps.json` - required questions Easy Apply couldn't
  confidently answer, waiting to be answered via `job-bot review-answers`
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
