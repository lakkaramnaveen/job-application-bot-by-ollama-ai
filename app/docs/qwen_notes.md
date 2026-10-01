# Working with qwen3:30b (Ollama) in this project

This project's default local model is `qwen3:30b` (a 30B-A3B MoE, ~18GB,
run via Ollama - see `app/README.md`'s "Local Ollama vs Claude (cloud)"
section for why it was chosen over smaller models). It's noticeably
stronger at scoring/extraction than `llama3.1:8b`, but it has a handful of
recurring, model-specific failure patterns that cost real debugging time to
track down the first time. This doc consolidates what was learned fixing
each of them, so a future session hitting a similar symptom can jump
straight to the cause instead of re-diagnosing from scratch. Each section
names the fix commit and the file(s) it lives in.

## 1. Reasoning leaks into the real answer field

**Symptom:** a structured-output field (`ApplicationAnswer.answer`,
`CoverLetter.body`, `TailoredResume.summary`/`.bullet_points`/
`.highlighted_skills`, `JobMatchScore.reasoning`/`.eligibility_note`,
`EmailClassification.company_guess`)
contains the model's own
chain-of-thought instead of (or in addition to) the finished content -
phrasing like "I need to answer the question about X...", "Let me
carefully/check/search...", "I should not fabricate...", or a self-review
after an otherwise-complete answer ("Let me check if I've included only
facts from the resume... Final version:") followed by a second, duplicate
copy of the content.

**Why it happens:** qwen3 is a reasoning model. `think:false` (see §4)
skips its *hidden* `<think>` pass, but doesn't stop it from reasoning
out loud *inside* a structured field when the prompt doesn't explicitly
forbid it - the field is syntactically a valid string either way, so
nothing rejects it without an explicit check.

**Scale confirmed live:** 117 of 1,644 recorded `ApplicationAnswer`s in a
real user's `qa_history` were pure leaked reasoning (~7%) - and because
`Tracker.recent_qa_pairs()` reuses past answers as few-shot context for
future questions, each leak was compounding the problem rather than the
"learning loop" closing it. Cover letters/resumes leaked far less often
(1 of 338 on disk) but the one real occurrence contained a duplicated
letter body.

**Fix:** two complementary layers, applied per-field:
- The system prompt (`qa_answerer.py`, `cover_letter.py`,
  `resume_tailor.py`) states explicitly that the field is typed/inserted
  directly into a real form or document - no reasoning process, no
  restating what was checked, no second draft.
- A `field_validator` on `ApplicationAnswer.answer`, `CoverLetter.body`,
  all three `TailoredResume` free-text fields (`summary`, `bullet_points`,
  `highlighted_skills` - the latter two per-list-item, via a shared
  `_reject_leaked_reasoning_in_list` wrapper), `JobMatchScore.reasoning`/
  `.eligibility_note`, and `EmailClassification.company_guess`
  (models/schemas.py) raises on a shared list of this
  model's own consistent reasoning-trace phrasing. The raise routes back
  through `generate_structured()`'s existing retry-on-`ValidationError`
  loop (`ollama_provider.py`), so the model gets another attempt instead of
  the leak silently going through. Each of these three groups only got
  this once it became real, user-facing output - `JobMatchScore.reasoning`/
  `.eligibility_note` via `job-bot status`/`--format json`/the dashboard's
  score tooltip (see `568ccd5`/`ef9d6d3`), `EmailClassification.
  company_guess` via the "gmail_sync_unmatched" entry `gmail_sync.py` logs
  whenever it can't tie a guess to a tracked job, itself readable via
  `job-bot audit-log`/the dashboard's audit view - they were pure
  internal/discarded values before that, not worth guarding since nothing
  ever showed them to anyone. `EmailClassification.role_guess` is still in
  that pre-output state (set but never read, logged, or displayed
  anywhere) and stays unguarded for the same reason, until it isn't.

**If you see this again:** a new leak phrasing wasn't in the marker list.
Add it there (the list is deliberately named/shared across every guarded
field rather than duplicated) rather than writing a new field-specific
check. Commits: `2523b4f`, `f91c080`.

## 2. A finished response gets truncated by pure padding, never closes the JSON

**Symptom:** generation fails with "did not return schema-valid JSON", but
the actual content (e.g. a cover letter body) looks complete and
well-written up to some point - it just never got a closing quote/brace.
Confirmed live: the model reliably (2-3 of every 5 generations, regardless
of prompt length or `num_ctx`/`num_predict`) finishes a coherent letter
and then keeps the JSON string open for hundreds to thousands more
characters of pure `"\n"` (literal backslash-n) padding before generation
is cut off with nothing to close it. This is not a context-window or
output-length-limit issue - it reproduces identically across settings.

**Fix:** two complementary layers (`ollama_provider.py`,
`models/schemas.py`):
- `CoverLetter.body` (and similarly bounded fields) got a `max_length`
  set above every real value observed in practice but low enough to reach
  Ollama's own JSON-schema-constrained decoding, which forces the string
  closed once padding hits that length. Confirmed this alone recovers
  roughly half of prior failures.
- `_repair_truncated_json_string()` covers generations that stall on
  padding *before* reaching that cap: it trims a long run (3+) of
  trailing `\n` escapes off content that's already inside an open string
  within exactly one open top-level object, then closes it - deliberately
  narrow, never fabricates or alters real content, and falls through to a
  normal retry for anything structurally different (cut off mid-word, no
  padding, more than one open brace). Its own quote-parity check must
  count a run of backslashes correctly (an *even* run before a quote
  still leaves that quote unescaped) - a naive "one preceding character"
  regex gets this wrong; see `_count_unescaped_quotes()`.

**Ollama 0.34+ aborts the padding loop itself.** Newer Ollama detects the
repetition and stops generation with `prediction aborted, token repeat limit
reached`. A non-streamed request then gets only that HTTP 500 - none of the
content - so the repair above never ran, and every retry failed identically
(7 of 25 postings in a live run on 2026-09-30). `generate_structured()` now
streams (`_read_chat_stream()`): the finished letter arrives before the final
`{"error": ...}` chunk, and the same repair closes it. Confirmed live: two
postings that failed every time non-streamed now succeed every time.

A plain, unconditional retry (up to `MAX_GENERATION_ATTEMPTS = 3`) also
helps independently for any other truncated/malformed response shape - a
`ConnectError` (Ollama unreachable) or a 404 (model not pulled) are the
only cases that skip retrying, since neither can be fixed by trying again.
Commits: `4e0d853`, `02c9b7b`.

## 3. Formatting a number as prose instead of a bare digit

**Symptom:** a years-of-experience question gets answered "5+ years"
instead of `5`, which then gets typed into a numeric field
(`input[type="number"]`) - the browser silently empties such a field on
an invalid value, so the submitted form ends up with the question
unanswered even though the model "answered" it.

**Fix:** the system prompt now explicitly requires a plain integer for
these questions (`qa_answerer.py`). As a safety net independent of the
model's own compliance, the adapter also strips non-digit characters from
any answer before filling a numeric field. Watch for this pattern
recurring in the FAQ cache too - a malformed answer saved once kept
resurfacing on every future occurrence of the same question until the
prompt fix, since `FAQ_SAVE_CONFIDENCE`-gated caching doesn't itself
validate the *shape* of what it caches. Commit: `ff992d6`.

**Still recurring (2026-10-01), in fields that aren't `type="number"`:**
the safety net above only covered `input[type="number"]`, and LinkedIn
doesn't use that for these questions.
- "How many years of work experience do you have with Java?" is a plain
  text field (with a "0/20" character counter) that LinkedIn validates as
  a whole number on Next/Review. "5+ years" shows "Invalid input" and the
  step never advances; "5" goes through. Fix: `_asks_for_a_number()` (a
  label starting "How many", or asking for "years of ... experience") gets
  the same digit extraction. Commit: `9ace082`.
- The cached answer for "Mobile phone number*" was a whole contact block
  (name, phone, email, city). Typed into the `tel` field, LinkedIn refused
  the contact step. Fix: `_phone_value()` / `_email_value()` pull just the
  number or address for `tel` / `email` fields. Commit: `d364b88`.

Both showed up only as "stuck on a step ... buttons visible: 'Next'" (or
"'Back', 'Review'"), because the bot kept clicking a button whose step
LinkedIn wouldn't leave. The stuck error now also names the fields LinkedIn
rejected and its message ("Fields LinkedIn rejected: ...: Invalid input"),
so the next variant of this is visible in `failed_applications.log`
without a live probe. Commit: `6a9b019`.

## 4. Enable `think:false` for latency, but understand it doesn't stop in-band reasoning

`generate_structured()` passes `think:false` to Ollama for every request.
This skips qwen3's hidden `<think>` pass entirely (~14s -> ~1s in local
testing) and is a silent no-op on non-reasoning models, so it's safe
regardless of which model `OLLAMA_MODEL` is set to. It is *not* a fix for
§1 above - it only removes the separate, unused reasoning trace Ollama
would otherwise emit before the JSON payload; it does nothing to stop the
model from reasoning inline inside a field it was never told not to.
Commit: `022d944`.

## 5. Answer-matching against form options needs both match directions

Not a qwen generation bug, but a direct consequence of qwen's answering
style: `qa_answerer.py`'s prompt allows (and in practice usually
produces) an explanatory free-text answer rather than a bare "yes"/"no",
even for a question whose only form options ARE `["Yes", "No"]`. The
original `_best_match_index()` (`linkedin_adapter.py`,
`external_apply_adapter.py`) only checked one direction - does the whole
answer text appear within an option's label - which can never match here,
since the answer is always longer than either option.

**Fix:** a second fallback tier, only reached if the first direction finds
nothing: does a short option appear as a whole word within the longer
answer (same word-boundary protection as the existing direction).
Confirmed live: the exact same "are you comfortable commuting" question
was logged as an unanswerable gap 27 times despite the model giving a
clear, correct answer to the free-text version of the same question every
time. Commit: `815104e`.

## 6. Echoing a bare form-field label back as the "answer"

**Symptom:** a required select/radio keeps failing as an unanswerable gap
("The LLM's answer didn't clearly match any option") for a field that is
a bare label rather than a question - `Phone country code`, `Year of
From`/`Month of From`/`Year of To` (a work-history date range),
`LinkedIn`. `data/faq_answers.json` holds the label as its own answer
(`"Phone country code": "Phone country code"`).

**Why it happens:** with no question to answer, the model repeats the
label. That was recorded, promoted to FAQ_PATH (it came back with high
confidence), and from then on replayed by cli.py's exact-match FAQ
shortcut on every posting that asked it - the §3 "the cache doesn't
validate the shape of what it caches" pattern again. Confirmed live: 3
echoed FAQ entries and 29 echoed `qa_history` rows.

**Fix:** `is_echoed_question()` (text_utils.py). `answer()` in cli.py
returns `""` for an echoed LLM answer - unanswered, so the adapter's
normal answer-gap flow takes over - and never records or caches it; an
echoed entry already in FAQ_PATH is skipped rather than replayed; and
`Tracker.recent_qa_pairs()` leaves echoes out of the few-shot context.
Echoed entries already on disk are not rewritten - remove them with
`job-bot faq remove` if you want them gone from the file too.

## 7. LinkedIn refusing page loads in bursts (not a model issue)

**Symptom:** many `prep_error`s in a row for different postings, all
"Failed to load https://www.linkedin.com/jobs/view/N/... after N attempts"
(or, before retries existed, `net::ERR_HTTP_RESPONSE_CODE_FAILURE`). `job-bot
report --by-failure` shows them as one of the biggest kinds.

**Why it matters:** they aren't independent - `data/failed_applications.log`
shows bursts of 18, 16, 15 and 11 consecutive refusals within seconds, i.e.
LinkedIn rate-limiting the session. Loading the remaining postings anyway is
the request pattern most likely to get an automated account restricted.

**Fix:** `_goto_with_retry()` raises `NavigationFailed` once its retries are
exhausted; `_run_apply_cycle()` stops the cycle after
`NAVIGATION_FAILURE_STREAK_LIMIT` (3) of them in a row (a successful load
resets the count), and `--loop` then backs off `loop_interval_minutes` even if
the cycle applied to something.

## 8. A generation that never ends, and prompts far bigger than they need to be

**Symptom:** a run sat at "Writing a tailored resume and cover letter..."
for 17m52s (from Ollama's `~/.ollama/logs/server.log`), with the model
server busy the whole time.

**Why:** after responses were streamed (`d192b86`), httpx's timeout
applies per chunk, so a generation that keeps producing tokens never
times out. Separately, every form question answered by the model carried
the entire FAQ cache: ~19,600 prompt tokens, against 2.8-5k for every other
call. That forced a 32k context window, whose KV cache is reserved up front
(~98 KB per token on qwen3:30b, ~3.2 GB).

**Fix:** every request sends `num_predict` (`MAX_OUTPUT_TOKENS`) and is cut
off after `MAX_GENERATION_SECONDS` (`270ea79`); only the relevant cached
answers go in a Q&A prompt, ~3k tokens (`200a4b6`); `num_ctx` is 12,288 and
grows only for an unusually large prompt (`6bb1f29`). Each `job-bot run`
cycle ends with a "Model: N call(s), ... avg prompt ... tokens, ...
tokens generated (N/s)" line (`c021235`) - check it first when a run feels
slow.

## General guidance for a future session

- When a local-model failure looks bizarre or shows up as a repeating
  "unanswerable gap" for a question the model clearly knows the answer
  to, check `data/failed_applications.log`, `data/answer_gaps.json`, and
  `data/audit.log` for the real shape of what was actually produced or
  matched against, before assuming the LLM logic itself is wrong - several
  of the bugs above were caused by adapter-side matching/validation, not
  bad generations.
- Reasoning-leak marker lists (`models/schemas.py`) and truncation-repair
  heuristics (`ollama_provider.py`) are deliberately narrow and grounded
  in confirmed real output, not preemptive. If a new variant of either
  shows up, extend the existing narrow check with the new confirmed
  pattern rather than loosening it into a broad heuristic.
- These fixes target `qwen3:30b`'s specific behavior, confirmed against
  its actual output - they may not generalize to a different
  `OLLAMA_MODEL`. Re-verify against real generations before assuming they
  transfer.
