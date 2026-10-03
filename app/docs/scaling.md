# Scaling: from one user's laptop to ~1,000 users

Status: proposed (2026-10-02). Builds on `docs/architecture.md`, whose
`job_bot/pipeline/` modules are the seams this design splits along.

## The constraint that decides the architecture

The bot drives a real LinkedIn session in a real browser. On 2026-10-02 one
user's session was rate-limited after about four hours of activity: LinkedIn
refused every page load, search included, for 2+ hours. Running 1,000 users'
sessions from a few datacenter IPs would get them detected and restricted
quickly. It would also mean holding 1,000 people's LinkedIn credentials or
session cookies centrally, which is the most sensitive data this system could
touch.

**So the browser stays on each user's machine.** What centralizes is
everything that doesn't need the user's session: the model calls, resume
tailoring, the tracker, analytics, and configuration. That split also matches
the cost profile below.

## Capacity, from measured data

Measured on 2026-10-02 (`data/audit.log`, `qa_history`), per **successful**
application:

| Work                                     | Per application | Notes                                  |
|------------------------------------------|-----------------|----------------------------------------|
| Model calls (score, tailor, cover, Q&A)  | ~13             | 88 scorings, 60 material calls, 95 Q&A answers / 15 applied |
| Model time (qwen3:30b, M4 Pro)           | ~4.4 s / call   | ~57 s per application, serial          |
| Browser time filling the form            | ~9 s median     | plus page loads                        |
| Prompt size                              | ~2.5k tokens    | after FAQ retrieval (200a4b6)          |

At **1,000 users × 30 applications/day = 30k applications/day**:

- **Model calls:** ~400k/day, about 4.6/s on average and ~15/s at a 3× peak.
  Tokens: ~1.1B/day.
- **Browser time:** spread across 1,000 user machines, so it isn't a central
  cost. That's another reason it stays local.

Conclusions:

1. **The model is the bottleneck and the cost center, not the browser.** One
   laptop-class local model serves roughly 0.2 calls/s, so the central tier
   needs a batched GPU server (vLLM or similar: a handful of GPUs at peak) or a
   hosted API. Either way it sits behind one gateway.
2. **Model work spent on postings that never get submitted is the biggest
   cost lever.** On 2026-10-02 there were 88 scorings for 15 applications.
   That isn't mostly weak matches: of 715 postings ever scored, only 13% fell
   below the bar, and no title word predicts a low score (checked; a title
   pre-filter would save little and risk skipping good fits). The waste was
   strong fits that then failed to submit: form bugs since fixed, the
   one-per-company limit, throttling. Raising the submit rate of good fits
   (`job-bot report --by-failure`) saves more model work than any filter or
   infrastructure choice.

## Target architecture

```
 user's machine                                   central (stateless + Postgres)
 ┌──────────────────────────┐   HTTPS + token   ┌──────────────────────────────┐
 │ Agent (this repo's CLI)  │ ────────────────▶ │ API (FastAPI)                │
 │  - Playwright + session  │                   │  /score /tailor /answer      │
 │  - pipeline/cycle.py     │ ◀──────────────── │  /applications /config       │
 │  - local queue + retries │                   ├──────────────────────────────┤
 └──────────────────────────┘                   │ LLM gateway                  │
                                                │  rate limits, per-user quota,│
                                                │  batching, cache, fallback   │
                                                ├──────────────────────────────┤
                                                │ Postgres: tracker, FAQ,      │
                                                │ answer gaps, audit (per user)│
                                                ├──────────────────────────────┤
                                                │ Queue (Redis/SQS) + workers  │
                                                │  tailoring, docx rendering,  │
                                                │  gmail sync, analytics       │
                                                └──────────────────────────────┘
```

### Microservices: not yet

Start with a **modular monolith**: one codebase and one Postgres schema, deployed
as three process types that scale independently:

- **`api`:** request/response endpoints.
- **`worker`:** queue consumers.
- **`llm-gateway`:** the only process that talks to models.

Split a module into its own service only when its scaling or failure profile
actually diverges. The LLM gateway is the first candidate: GPU hardware, its
own quotas, and the largest failure blast radius. At 1,000 users,
microservices from day one would add network failure modes, distributed
transactions and operational load without solving a measured problem. The
module boundaries already exist (`job_bot/pipeline/`), so splitting later
moves code; it doesn't redesign it.

Mapping today's modules:

| Today (`job_bot/`)                 | Where it runs                         |
|------------------------------------|---------------------------------------|
| `pipeline/cycle.py`, `browser/`    | agent (user's machine)                |
| `pipeline/skip.py`, `failures.py`  | agent (pure policy, also importable by the API) |
| `pipeline/answers.py`, `generation/`, `matching/` | API + LLM gateway       |
| `tracker/`, `safety/answer_gaps.py`, `resume/store.py` | API → Postgres     |
| `llm/`                             | LLM gateway                           |

## Error handling as a system

Today, errors are handled where they're caught. At scale, every failure needs
a **class** that determines retry, back-off and alerting the same way
everywhere. Five classes, already latent in `pipeline/failures.py`:

| Class             | Meaning                           | Response                                   | Examples |
|-------------------|-----------------------------------|--------------------------------------------|----------|
| `TRANSIENT`       | likely to succeed if retried soon | retry with jitter, bounded                 | network blip, model timeout, bad JSON from the model |
| `THROTTLED`       | the other side is rate-limiting   | exponential back-off; circuit-break the dependency | LinkedIn refusing loads (235beef), model 429 |
| `POSTING`         | this item can't be done           | record, move on; no retry storm            | stuck form, rejected field, posting closed |
| `USER_ACTION`     | only the user can fix it          | stop, notify, wait                         | signed out of LinkedIn, an unanswered question, a bad API key |
| `FATAL`           | the run can't continue            | stop the run, alert                        | browser gone, model server unreachable |

Rules that make this production-safe:

- **Idempotent submission.** An application is keyed by `(user, job_id)`. The
  agent records "submitting" before the final click and "submitted" after, so
  a crash between the two is detectable and **never re-submitted blindly**.
  Double-applying is the one failure a user can't undo.
- **Circuit breakers per dependency** (LinkedIn per user, the model gateway
  globally). Once open, callers fail fast as `THROTTLED` instead of queuing
  work that will fail.
- **Bounded everything.** Every external call has a deadline (Ollama: `270ea79`;
  forms: `c20c7ac`), every retry has a cap, every queue has a dead-letter
  destination.
- **Errors are data.** Each failure is logged with its class, so dashboards and
  alerts aggregate by class rather than by error string. That's the first
  implementation step below.

## Observability

- **Structured logs:** JSON, with `user_id`, `run_id`, `job_id` and
  `failure_class` on every line.
- **Metrics:**
  - applications/hour;
  - success rate per form step;
  - failures by class;
  - model latency p50/p95 and tokens per application (today's per-cycle
    "Model:" line, `c021235`, as a time series);
  - LinkedIn throttle events per user.
- **Alerts:** a `FATAL` or `USER_ACTION` spike; the throttle rate across users,
  which is the early warning that LinkedIn changed something for everyone;
  model p95 above its deadline.

## Rollout, smallest useful steps first

1. ~~Classify every failure (`FailureClass` on `FailureVerdict`) and log the
   class with each failure.~~ Done (`88b24e6`).
2. ~~Idempotent submission markers in the tracker (`submitting` → `applied`).~~
   Done (`1e3b1ff`).
3. ~~A circuit breaker around the model provider (a `TRANSIENT` streak
   becomes `THROTTLED` and fails fast).~~ Done (`llm/circuit_breaker.py`).
4. ~~The LLM gateway interface: the agent calls the model through one client
   that can be pointed at local Ollama or a remote gateway (config only).~~
   Done: `OLLAMA_BASE_URL` plus `OLLAMA_API_KEY` (bearer token, with a warning
   if it would go over plain http to a remote host).
5. **Postgres-backed tracker behind the existing `Tracker` interface;** the API
   and auth come after that.

Steps 1–4 make today's single-user tool more robust on their own; none of
them is only for the future multi-user system.
