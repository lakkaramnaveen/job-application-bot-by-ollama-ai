# Architecture: the application pipeline

Status: accepted, in progress (2026-10-02). Owner: whoever runs the dev loop.

## Context

`job-bot run` is one pipeline: search → per posting (skip checks → load the
description → score → tailor resume and cover letter → fill and submit the
Easy Apply form) → record the outcome. Almost all of it lives in one function:

| Hotspot (2026-10-02)              | Size                                   |
|-----------------------------------|----------------------------------------|
| `cli.py`                          | 3,304 lines, ~29% of `job_bot/`        |
| `cli._run_apply_cycle()`          | 523 lines, 20 parameters, 6 closures   |
| `cli.cmd_run()`                   | 243 lines (setup + the `--loop` driver) |
| `browser/linkedin_adapter.py`     | 1,420 lines (search + form filling)    |

Every production bug fixed during the week of 2026-09-28 touched
`_run_apply_cycle()` or its closures. The structure has concrete costs:

- **Duplicated policy.** "Is this failure fatal?" was decided in three
  copy-pasted chains (search, prep, apply). Adding the dead-browser check
  meant three edits, and a duplicated block shipped once before review caught it.
- **Untestable seams.** The closures (`should_skip`, `is_dead_end`, `answer`,
  `apply_to`, ...) can only be exercised by running a whole `cmd_run()` with
  fakes. That's why `tests/test_cli_run.py` is ~2,800 lines of end-to-end setup.
- **Hidden coupling.** The closures share mutable state through the enclosing
  scope (`requested_ids`, counters, `fatal_error`), so a change to one
  silently affects the others.

## Decision

Grow a `job_bot/pipeline/` package and move policy out of `cli.py` one
cohesive, behavior-preserving piece at a time. `cli.py` stays the composition
root: argument parsing, wiring, printing, and the `--loop` driver.

Target shape:

```
job_bot/pipeline/
  failures.py   classify_failure(e, page) -> FailureVerdict      [done]
  skip.py       SkipPolicy: reason() (audited skip) and
                is_dead_end() (search's silent filter), one set
                of rules                                          [done]
  answers.py    AnswerService: FAQ lookup, model call, caching shape
                (cacheable_answer), learning from FieldsRejected  [done]
  context.py    RunContext: the shared dependencies (tracker,   [next]
                rate_limiter, blacklist, audit, failure_log,
                answer_gaps, settings, provider, resume_store) as one
                frozen dataclass, replacing 20-parameter signatures
  cycle.py      run_cycle(ctx, adapter, ...) -> CycleResult, the loop
                body now in _run_apply_cycle()
```

### Principles

1. **Policy is pure, effects are injected.** Classifiers and policies take
   plain values and return verdicts (`FailureVerdict`, skip reasons). Printing,
   audit logging and sleeping stay with the caller, so policies are
   unit-testable without browsers, models or files.
2. **One decision, one place.** A rule that can end a run, skip a posting or
   change what gets cached lives in exactly one function.
3. **Behavior-preserving steps.** Each extraction ships alone. The full suite
   (1,250+ tests) must pass unchanged, with no edited assertions, before the
   next step. New seams get direct unit tests on top.
4. **Keep the test seams stable.** Tests patch `job_bot.cli.get_provider`,
   `browser_session`, `LinkedInAdapter`, `time.sleep` and a few constants.
   Extracted code receives these as arguments rather than importing them, so
   existing patches keep working.
5. **Data-driven priorities.** The order of the remaining steps follows where
   live failures come from (`data/failed_applications.log`, `job-bot report
   --by-failure`), not size alone.

## Consequences

- `cli.py` shrinks toward wiring and presentation; policies get small,
  focused test files (`tests/test_pipeline_*.py`).
- One more package to learn. Each module's docstring says why it exists
  and which live failure motivated it.
- Until `RunContext` lands, extracted functions take explicit parameters. That's
  acceptable for the first steps; `RunContext` removes it.

## Not doing (yet)

- **Rewriting `linkedin_adapter.py`'s form filling into a parse → decide →
  apply model** (snapshot the step's fields as data, answer in Python, then
  act). It's the right long-term shape against LinkedIn's frequent markup
  changes, but it's a larger change that's only worth making once the pipeline
  seams above exist to test it through. Revisit after `cycle.py`.
- **Splitting `dashboard/render.py` and `server.py`.** They're large but stable,
  and none of this week's failures came from them.
