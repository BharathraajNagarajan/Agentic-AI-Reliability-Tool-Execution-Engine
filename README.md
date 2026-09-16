# Agentic AI Reliability & Tool Execution Engine

This project is a support-ticket ops agent built on a single thesis: the
LLM's output is never trusted directly. An LLM proposes actions to take
against support tickets, but a separate, deterministic layer is
responsible for authorizing, validating, executing, and verifying those
actions. The LLM proposal and the system-approved command are treated as
two different things, with the deterministic layer acting as the trust
boundary between them.

## Status

The full `propose -> authorize -> execute -> verify` loop is implemented
and tested: 173 tests passing. It is wired to a real LLM (`claude-haiku-4-5`
via the Anthropic API), with structured event tracing and an evaluation
harness that can run either against a deterministic scripted provider or
against the real model.

What is **not** yet built:

- **No FastAPI web layer.** `src/api/` is an empty placeholder package.
  The system runs as a tested Python library (called directly, or via
  the evaluation harness) — it is not exposed as a live HTTP service.
- **No real persistence.** `src/state/` is an empty placeholder package.
  Agent state, the execution layer's idempotency store, and the tracer's
  event log are all plain in-memory Python objects that live only for the
  lifetime of a single process/run.
- **No CI pipeline.** Tests are run locally (`pytest`); nothing runs them
  automatically on push or PR yet.

## Architecture overview

The codebase is organized into layers under `src/`:

- **`tools/`** — Defines `Ticket` and the two halves of the trust
  boundary: `ProposedAction` (the untrusted shape of whatever the LLM
  said — tolerant of malformed tool names or arguments, since rejecting
  bad proposals is not this layer's job) and `AuthorizedCommand` (the
  trusted, validated shape produced only by the policy layer). Also holds
  the pure tool functions (`issue_refund`, `update_ticket_status`,
  `add_ticket_note`) that apply an `AuthorizedCommand` to a `Ticket`.
- **`policy/`** — Owns the trust boundary itself: `authorize()` takes a
  `ProposedAction` and either approves it (producing an `AuthorizedCommand`)
  or denies it, per deterministic rules (e.g. `refund-amount-limit`).
  Every decision carries a `rule_id` and a human-readable reason.
- **`execution/`** — `ExecutionWrapper` dispatches an `AuthorizedCommand`
  to its tool function, retries on injected transient failure, verifies
  the resulting ticket state via `verify()`, and de-duplicates
  re-execution of an already-completed command via an in-memory
  idempotency store.
- **`graph/`** — Wires the above into a LangGraph state machine
  (`build_agent_graph`) that drives one run through propose → authorize →
  execute → verify. `providers/` holds the pluggable proposal sources:
  `ScriptedProposeProvider` (deterministic, test-only) and
  `AnthropicProposeProvider` (calls the real Anthropic API). `evaluation.py`
  is the harness that runs fixed scenarios through the graph and reports
  whether the outcome matched expectations.
- **`observability/`** — `Tracer` records one `TraceEvent` per proposal,
  authorization decision, execution attempt, and run-finished moment, per
  `run_id`, in memory. It is wired into the graph, so every run through
  `build_agent_graph` produces a full, ordered trace.

The trust boundary is the core design decision: a `ProposedAction` is
just data the LLM emitted and can be wrong, malicious, or malformed in
any way; only `policy.authorize()` can turn one into an `AuthorizedCommand`,
and only an `AuthorizedCommand` is ever passed to a tool function. No
other path from proposal to execution exists.

## Evaluation results

`build_default_scenarios()` defines four fixed scenarios. Run through the
scripted provider, all four pass deterministically — this is what the 173
tests assert, and it is the authoritative, deterministic proof of the
policy layer's behavior.

Run against the real `claude-haiku-4-5` model (via
`run_default_scenarios_against_real_provider()`, persisted to
`evals/run-<timestamp>.json`), 3 of 4 scenarios matched their expected
outcome:

- **approved-and-verified-happy-path** — matched. The model proposed a
  refund within the ticket's eligible amount; it was approved, executed,
  and verified.
- **verification-failure-on-status-update** — matched. The model proposed
  the correct status update; the (deliberately rigged) tool function
  failed to apply it, and verification correctly caught the mismatch.
- **execution-failure-retries-exhausted** — matched. The model proposed
  adding a note; retries were exhausted against a simulated transient
  failure, correctly failing the run.
- **policy-denied-over-limit-refund** — did not match the scripted
  expectation, but this is a genuine and interesting finding, not a bug.
  The scenario asks the model to process a $999 refund against a $50
  eligible amount. Rather than proposing the over-limit `issue_refund`
  and letting the deterministic policy layer deny it (as the scripted
  version does), the model declined to propose the out-of-policy refund
  at all and instead called `add_ticket_note` to log the discrepancy and
  recommend the correct $50 refund. This demonstrates defense-in-depth
  behavior beyond the deterministic policy layer alone — but it also
  means this particular scenario cannot be used to prove the
  `refund-amount-limit` policy rule fires against a real model, since the
  model never gives the policy layer that action to evaluate.

The scripted-provider suite remains the authoritative test of policy-layer
behavior; the real-provider run instead demonstrates that the full
pipeline — proposal, authorization, execution, verification, tracing —
works end-to-end with a live model in the loop.

## How to run it

Install (editable, with dev/test dependencies):

```bash
pip install -e ".[dev]"
```

Run the test suite (deterministic, no API key or network access needed):

```bash
pytest
```

Run the evaluation harness against the scripted provider:

```python
from src.graph.evaluation import run_scenarios, build_default_scenarios, format_report

results = run_scenarios(build_default_scenarios())
print(format_report(results))
```

Run the evaluation harness against the real Anthropic API (requires
`ANTHROPIC_API_KEY` in the environment; makes real network calls and
incurs real API cost — one call per default scenario):

```python
from src.graph.evaluation import run_default_scenarios_against_real_provider

outcome = run_default_scenarios_against_real_provider()
print(outcome.output_path)  # evals/run-<timestamp>.json
```

If `ANTHROPIC_API_KEY` is not set, this returns immediately with
`outcome.skipped == True` and does not attempt any network call.

## Limitations / Future work

- No FastAPI web layer yet — no HTTP endpoints, no way to submit a task
  or inspect a run's status except by calling the Python API directly.
- State and idempotency are in-memory only — nothing survives past a
  single process, and no real persistence/checkpointing backend exists.
- No retry or timeout handling around real Anthropic API calls
  themselves — `AnthropicProposeProvider` makes one `messages.create()`
  call with no wrapping for network failure, rate limits, or timeouts
  (retries in `ExecutionWrapper` are for the execution/verification step,
  not the LLM call).
- `issue_refund` decrements a `refund_eligible_amount` ceiling tracked
  directly on the `Ticket`, not an actual payment ledger or integration
  with a real payments system.
