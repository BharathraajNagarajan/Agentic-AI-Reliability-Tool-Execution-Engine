# Agentic AI Reliability & Tool Execution Engine

**Stack:** Python · FastAPI · LangGraph · MCP · Anthropic API (Claude) · Pydantic · pytest

This project is a support-ticket ops agent built on a single thesis: the
LLM's output is never trusted directly. An LLM proposes actions to take
against support tickets, but a separate, deterministic layer is
responsible for authorizing, validating, executing, and verifying those
actions. The LLM proposal and the system-approved command are treated as
two different things, with the deterministic layer acting as the trust
boundary between them.

## Status

The full `propose -> authorize -> execute -> verify` loop is implemented
and tested: 191 tests passing. It is wired to a real LLM
(`claude-haiku-4-5` via the Anthropic API), with structured event tracing
and an evaluation harness that can run either against a deterministic
scripted provider or against the real model.

A FastAPI layer (`POST /tasks`, `GET /runs/{run_id}`, `GET /health`)
exposes the agent as a real HTTP service. `POST /tasks` supports two
modes: a scripted-proposal mode (the caller supplies the `ProposedAction`
directly, no LLM call) and a real-LLM mode (omit it, and the request
instead calls the real Anthropic API via `AnthropicProposeProvider`,
failing cleanly with `400` — not a crash — if `ANTHROPIC_API_KEY` isn't
set).

A standalone MCP server exposes the three tool functions (`issue_refund`,
`update_ticket_status`, `add_ticket_note`) as MCP tools for external MCP
clients to call directly, backed by its own in-memory ticket store. See
["MCP: what's built, and a deliberate scoping
decision"](#mcp-whats-built-and-a-deliberate-scoping-decision) below for
what this does and does not cover.

What remains true as a limitation:

- **No real persistence.** Agent state, the execution layer's idempotency
  store, the tracer's event log, and the MCP server's ticket store are
  all plain in-memory Python objects that live only for the lifetime of
  a single process/run.
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
- **`api/`** — A FastAPI application (`src.api.app`) exposing the agent
  over HTTP: `POST /tasks` builds an `AgentState` and runs it through
  `build_agent_graph()`, `GET /runs/{run_id}` returns the traced event
  sequence for a run from a shared in-memory `Tracer`, and `GET /health`
  reports liveness.
- **`mcp/`** — A standalone MCP server (`src.mcp.server`) exposing
  `issue_refund`/`update_ticket_status`/`add_ticket_note` as MCP tools,
  with input schemas derived from the same `*Args` pydantic models via
  `model_json_schema()`, and its own in-memory `TicketStore`
  (`src.mcp.store`) seeded with sample tickets. See below for what this
  server does and does not do.

The trust boundary is the core design decision: a `ProposedAction` is
just data the LLM emitted and can be wrong, malicious, or malformed in
any way; only `policy.authorize()` can turn one into an `AuthorizedCommand`,
and only an `AuthorizedCommand` is ever passed to a tool function. No
other path from proposal to execution exists.

### MCP: what's built, and a deliberate scoping decision

A real, tested MCP server exists (`src/mcp/server.py`, `src/mcp/store.py`,
`tests/test_mcp_server.py`) exposing `issue_refund`, `update_ticket_status`,
and `add_ticket_note` as MCP tools — with input schemas derived from the
same `UpdateTicketStatusArgs`/`IssueRefundArgs`/`AddTicketNoteArgs`
pydantic models via `model_json_schema()` — for any external MCP client
(a different agent, Claude Desktop, etc.) to call directly. It does not
enforce `policy.authorize()` or any other business rule; it trusts that
whatever called it has already decided the action is authorized. That
caller-side authorization responsibility is documented explicitly in the
module's own docstring.

The agent's own internal execution path — `graph` → `authorize()` →
`ExecutionWrapper` → `src.tools.functions` — deliberately does **not**
route through this MCP server. This is a considered decision, not an
unfinished feature: doing so would introduce a second, divergent ticket
store (the MCP server's own `TicketStore`, seeded independently of
whatever `Ticket` the agent is carrying in `AgentState`), would weaken
`verify()`'s correctness guarantee from "structurally true, because it's
comparing the same Python object before and after" to "true only if the
server happens to echo full post-state back in its tool response," and
would reopen a duplicate-effect risk on retry-after-timeout that
`ExecutionWrapper`'s current in-process idempotency check closes for
free. None of that buys anything: the agent and the MCP server would be
the same trust domain, same process family, same deploy — there is no
real distributed-trust or scaling boundary between them to justify
crossing it. The MCP server's actual purpose here is external tool
exposure, not an internal execution transport.

## Evaluation results

`build_default_scenarios()` defines four fixed scenarios. Run through the
scripted provider, all four pass deterministically — this is part of what
the 191 tests assert, and it is the authoritative, deterministic proof of
the policy layer's behavior.

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

Start the FastAPI service:

```bash
uvicorn src.api.app:app --reload
```

Then `POST http://127.0.0.1:8000/tasks`, `GET
http://127.0.0.1:8000/runs/{run_id}`, and `GET
http://127.0.0.1:8000/health` are live; interactive docs are at
`http://127.0.0.1:8000/docs`. By default `POST /tasks` still needs either
a `proposed_action` in the request body or `ANTHROPIC_API_KEY` set in the
environment for the real-LLM mode.

Run/exercise the MCP server: there is no standalone CLI script beyond the
server module itself yet, so `tests/test_mcp_server.py` is the reference
for how to drive it — using the MCP SDK's own in-memory client/server
session helper
(`mcp.shared.memory.create_connected_server_and_client_session`) to call
`list_tools()`/`call_tool()` against `src.mcp.server.build_mcp_server()`
directly, with no process or transport involved. To run it as a real
standalone process instead, speaking MCP over stdio to whatever MCP
client launches it:

```bash
python -m src.mcp.server
```

## Limitations / Future work

- No real persistence backend — state and idempotency (agent state, the
  execution layer's idempotency store, the tracer's event log, the MCP
  server's ticket store) are in-memory only; nothing survives past a
  single process.
- No MCP-based idempotency — see ["MCP: what's built, and a deliberate
  scoping
  decision"](#mcp-whats-built-and-a-deliberate-scoping-decision): the
  agent's internal path doesn't route through MCP specifically because
  the MCP server has no idempotency-key awareness of its own, so a
  retry-after-timeout could double-apply an effect if it ever did.
- No CI pipeline — tests are run locally (`pytest`); nothing runs them
  automatically on push or PR yet.
- No retry or timeout handling around real Anthropic API calls
  themselves — `AnthropicProposeProvider` makes one `messages.create()`
  call with no wrapping for network failure, rate limits, or timeouts
  (retries in `ExecutionWrapper` are for the execution/verification step,
  not the LLM call).
- `issue_refund` decrements a `refund_eligible_amount` ceiling tracked
  directly on the `Ticket`, not an actual payment ledger or integration
  with a real payments system.
