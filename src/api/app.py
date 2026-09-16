"""FastAPI application exposing the agent graph as an HTTP service.

`POST /tasks` supports two ways of producing a proposal:

- `request.proposed_action` given: a fresh `ScriptedProposeProvider`
  scripted with exactly that one action is used — no real LLM call.
- `request.proposed_action` omitted: the real `AnthropicProposeProvider`
  is used instead, the same real-provider path `src.graph.evaluation`'s
  `run_default_scenarios_against_real_provider()` already exercises. The
  `ANTHROPIC_API_KEY` environment variable is checked BEFORE attempting
  anything, the same way that function checks it — if it's not set, this
  returns a clean `400` instead of crashing partway through a real call.

No authentication, rate limiting, or other production hardening is
implemented here, matching this project's existing scope discipline.

`_tracer` is a single module-level `Tracer` shared across every request
for the lifetime of the process — the same in-memory-only pattern already
used elsewhere in this codebase (e.g. `ExecutionWrapper.idempotency_store`).
It is not persisted and does not survive a process restart.

`_get_anthropic_client()` is a FastAPI dependency that returns `None` by
default, letting `AnthropicProposeProvider` lazily construct a real
`anthropic.Anthropic()` client on first use — this module never imports
`anthropic` itself, at module level or otherwise. Tests override this
dependency via `app.dependency_overrides` to inject a duck-typed fake
client (same technique as `tests/test_graph_evaluation_real_provider.py`),
so no test ever makes a real network call.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from fastapi import Depends, FastAPI, HTTPException

from src.api.models import HealthResponse, RunEventsResponse, TaskRequest, TaskResponse
from src.graph.build import build_agent_graph
from src.graph.nodes import ProposeFn
from src.graph.providers.anthropic_provider import AnthropicProposeProvider
from src.graph.providers.mock import ScriptedProposeProvider
from src.graph.state import AgentState
from src.observability.tracer import Tracer

app = FastAPI(title="Agentic AI Reliability Tool Execution Engine")

_tracer = Tracer()


def _get_anthropic_client() -> Any | None:
    """FastAPI dependency supplying the `client` used to construct `AnthropicProposeProvider`.

    Returns `None` in production, so the provider lazily builds a real
    `anthropic.Anthropic()` client (reading `ANTHROPIC_API_KEY`) only when
    actually called. Tests override this via `app.dependency_overrides` to
    inject a fake client instead.
    """
    return None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report service liveness. No dependencies checked."""
    return HealthResponse()


@app.post("/tasks", response_model=TaskResponse)
def create_task(request: TaskRequest, anthropic_client: Any = Depends(_get_anthropic_client)) -> TaskResponse:
    """Run one agent task end-to-end and return its final state.

    Builds an `AgentState` from the request, runs it through
    `build_agent_graph()` with either a `ScriptedProposeProvider` (when
    `request.proposed_action` is given) or an `AnthropicProposeProvider`
    (when it's omitted), and records the run's events on the shared
    `_tracer` under a freshly generated `run_id` so they can later be
    retrieved via `GET /runs/{run_id}`.

    Raises `400` — before building any state or touching the graph — if
    no proposed action was given and no real provider is usable: either no
    client was injected and `ANTHROPIC_API_KEY` isn't set, matching the
    clean-skip check `run_default_scenarios_against_real_provider()`
    already does before attempting a real-provider run.
    """
    propose_fn: ProposeFn
    if request.proposed_action is not None:
        propose_fn = ScriptedProposeProvider([request.proposed_action])
    else:
        if anthropic_client is None and not os.environ.get("ANTHROPIC_API_KEY"):
            raise HTTPException(
                status_code=400,
                detail=(
                    "proposed_action was omitted and ANTHROPIC_API_KEY is not set in the environment; "
                    "cannot call the real Anthropic API."
                ),
            )
        propose_fn = AnthropicProposeProvider(client=anthropic_client)

    run_id = str(uuid.uuid4())
    graph = build_agent_graph(propose_fn, tracer=_tracer)

    initial_state = AgentState(run_id=run_id, ticket=request.ticket, task_description=request.task_description)
    raw_result = graph.invoke(initial_state)
    final_state = AgentState.model_validate(raw_result)

    return TaskResponse(
        run_id=run_id,
        run_status=final_state.run_status,
        authorization_result=final_state.authorization_result,
        execution_result=final_state.execution_result,
    )


@app.get("/runs/{run_id}", response_model=RunEventsResponse)
def get_run(run_id: str) -> RunEventsResponse:
    """Return the traced event sequence for `run_id`, or 404 if unknown.

    `Tracer.events_for_run()` returns an empty list for both an unknown
    run and a known-but-eventless one; every run driven through
    `POST /tasks` always produces at least a PROPOSED event, so an empty
    result here is treated as "no such run" for this HTTP-facing lookup.
    """
    events = _tracer.events_for_run(run_id)
    if not events:
        raise HTTPException(status_code=404, detail=f"No run found with run_id {run_id!r}")
    return RunEventsResponse(run_id=run_id, events=events)
