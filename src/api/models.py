"""Pydantic request/response models for the FastAPI layer.

Distinct HTTP-facing models exist only where the internal shape isn't
already the right request/response shape: `TaskRequest` bundles a
`Ticket`, a `task_description`, and a `ProposedAction` together (the
latter standing in for a real LLM call, per `src.api.app`'s docstring),
and `TaskResponse`/`RunEventsResponse`/`HealthResponse` shape what
`POST /tasks`/`GET /runs/{run_id}`/`GET /health` return.
`Ticket`/`ProposedAction`/`AuthorizationResult`/`ExecutionResult`/
`TraceEvent` are reused directly as fields — each is already a thin,
well-typed domain model, so wrapping them again would just be
duplication.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.execution.schemas import ExecutionResult
from src.graph.state import RunStatus
from src.observability.tracer import TraceEvent
from src.policy.schemas import AuthorizationResult
from src.tools.schemas import ProposedAction, Ticket


class TaskRequest(BaseModel):
    """Request body for `POST /tasks`.

    `proposed_action`, when given, stands in for a real LLM call: the API
    feeds it to a `ScriptedProposeProvider` as this run's single scripted
    proposal. When omitted, the API instead calls the real Anthropic API
    via `AnthropicProposeProvider` (see `src.api.app`), failing cleanly
    with a 4xx if `ANTHROPIC_API_KEY` isn't configured.
    """

    ticket: Ticket = Field(..., description="The support ticket this run works against.")
    task_description: str = Field(..., description="The instruction/request that initiates this run.")
    proposed_action: ProposedAction | None = Field(
        default=None,
        description=(
            "The action to inject as this run's single scripted proposal, standing in for a real LLM call. "
            "When omitted, a real Anthropic API call is made instead."
        ),
    )


class TaskResponse(BaseModel):
    """Response body for `POST /tasks`: the run's final state plus its identifier."""

    run_id: str = Field(..., description="Unique identifier for this run, usable with GET /runs/{run_id}.")
    run_status: RunStatus = Field(..., description="The run's final coarse-grained status.")
    authorization_result: AuthorizationResult | None = Field(
        default=None,
        description="Outcome of the policy check on the proposed action, or None if authorization never ran.",
    )
    execution_result: ExecutionResult | None = Field(
        default=None,
        description="Outcome of execution, or None if the run never reached execution (e.g. denied).",
    )


class RunEventsResponse(BaseModel):
    """Response body for `GET /runs/{run_id}`: the traced event sequence for that run."""

    run_id: str = Field(..., description="Identifier of the run these events belong to.")
    events: list[TraceEvent] = Field(..., description="The run's traced events, in recorded order.")


class HealthResponse(BaseModel):
    """Response body for `GET /health`."""

    status: str = Field(default="ok", description="Static liveness indicator.")
