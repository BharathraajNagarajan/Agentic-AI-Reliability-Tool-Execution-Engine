"""In-memory structured event recording for agent runs.

`Tracer` records an ordered sequence of `TraceEvent`s per `run_id`,
covering the four moments this layer is responsible for observing: each
proposal, each authorization decision, each execution attempt, and the
run's final status. It is a plain in-memory dict, in the same spirit as
`ExecutionWrapper`'s idempotency store — no database, no file I/O, gone
once the `Tracer` instance is garbage collected. A future phase can swap
this for a real store without changing the recording methods' signatures.

`TraceEvent` uses one flat schema with an `event_type` discriminator and
several optional payload fields (only one populated per event) rather
than a discriminated union of distinct event classes: unlike
`AuthorizedCommand` (a genuine trust boundary where mixing up shapes would
be a bug), a trace log is read generically — filtered, exported, queried
by field — and a single wide schema is the more natural fit for that.

This module is NOT wired into `src.graph.nodes`/`src.graph.build` yet;
that integration is a follow-up decision.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from src.execution.schemas import ExecutionResult
from src.graph.state import RunStatus
from src.policy.schemas import AuthorizationResult
from src.tools.schemas import ProposedAction


class EventType(str, Enum):
    """Which kind of moment in an agent run a `TraceEvent` records."""

    PROPOSED = "proposed"
    AUTHORIZED = "authorized"
    EXECUTED = "executed"
    RUN_FINISHED = "run_finished"


class TraceEvent(BaseModel):
    """One recorded moment in an agent run's trace.

    Exactly one of `proposed_action`/`authorization_result`/
    `execution_result`/`run_status` is populated, matching `event_type`.
    """

    run_id: str = Field(
        ..., description="Identifier of the agent run this event belongs to, matching AgentState.run_id."
    )
    sequence: int = Field(
        ..., ge=0, description="0-indexed position of this event within its run's ordered event sequence."
    )
    timestamp: datetime = Field(..., description="When this event was recorded.")
    event_type: EventType = Field(
        ..., description="Which kind of event this is; determines which payload field below is populated."
    )
    proposed_action: ProposedAction | None = Field(
        default=None,
        description="Populated when event_type is PROPOSED: the raw, untrusted action the LLM proposed.",
    )
    authorization_result: AuthorizationResult | None = Field(
        default=None,
        description=(
            "Populated when event_type is AUTHORIZED: the policy layer's decision for the proposal, "
            "including which rule_id denied it (if denied) or the resulting AuthorizedCommand (if approved)."
        ),
    )
    execution_result: ExecutionResult | None = Field(
        default=None,
        description=(
            "Populated when event_type is EXECUTED: the outcome of one execution attempt, including its "
            "1-indexed attempt number and verification outcome (verified flag / error reason)."
        ),
    )
    run_status: RunStatus | None = Field(
        default=None,
        description="Populated when event_type is RUN_FINISHED: the run's final status (e.g. COMPLETED/DENIED/FAILED).",
    )


class Tracer:
    """Records and retrieves `TraceEvent`s per run_id, in memory, for the lifetime of this instance."""

    def __init__(self) -> None:
        self._events_by_run: dict[str, list[TraceEvent]] = {}

    def _record(self, run_id: str, event_type: EventType, **payload) -> TraceEvent:
        events = self._events_by_run.setdefault(run_id, [])
        event = TraceEvent(
            run_id=run_id,
            sequence=len(events),
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            **payload,
        )
        events.append(event)
        return event

    def record_proposal(self, run_id: str, proposed_action: ProposedAction) -> TraceEvent:
        """Record that `proposed_action` was proposed for `run_id`."""
        return self._record(run_id, EventType.PROPOSED, proposed_action=proposed_action)

    def record_authorization(self, run_id: str, authorization_result: AuthorizationResult) -> TraceEvent:
        """Record the policy layer's decision on the most recent proposal for `run_id`."""
        return self._record(run_id, EventType.AUTHORIZED, authorization_result=authorization_result)

    def record_execution(self, run_id: str, execution_result: ExecutionResult) -> TraceEvent:
        """Record the outcome of one execution attempt for `run_id`.

        Call once per attempt if a caller wants per-attempt visibility
        into retries — `execution_result.attempt` distinguishes them.
        """
        return self._record(run_id, EventType.EXECUTED, execution_result=execution_result)

    def record_run_finished(self, run_id: str, run_status: RunStatus) -> TraceEvent:
        """Record the final `run_status` a run ended in."""
        return self._record(run_id, EventType.RUN_FINISHED, run_status=run_status)

    def events_for_run(self, run_id: str) -> list[TraceEvent]:
        """Return the events recorded for `run_id`, in the order they were recorded.

        Returns an empty list for a `run_id` with no recorded events,
        rather than raising — an unknown or not-yet-started run has an
        empty trace, not an error.
        """
        return list(self._events_by_run.get(run_id, []))

    def run_ids(self) -> list[str]:
        """All run_ids with at least one recorded event, in the order first seen."""
        return list(self._events_by_run.keys())
