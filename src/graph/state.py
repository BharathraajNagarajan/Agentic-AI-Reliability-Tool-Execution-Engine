"""Shared LangGraph state schema for the support-ticket ops agent run loop.

This module defines the single state object that flows between graph nodes
for one run of the agent (propose -> authorize -> execute -> verify, with
retries). Node implementations do not exist yet; this only fixes the shape
of the data those nodes will read and write, so the loop's data contracts
can be reviewed and tested independently of graph wiring.

`AuthorizationResult` lives in `src.policy.schemas` and `ExecutionResult`
lives in `src.execution.schemas`, since those layers own their meaning;
both are imported here only because `AgentState` needs typed slots for
"the current authorization result" and "the current execution result". No
authorization or execution logic is implemented here.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from src.execution.schemas import ExecutionResult, ExecutionStatus
from src.policy.schemas import AuthorizationDecision, AuthorizationResult
from src.tools.schemas import ProposedAction, Ticket


class ConversationMessage(BaseModel):
    """A single turn in the task/conversation context accumulated during a run."""

    role: Literal["user", "assistant", "system", "tool"] = Field(
        ..., description="Who produced this message: the requester, the LLM, the system, or a tool result."
    )
    content: str = Field(..., description="Text content of the message.")
    timestamp: datetime = Field(..., description="When this message was recorded.")


class RunStatus(str, Enum):
    """Coarse-grained status of an agent run, used for control flow and reporting."""

    PENDING = "pending"
    PROPOSING = "proposing"
    AUTHORIZING = "authorizing"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    RETRYING = "retrying"
    COMPLETED = "completed"
    DENIED = "denied"
    FAILED = "failed"


class AgentState(BaseModel):
    """Single shared state object passed between nodes for one agent run.

    LangGraph nodes are expected to read this state and return partial
    updates (e.g. a dict of the fields they changed) rather than mutating
    it directly; this model documents the canonical shape of that state and
    provides validation when constructing or checkpointing it.
    """

    run_id: str = Field(..., description="Unique identifier for this agent run, used for tracing and idempotency.")
    ticket: Ticket = Field(..., description="The support ticket currently being worked by this run.")
    task_description: str = Field(
        ..., description="The instruction/request that initiated this run (e.g. the operator's ask)."
    )
    conversation_history: list[ConversationMessage] = Field(
        default_factory=list,
        description="Turn-by-turn record of messages exchanged so far while working this ticket.",
    )
    proposed_action: ProposedAction | None = Field(
        default=None,
        description="The most recent untrusted action proposed by the LLM, or None if none has been proposed yet.",
    )
    authorization_result: AuthorizationResult | None = Field(
        default=None,
        description="Outcome of the most recent policy check on `proposed_action`, or None if not yet evaluated.",
    )
    execution_result: ExecutionResult | None = Field(
        default=None,
        description="Outcome of the most recent execution attempt, or None if nothing has been executed yet.",
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of times the current proposed action has been retried after failure.",
    )
    run_status: RunStatus = Field(
        default=RunStatus.PENDING,
        description="Coarse-grained status of the overall run, used to drive graph control flow.",
    )
