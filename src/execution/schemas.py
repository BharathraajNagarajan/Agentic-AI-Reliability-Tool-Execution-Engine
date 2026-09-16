"""Execution layer data contracts.

`ExecutionStatus`/`ExecutionResult` used to live in `src.graph.state`
(mirroring how `AuthorizationResult` originally lived there too) since
`AgentState` needs a typed slot for "the current execution result". The
execution layer owns their meaning, so — following the same move already
done for `AuthorizationResult` into `src.policy.schemas` — they live here,
and `src.graph.state` imports them.

This module intentionally does not import anything from `src.graph`: the
dependency runs one way (graph depends on execution's data shapes, not
the other way around), which is what keeps `src.execution.executor`
(which needs `ExecutionResult`) and `src.graph.nodes`/`src.graph.build`
(which need `src.execution.executor.ExecutionWrapper`) from forming an
import cycle.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExecutionStatus(str, Enum):
    """Outcome status of attempting to run an authorized command."""

    SUCCESS = "success"
    FAILURE = "failure"
    VERIFICATION_FAILED = "verification_failed"


class ExecutionResult(BaseModel):
    """Outcome of the execution layer running an `AuthorizedCommand`."""

    status: ExecutionStatus = Field(..., description="Result of the execution attempt.")
    output: dict[str, Any] | None = Field(
        default=None, description="Raw result payload returned by the tool call, if any."
    )
    error: str | None = Field(default=None, description="Error message when status is not SUCCESS.")
    verified: bool = Field(
        default=False,
        description="Whether post-execution verification confirmed the intended effect actually took place.",
    )
    attempt: int = Field(
        default=1, ge=1, description="Which retry attempt (1-indexed) this execution result corresponds to."
    )
