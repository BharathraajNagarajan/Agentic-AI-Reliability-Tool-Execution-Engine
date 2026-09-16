"""Policy layer data contracts.

This module defines the data shapes the policy layer produces and consumes:

- ``AuthorizationDecision`` / ``AuthorizationResult`` are the outcome of
  running a ``ProposedAction`` through policy — approved (with a trusted
  ``AuthorizedCommand``) or denied (with a reason). These used to live in
  ``src.graph.state`` since ``AgentState`` needs a typed slot for them, but
  the policy layer owns their meaning, so they live here and
  ``src.graph.state`` imports them.
- ``RuleOutcome`` is the minimal data representation of "a policy rule's
  verdict on one proposed action": a rule id, a human-readable description,
  a pass/fail boolean, and a reason. No actual rule logic is defined here —
  only the shape a rule's result takes.
- ``AuthorizationContext`` is a deliberately open-ended bag of extra
  context (beyond the ticket) that a rule might need, so that adding real
  rules later does not require changing the ``authorize()`` signature.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from src.tools.schemas import AuthorizedCommand


class AuthorizationDecision(str, Enum):
    """Outcome of running a proposed action through the policy layer."""

    APPROVED = "approved"
    DENIED = "denied"


class AuthorizationResult(BaseModel):
    """Outcome of the policy layer evaluating a `ProposedAction`."""

    decision: AuthorizationDecision = Field(..., description="Whether the proposed action was approved or denied.")
    authorized_command: AuthorizedCommand | None = Field(
        default=None,
        description="The validated, trusted command to execute. Set only when decision is APPROVED.",
    )
    reason: str = Field(
        ..., description="Human-readable explanation for the decision, especially important when denied."
    )
    rule_id: str | None = Field(
        default=None,
        description="Identifier of the policy rule that produced this decision, for audit/traceability.",
    )


class RuleOutcome(BaseModel):
    """The verdict a single policy rule reached for one proposed action.

    This is only a data shape — no rule logic is implemented here. Real
    rules (refund thresholds, status-transition tables, role checks, etc.)
    will each produce one of these when they run.
    """

    rule_id: str = Field(..., min_length=1, description="Stable identifier of the rule that produced this outcome.")
    description: str = Field(
        ..., min_length=1, description="Human-readable description of what the rule checks."
    )
    passed: bool = Field(..., description="Whether the proposed action satisfied this rule.")
    reason: str = Field(
        ...,
        description=(
            "Human-readable explanation of the verdict, populated whether the "
            "rule passed or failed, useful for audit trails either way."
        ),
    )


class AuthorizationContext(BaseModel):
    """Extra context beyond the ticket that a policy rule might need to decide.

    Deliberately minimal and open-ended: as real rules are added (e.g. role
    checks, per-day spend limits) they can read whatever fields they need
    from here, or this model can grow new typed fields, without changing
    the `authorize()` signature.
    """

    model_config = ConfigDict(extra="allow")

    requested_by: str | None = Field(
        default=None, description="Identifier of the actor (operator, automated run, etc.) who initiated this run."
    )
