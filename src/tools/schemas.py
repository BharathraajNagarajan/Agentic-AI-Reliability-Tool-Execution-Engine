"""Ticket and tool-call data contracts for the support-ticket ops agent.

This module defines the shapes that cross the LLM-proposal / system-command
trust boundary described in the policy layer:

- ``Ticket`` is the domain object the agent works against.
- ``ProposedAction`` is the UNTRUSTED shape of whatever the LLM emits: a
  tool name plus loosely-typed arguments. It must accept malformed or
  incomplete data, because that is exactly what an LLM can produce.
- ``AuthorizedCommand`` is the TRUSTED shape that exists only after the
  (not-yet-implemented) policy layer has validated a ``ProposedAction``
  against per-tool argument schemas. It is a discriminated union of
  concrete, strongly-typed command models, deliberately incompatible with
  ``ProposedAction`` so the type system enforces the trust boundary: you
  cannot pass a raw LLM proposal anywhere an ``AuthorizedCommand`` is
  expected without going through validation.

No tool functions (e.g. the code that actually mutates a ticket) and no
policy/authorization logic are defined here — only the data shapes.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class TicketStatus(str, Enum):
    """Lifecycle states a support ticket can be in."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    PENDING_CUSTOMER = "pending_customer"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketPriority(str, Enum):
    """Operator-facing urgency level for a ticket."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class Ticket(BaseModel):
    """A support ticket the agent is working on.

    Deliberately minimal: this is not a full CRM record, only the fields a
    support-ops agent needs to reason about and act on a ticket.
    """

    id: str = Field(..., description="Unique identifier of the ticket in the ticketing system.")
    status: TicketStatus = Field(..., description="Current lifecycle status of the ticket.")
    priority: TicketPriority = Field(
        default=TicketPriority.NORMAL,
        description="Operator-facing urgency level used to prioritize handling.",
    )
    customer_ref: str = Field(
        ...,
        description=(
            "Opaque reference identifying the customer this ticket belongs to "
            "(e.g. account ID or CRM contact ID). Not a full customer profile."
        ),
    )
    subject: str = Field(..., description="Short human-readable summary of what the ticket is about.")
    order_ref: str | None = Field(
        default=None,
        description=(
            "Reference to the order/transaction this ticket relates to, if any. "
            "Required context for refund-relevant actions."
        ),
    )
    refund_eligible_amount: Decimal | None = Field(
        default=None,
        description=(
            "Maximum amount that may be refunded against this ticket's order, if "
            "applicable. None when the ticket has no refund-relevant order."
        ),
    )
    currency: str | None = Field(
        default=None,
        description="ISO 4217 currency code for `refund_eligible_amount`, if that field is set.",
    )
    created_at: datetime = Field(..., description="Timestamp the ticket was created.")
    updated_at: datetime = Field(..., description="Timestamp the ticket was last modified.")
    notes: list[str] = Field(
        default_factory=list,
        description="Internal notes attached to the ticket via add_ticket_note, in chronological order.",
    )


class ProposedAction(BaseModel):
    """The UNTRUSTED shape of an action the LLM proposes taking on a ticket.

    This represents "whatever the LLM said" and must therefore tolerate
    malformed, incomplete, or extra data: unknown tool names, missing
    arguments, wrong argument types, or unexpected extra fields. None of
    that should raise a validation error here — rejecting bad proposals is
    the policy layer's job, not this model's.
    """

    model_config = ConfigDict(extra="allow")

    tool_name: str = Field(
        ...,
        description=(
            "Name of the tool the LLM says it wants to call, as raw text. Not "
            "constrained to known tool names, since the LLM may hallucinate one."
        ),
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Raw, loosely-typed arguments the LLM supplied for the tool call. "
            "May be missing required keys, contain wrong types, or include "
            "keys that don't correspond to any known tool argument."
        ),
    )
    raw_response: str | None = Field(
        default=None,
        description=(
            "Optional raw text of the LLM's response/tool-call payload this "
            "proposal was parsed from, kept for debugging and audit trails."
        ),
    )


class ToolName(str, Enum):
    """Enumerates the known tool names that can appear in an AuthorizedCommand.

    This is the trusted vocabulary of tools the system actually knows how to
    execute; it is intentionally a stricter, closed set compared to
    ``ProposedAction.tool_name``, which accepts arbitrary text.
    """

    UPDATE_TICKET_STATUS = "update_ticket_status"
    ISSUE_REFUND = "issue_refund"
    ADD_TICKET_NOTE = "add_ticket_note"


class UpdateTicketStatusArgs(BaseModel):
    """Validated arguments for the `update_ticket_status` tool."""

    ticket_id: str = Field(..., min_length=1, description="ID of the ticket whose status is being changed.")
    new_status: TicketStatus = Field(..., description="Status to transition the ticket to.")
    reason: str = Field(..., min_length=1, description="Human-readable justification for the status change.")


class IssueRefundArgs(BaseModel):
    """Validated arguments for the `issue_refund` tool."""

    ticket_id: str = Field(..., min_length=1, description="ID of the ticket the refund is being issued against.")
    amount: Decimal = Field(..., gt=0, description="Refund amount. Must be strictly positive.")
    currency: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="ISO 4217 currency code the refund amount is denominated in.",
    )
    reason: str = Field(..., min_length=1, description="Human-readable justification for the refund.")


class AddTicketNoteArgs(BaseModel):
    """Validated arguments for the `add_ticket_note` tool."""

    ticket_id: str = Field(..., min_length=1, description="ID of the ticket the note is being attached to.")
    note: str = Field(..., min_length=1, description="Internal note text to attach to the ticket.")


class AuthorizedUpdateTicketStatusCommand(BaseModel):
    """A system-approved `update_ticket_status` command with validated arguments."""

    tool_name: Literal[ToolName.UPDATE_TICKET_STATUS] = Field(
        default=ToolName.UPDATE_TICKET_STATUS,
        description="Discriminator identifying this as an update_ticket_status command.",
    )
    args: UpdateTicketStatusArgs = Field(..., description="Validated, strongly-typed arguments for this tool.")


class AuthorizedIssueRefundCommand(BaseModel):
    """A system-approved `issue_refund` command with validated arguments."""

    tool_name: Literal[ToolName.ISSUE_REFUND] = Field(
        default=ToolName.ISSUE_REFUND,
        description="Discriminator identifying this as an issue_refund command.",
    )
    args: IssueRefundArgs = Field(..., description="Validated, strongly-typed arguments for this tool.")


class AuthorizedAddTicketNoteCommand(BaseModel):
    """A system-approved `add_ticket_note` command with validated arguments."""

    tool_name: Literal[ToolName.ADD_TICKET_NOTE] = Field(
        default=ToolName.ADD_TICKET_NOTE,
        description="Discriminator identifying this as an add_ticket_note command.",
    )
    args: AddTicketNoteArgs = Field(..., description="Validated, strongly-typed arguments for this tool.")


AuthorizedCommand = Annotated[
    Union[
        AuthorizedUpdateTicketStatusCommand,
        AuthorizedIssueRefundCommand,
        AuthorizedAddTicketNoteCommand,
    ],
    Field(discriminator="tool_name"),
]
"""The TRUSTED shape of a command approved for execution.

This only exists on the far side of policy + argument-schema validation: it
is a discriminated union keyed on `tool_name`, where each variant pins
`args` to that tool's specific, strongly-typed argument model. Unlike
`ProposedAction`, arbitrary/malformed data does not fit this type — that is
the point. Construct or validate instances via `AuthorizedCommandAdapter`.
"""

AuthorizedCommandAdapter: TypeAdapter[Any] = TypeAdapter(AuthorizedCommand)
"""Reusable `TypeAdapter` for validating/parsing data into an `AuthorizedCommand`.

Needed because `AuthorizedCommand` is a `Union` type alias, not a class, so
it has no `.model_validate()` of its own.
"""
