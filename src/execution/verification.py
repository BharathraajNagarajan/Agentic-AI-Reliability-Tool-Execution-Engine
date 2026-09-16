"""Post-execution verification: did an executed command actually take effect?

`verify()` compares the ticket state before and after a tool function ran
against what the `AuthorizedCommand` says should have happened, and
reports whether the ticket actually reflects that effect. This exists to
populate `ExecutionResult.verified` (currently hardcoded to its default of
`False` everywhere it's constructed) — but it is NOT wired into
`ExecutionWrapper.execute()` yet. That integration, and any policy for
what happens when verification fails (retry again? escalate? something
else?), is a separate decision for a later phase.

Verification here is a structural check only: it re-derives the expected
`ticket_after` state from `ticket_before` and `command` using the exact
same rules the pure tool functions in `src.tools.functions` use (e.g. the
floor-at-zero clamp for refunds), and compares. It is not a policy
decision and does not re-run any business rule.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Callable

from pydantic import BaseModel, Field

from src.tools.schemas import (
    AuthorizedAddTicketNoteCommand,
    AuthorizedCommand,
    AuthorizedIssueRefundCommand,
    AuthorizedUpdateTicketStatusCommand,
    Ticket,
)


class VerificationOutcome(BaseModel):
    """Result of verifying that an executed command's intended effect actually took place."""

    verified: bool = Field(
        ..., description="Whether ticket_after reflects the effect command was supposed to have."
    )
    reason: str = Field(
        ...,
        description="Human-readable explanation of the outcome, populated whether verification passed or failed.",
    )


def _verify_update_ticket_status(
    command: AuthorizedUpdateTicketStatusCommand, ticket_before: Ticket, ticket_after: Ticket
) -> VerificationOutcome:
    if ticket_after.status != command.args.new_status:
        return VerificationOutcome(
            verified=False,
            reason=(
                f"Expected ticket status to be '{command.args.new_status.value}' after execution, "
                f"but found '{ticket_after.status.value}'."
            ),
        )
    return VerificationOutcome(
        verified=True, reason=f"Ticket status is '{command.args.new_status.value}' as expected."
    )


def _verify_issue_refund(
    command: AuthorizedIssueRefundCommand, ticket_before: Ticket, ticket_after: Ticket
) -> VerificationOutcome:
    if ticket_before.refund_eligible_amount is None:
        expected_after = None
    else:
        expected_after = max(ticket_before.refund_eligible_amount - command.args.amount, Decimal("0"))

    if ticket_after.refund_eligible_amount != expected_after:
        return VerificationOutcome(
            verified=False,
            reason=(
                f"Expected refund_eligible_amount to be {expected_after} "
                f"(was {ticket_before.refund_eligible_amount}, refund {command.args.amount}, floored at 0), "
                f"but found {ticket_after.refund_eligible_amount}."
            ),
        )
    return VerificationOutcome(
        verified=True,
        reason=(
            f"refund_eligible_amount decreased from {ticket_before.refund_eligible_amount} to "
            f"{ticket_after.refund_eligible_amount} as expected."
        ),
    )


def _verify_add_ticket_note(
    command: AuthorizedAddTicketNoteCommand, ticket_before: Ticket, ticket_after: Ticket
) -> VerificationOutcome:
    if command.args.note not in ticket_after.notes:
        return VerificationOutcome(
            verified=False,
            reason=f"Expected note '{command.args.note}' to appear in ticket.notes, but it was not found.",
        )
    return VerificationOutcome(verified=True, reason=f"Note '{command.args.note}' found in ticket.notes.")


_VERIFIERS: dict[type, Callable[[AuthorizedCommand, Ticket, Ticket], VerificationOutcome]] = {
    AuthorizedUpdateTicketStatusCommand: _verify_update_ticket_status,
    AuthorizedIssueRefundCommand: _verify_issue_refund,
    AuthorizedAddTicketNoteCommand: _verify_add_ticket_note,
}
"""Dispatch table from concrete AuthorizedCommand type to the check that verifies its effect."""


def verify(command: AuthorizedCommand, ticket_before: Ticket, ticket_after: Ticket) -> VerificationOutcome:
    """Verify that `ticket_after` reflects the effect `command` was authorized to have on `ticket_before`.

    First confirms `ticket_after` is actually the same ticket (`id` match)
    as `ticket_before` — verifying a command against an unrelated ticket is
    meaningless regardless of which fields happen to match. Then dispatches
    to the tool-specific check for `command`'s concrete type.
    """
    if ticket_after.id != ticket_before.id:
        return VerificationOutcome(
            verified=False,
            reason=(
                f"ticket_after.id ('{ticket_after.id}') does not match ticket_before.id "
                f"('{ticket_before.id}'); cannot verify a mismatched ticket identity."
            ),
        )

    verifier = _VERIFIERS.get(type(command))
    if verifier is None:
        return VerificationOutcome(
            verified=False, reason=f"No verifier registered for command type '{type(command).__name__}'."
        )
    return verifier(command, ticket_before, ticket_after)
