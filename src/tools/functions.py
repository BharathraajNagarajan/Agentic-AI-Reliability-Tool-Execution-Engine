"""Tool functions that execute an `AuthorizedCommand` against a `Ticket`.

Each function here takes the TRUSTED, already-validated command produced
by `src.policy.interface.authorize()` (never a raw `ProposedAction`) plus
the current `Ticket` state, and returns the resulting `Ticket`.

These are pure state-transition functions: no I/O, no persistence, no
retry/idempotency handling, and no policy decisions (the command has
already been authorized by the time it reaches here). They do not mutate
the `Ticket` passed in — they return a new instance via `model_copy`. The
one exception to strict purity is reading the wall clock to stamp
`updated_at`, which is unavoidable if that field is to mean anything;
callers who need deterministic timestamps in tests can inspect the
returned value relative to the input rather than pin the current value.

Wiring retries, idempotency keys, and actually calling out to any real
ticketing/payment system belongs to the execution layer, not here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from src.tools.schemas import (
    AuthorizedAddTicketNoteCommand,
    AuthorizedIssueRefundCommand,
    AuthorizedUpdateTicketStatusCommand,
    Ticket,
)


def _ensure_ticket_id_matches(command_ticket_id: str, ticket: Ticket) -> None:
    """Guard against applying a command to the wrong ticket.

    This is a contract check, not a policy/business rule: the caller is
    responsible for looking up `ticket` by `command_ticket_id` before
    calling a tool function, and this only makes a mismatch loud instead
    of silently mutating an unrelated ticket.
    """
    if command_ticket_id != ticket.id:
        raise ValueError(
            f"Command targets ticket '{command_ticket_id}' but was applied to ticket '{ticket.id}'."
        )


def update_ticket_status(command: AuthorizedUpdateTicketStatusCommand, ticket: Ticket) -> Ticket:
    """Apply a validated `update_ticket_status` command, returning the updated ticket.

    Sets `ticket.status` to `command.args.new_status` and bumps `updated_at`.
    """
    _ensure_ticket_id_matches(command.args.ticket_id, ticket)
    return ticket.model_copy(
        update={
            "status": command.args.new_status,
            "updated_at": datetime.now(timezone.utc),
        }
    )


def issue_refund(command: AuthorizedIssueRefundCommand, ticket: Ticket) -> Ticket:
    """Apply a validated `issue_refund` command, returning the updated ticket.

    Reduces `refund_eligible_amount` by the refunded amount (floored at
    zero) to reflect that this amount has now been consumed, and bumps
    `updated_at`. Does not move any money — that belongs to a payment
    integration in the execution layer, not this state-transition function.
    """
    _ensure_ticket_id_matches(command.args.ticket_id, ticket)
    remaining_eligible_amount = ticket.refund_eligible_amount
    if remaining_eligible_amount is not None:
        remaining_eligible_amount = max(remaining_eligible_amount - command.args.amount, Decimal("0"))
    return ticket.model_copy(
        update={
            "refund_eligible_amount": remaining_eligible_amount,
            "updated_at": datetime.now(timezone.utc),
        }
    )


def add_ticket_note(command: AuthorizedAddTicketNoteCommand, ticket: Ticket) -> Ticket:
    """Apply a validated `add_ticket_note` command, returning the updated ticket.

    Appends `command.args.note` to `ticket.notes` and bumps `updated_at`.
    """
    _ensure_ticket_id_matches(command.args.ticket_id, ticket)
    return ticket.model_copy(
        update={
            "notes": [*ticket.notes, command.args.note],
            "updated_at": datetime.now(timezone.utc),
        }
    )
