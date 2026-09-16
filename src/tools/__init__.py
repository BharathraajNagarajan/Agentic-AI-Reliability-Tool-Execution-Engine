"""Tools layer.

Intended responsibility: domain tool definitions for support-ticket
operations, split conceptually into read-only tools (e.g. lookups,
inspection) and mutating tools (e.g. state-changing ticket operations).

The data contracts for this layer (`Ticket`, the untrusted `ProposedAction`
shape, and the trusted `AuthorizedCommand` shape plus per-tool argument
models) are defined in `src.tools.schemas`. The pure tool functions that
apply an already-authorized command to a `Ticket` (`update_ticket_status`,
`issue_refund`, `add_ticket_note`) are defined in `src.tools.functions`.
Retry/idempotency wrapping and any real I/O belong to the execution layer,
not here.
"""

from src.tools.functions import add_ticket_note, issue_refund, update_ticket_status
from src.tools.schemas import (
    AddTicketNoteArgs,
    AuthorizedAddTicketNoteCommand,
    AuthorizedCommand,
    AuthorizedCommandAdapter,
    AuthorizedIssueRefundCommand,
    AuthorizedUpdateTicketStatusCommand,
    IssueRefundArgs,
    ProposedAction,
    Ticket,
    TicketPriority,
    TicketStatus,
    ToolName,
    UpdateTicketStatusArgs,
)

__all__ = [
    "AddTicketNoteArgs",
    "AuthorizedAddTicketNoteCommand",
    "AuthorizedCommand",
    "AuthorizedCommandAdapter",
    "AuthorizedIssueRefundCommand",
    "AuthorizedUpdateTicketStatusCommand",
    "IssueRefundArgs",
    "ProposedAction",
    "Ticket",
    "TicketPriority",
    "TicketStatus",
    "ToolName",
    "UpdateTicketStatusArgs",
    "add_ticket_note",
    "issue_refund",
    "update_ticket_status",
]
