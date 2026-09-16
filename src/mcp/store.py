"""A minimal in-memory ticket store for the standalone MCP tool-execution server.

MCP tool calls arrive as just a tool name plus arguments — there is no
ambient `Ticket` state the way `AgentState.ticket` supplies one to
`src.graph.nodes.execute_node`. Since every tool function in
`src.tools.functions` needs a `Ticket` to operate on, this module gives
the MCP server (`src.mcp.server`) somewhere to look one up by the
`ticket_id` every tool's arguments carry, and somewhere to write the
updated `Ticket` back.

This is a new, standalone concern scoped to this MCP server only: it does
not read from, write to, or otherwise touch any existing ticket-handling
code (`src.execution`, `src.graph`, or any other layer's state). A
`TicketStore` is a plain in-memory dict, in the same spirit as
`ExecutionWrapper.idempotency_store` and `Tracer`'s per-run event dict —
no database, no file I/O, gone once the instance is garbage collected.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from src.tools.schemas import Ticket, TicketStatus


class TicketStore:
    """An in-memory `Ticket` store keyed by `Ticket.id`, for this MCP server's exclusive use."""

    def __init__(self, tickets: dict[str, Ticket] | None = None) -> None:
        self._tickets: dict[str, Ticket] = dict(tickets) if tickets else {}

    def get(self, ticket_id: str) -> Ticket | None:
        """Return the `Ticket` stored under `ticket_id`, or `None` if there isn't one."""
        return self._tickets.get(ticket_id)

    def put(self, ticket: Ticket) -> None:
        """Store `ticket`, keyed by its own `id` (overwriting any prior ticket with that id)."""
        self._tickets[ticket.id] = ticket

    def ticket_ids(self) -> list[str]:
        """All ticket ids currently in the store, for test/debug introspection."""
        return list(self._tickets.keys())


def seed_sample_tickets() -> dict[str, Ticket]:
    """Build the fixed set of sample tickets a fresh `TicketStore` is seeded with.

    A function rather than a module-level constant so each call produces
    fresh `Ticket` instances (Pydantic models are mutable, and stores
    should not accidentally share instances with each other).
    """
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    updated_at = datetime(2026, 1, 2, tzinfo=timezone.utc)

    ticket_one = Ticket(
        id="T-1",
        status=TicketStatus.OPEN,
        customer_ref="cust-123",
        subject="Package arrived damaged",
        order_ref="ORD-456",
        refund_eligible_amount=Decimal("50.00"),
        currency="USD",
        created_at=created_at,
        updated_at=updated_at,
    )
    ticket_two = Ticket(
        id="T-2",
        status=TicketStatus.IN_PROGRESS,
        customer_ref="cust-456",
        subject="Wrong item shipped",
        order_ref="ORD-789",
        refund_eligible_amount=Decimal("15.00"),
        currency="USD",
        created_at=created_at,
        updated_at=updated_at,
    )
    return {ticket.id: ticket for ticket in (ticket_one, ticket_two)}


def build_default_store() -> TicketStore:
    """Build a fresh `TicketStore` seeded with the default sample tickets (`T-1`, `T-2`)."""
    return TicketStore(seed_sample_tickets())
